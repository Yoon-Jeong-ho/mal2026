import assert from "node:assert/strict";
import test from "node:test";

import worker, {
  cleanReasons,
  constantTimeEqual,
  phaseFromRow,
  validTurnstileResult,
  validateScores,
} from "../src/index.js";

test("constant-time comparison returns the expected result", () => {
  assert.equal(constantTimeEqual("same", "same"), true);
  assert.equal(constantTimeEqual("same", "different"), false);
  assert.equal(constantTimeEqual("", ""), true);
});

test("phase follows the score and two rationale submissions", () => {
  assert.equal(phaseFromRow(null), "score");
  assert.equal(phaseFromRow({ score_submitted_at: "now" }), "rationale_a");
  assert.equal(phaseFromRow({ score_submitted_at: "now", rationale_a_submitted_at: "now" }), "rationale_b");
  assert.equal(phaseFromRow({
    score_submitted_at: "now",
    rationale_a_submitted_at: "now",
    rationale_b_submitted_at: "now",
  }), "complete");
});

test("score and reason validation is strict", () => {
  assert.equal(validateScores({ content: 1, organization: 3, expression: 5 }), true);
  assert.equal(validateScores({ content: 0, organization: 3, expression: 5 }), false);
  assert.equal(validateScores({ content: 1, organization: 3, expression: 5, extra: 2 }), false);
  assert.deepEqual(
    cleanReasons({ content: " a ", organization: "", expression: "c" }),
    { content: "a", organization: "", expression: "c" },
  );
  assert.throws(() => cleanReasons({ content: "a", organization: "b" }));
});

test("Turnstile result is bound to the production hostname and login action", () => {
  const valid = {
    success: true,
    hostname: "mal2026-human-validation.yoon300days.workers.dev",
    action: "login",
  };
  assert.equal(validTurnstileResult(valid, valid.hostname), true);
  assert.equal(validTurnstileResult({ ...valid, hostname: "example.com" }, valid.hostname), false);
  assert.equal(validTurnstileResult({ ...valid, action: "other" }, valid.hostname), false);
  assert.equal(validTurnstileResult({ ...valid, success: false }, valid.hostname), false);
});


test("public config excludes secrets and protected state requires a session", async () => {
  const env = { TURNSTILE_SITE_KEY: "public-key", TURNSTILE_SECRET: "synthetic-secret", PASSWORD_PEPPER: "synthetic-pepper" };
  const config = await worker.fetch(new Request("https://study.example/api/config"), env);
  assert.deepEqual(await config.json(), { turnstile_site_key: "public-key" });
  assert.equal(config.headers.get("Cache-Control"), "no-store");
  const state = await worker.fetch(new Request("https://study.example/api/state"), env);
  assert.equal(state.status, 401);
  assert.deepEqual(Object.keys(await state.json()), ["error"]);
});

test("cross-origin writes fail before touching bindings", async () => {
  const result = await worker.fetch(new Request("https://study.example/api/login", {
    method: "POST", headers: { Origin: "https://other.example", "Content-Type": "application/json" },
    body: JSON.stringify({ name: "정호", password: "synthetic-password" }),
  }), {});
  assert.equal(result.status, 403);
});

test("malformed or oversized bodies are rejected before authentication", async () => {
  for (const [body, status] of [["[]", 400], ["{", 400], [JSON.stringify({ value: "x".repeat(32768) }), 413]]) {
    const result = await worker.fetch(new Request("https://study.example/api/login", {
      method: "POST", headers: { Origin: "https://study.example", "Content-Type": "application/json" }, body,
    }), {});
    assert.equal(result.status, status);
  }
});

test("a session without the matching CSRF token cannot write", async () => {
  let lookups = 0;
  const env = {
    API_RATE: { limit: async () => ({ success: true }) },
    DB: { prepare(sql) {
      assert.match(sql, /^SELECT token_hash,user_name,csrf_token,expires_at FROM sessions/);
      lookups += 1;
      return { bind: () => ({ first: async () => ({ user_name: "정호", csrf_token: "synthetic-csrf" }) }) };
    } },
  };
  const response = await worker.fetch(new Request("https://study.example/api/score", {
    method: "POST", headers: { Origin: "https://study.example", "Content-Type": "application/json", Cookie: "hvsession=synthetic-token" },
    body: "{}",
  }), env);
  assert.equal(response.status, 403);
  assert.equal(lookups, 1);
});
