const USERS = new Set(["명훈", "찬희", "정호", "지민"]);
const AXES = ["content", "organization", "expression"];
const VERDICTS = new Set(["appropriate", "partial", "inappropriate"]);
const SESSION_TTL_SECONDS = 12 * 60 * 60;
const MAX_BODY_BYTES = 32_768;
const encoder = new TextEncoder();

class ApiError extends Error {
  constructor(status, message, headers = {}) {
    super(message);
    this.status = status;
    this.headers = headers;
  }
}

export function constantTimeEqual(left, right) {
  if (typeof left !== "string" || typeof right !== "string") return false;
  let mismatch = left.length ^ right.length;
  const size = Math.max(left.length, right.length);
  for (let index = 0; index < size; index += 1) {
    mismatch |= (left.charCodeAt(index % Math.max(left.length, 1)) || 0)
      ^ (right.charCodeAt(index % Math.max(right.length, 1)) || 0);
  }
  return mismatch === 0;
}

export function phaseFromRow(row) {
  if (!row || !row.score_submitted_at) return "score";
  if (!row.rationale_a_submitted_at) return "rationale_a";
  if (!row.rationale_b_submitted_at) return "rationale_b";
  return "complete";
}

function hasExactKeys(value, expected) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const keys = Object.keys(value).sort();
  return keys.length === expected.length
    && keys.every((key, index) => key === [...expected].sort()[index]);
}

export function validateScores(scores) {
  return hasExactKeys(scores, AXES)
    && AXES.every((axis) => Number.isInteger(scores[axis]) && scores[axis] >= 1 && scores[axis] <= 5);
}

export function cleanReasons(reasons) {
  if (!hasExactKeys(reasons, AXES)) throw new ApiError(400, "세 영역의 판단 이유가 필요합니다.");
  const cleaned = {};
  for (const axis of AXES) {
    if (typeof reasons[axis] !== "string") throw new ApiError(400, "판단 이유는 글자여야 합니다.");
    cleaned[axis] = reasons[axis].trim();
    if (cleaned[axis].length > 2000) throw new ApiError(400, "판단 이유는 2,000자 이하여야 합니다.");
  }
  return cleaned;
}

export function validTurnstileResult(result, expectedHostname) {
  return Boolean(result)
    && result.success === true
    && result.hostname === expectedHostname
    && result.action === "login";
}

function securityHeaders(contentType = "application/json; charset=utf-8") {
  return {
    "Cache-Control": "no-store",
    "Content-Type": contentType,
    "Content-Security-Policy": "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; style-src 'self'; img-src 'self'; connect-src 'self' https://challenges.cloudflare.com; frame-src https://challenges.cloudflare.com; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
  };
}

function json(value, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { ...securityHeaders(), ...extraHeaders },
  });
}

function error(status, message, headers = {}) {
  return json({ error: message }, status, headers);
}

function base64url(bytes) {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
}

function randomToken(size = 32) {
  return base64url(crypto.getRandomValues(new Uint8Array(size)));
}

async function sha256Hex(value) {
  const digest = await crypto.subtle.digest("SHA-256", encoder.encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function passwordDigest(password, pepper) {
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(pepper),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const digest = await crypto.subtle.sign("HMAC", key, encoder.encode(password));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function authenticatePassword(password, env) {
  if (!env.PASSWORD_PEPPER || !env.PASSWORD_DIGEST) return false;
  const observed = await passwordDigest(password, env.PASSWORD_PEPPER);
  return constantTimeEqual(observed, env.PASSWORD_DIGEST);
}

function cookieValue(request, name) {
  const cookie = request.headers.get("Cookie") || "";
  for (const part of cookie.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === name) return rest.join("=");
  }
  return "";
}

function sessionCookie(token, maxAge) {
  return `hvsession=${token}; Path=/; Max-Age=${maxAge}; HttpOnly; SameSite=Strict; Secure`;
}

function requestOriginAllowed(request) {
  const origin = request.headers.get("Origin");
  return Boolean(origin) && origin === new URL(request.url).origin;
}

async function readJson(request) {
  if (!(request.headers.get("Content-Type") || "").toLowerCase().startsWith("application/json")) {
    throw new ApiError(400, "요청 형식이 올바르지 않습니다.");
  }
  const declared = Number(request.headers.get("Content-Length") || "0");
  if (declared > MAX_BODY_BYTES) throw new ApiError(413, "요청 본문이 너무 큽니다.");
  const raw = await request.text();
  if (encoder.encode(raw).byteLength > MAX_BODY_BYTES) throw new ApiError(413, "요청 본문이 너무 큽니다.");
  let value;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new ApiError(400, "요청 JSON이 올바르지 않습니다.");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new ApiError(400, "요청 JSON은 객체여야 합니다.");
  }
  return value;
}

async function verifyTurnstile(token, request, env) {
  if (!env.TURNSTILE_SECRET || !env.TURNSTILE_SITE_KEY || !env.TURNSTILE_HOSTNAME) {
    throw new ApiError(503, "로그인 보안 설정이 완료되지 않았습니다.");
  }
  if (typeof token !== "string" || token.length < 10 || token.length > 2048) return false;
  const body = new FormData();
  body.set("secret", env.TURNSTILE_SECRET);
  body.set("response", token);
  const remoteIp = request.headers.get("CF-Connecting-IP");
  if (remoteIp) body.set("remoteip", remoteIp);
  const response = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    body,
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) return false;
  const result = await response.json();
  return validTurnstileResult(result, env.TURNSTILE_HOSTNAME);
}

async function applyRateLimit(binding, key, message) {
  if (!binding) throw new ApiError(503, "요청 제한 설정이 완료되지 않았습니다.");
  const result = await binding.limit({ key });
  if (!result.success) throw new ApiError(429, message, { "Retry-After": "60" });
}

async function sessionFor(request, env) {
  const token = cookieValue(request, "hvsession");
  if (!token || token.length > 128) return null;
  const tokenHash = await sha256Hex(token);
  const now = new Date().toISOString();
  const row = await env.DB.prepare(
    "SELECT token_hash,user_name,csrf_token,expires_at FROM sessions WHERE token_hash=? AND expires_at>?",
  ).bind(tokenHash, now).first();
  return row ? { ...row, token } : null;
}

async function nextRow(env, userName) {
  return env.DB.prepare(`
    SELECT i.item_index,i.source_id,i.split,i.topic_prompt,i.essay,
      i.api_rationale_json,i.model_rationale_json,i.first_source,
      r.content_score,r.organization_score,r.expression_score,
      r.score_submitted_at,r.rationale_a_submitted_at,r.rationale_b_submitted_at
    FROM study_items AS i
    LEFT JOIN responses AS r ON r.user_name=? AND r.item_index=i.item_index
    WHERE r.rationale_b_submitted_at IS NULL
    ORDER BY i.item_index
    LIMIT 1
  `).bind(userName).first();
}

async function stateFor(env, userName, csrfToken) {
  const next = await nextRow(env, userName);
  const [metaResult, totalResult, completedResult] = await env.DB.batch([
    env.DB.prepare("SELECT key,value FROM study_meta WHERE key IN ('common_notice','rubric_json','judge_guide_json')"),
    env.DB.prepare("SELECT COUNT(*) AS total FROM study_items"),
    env.DB.prepare("SELECT COUNT(*) AS completed FROM responses WHERE user_name=? AND rationale_b_submitted_at IS NOT NULL").bind(userName),
  ]);
  const meta = Object.fromEntries(metaResult.results.map((row) => [row.key, row.value]));
  if (!meta.common_notice || !meta.rubric_json || !meta.judge_guide_json) {
    throw new ApiError(503, "평가 데이터가 준비되지 않았습니다.");
  }
  const total = Number(totalResult.results[0]?.total || 0);
  const completed = Number(completedResult.results[0]?.completed || 0);
  const phase = next ? phaseFromRow(next) : "finished";
  const state = {
    user: userName,
    progress: { completed, total },
    phase,
    common_notice: meta.common_notice,
    rubric: JSON.parse(meta.rubric_json),
    judge_guide: JSON.parse(meta.judge_guide_json),
    csrf_token: csrfToken,
  };
  if (!next) return state;
  state.item = {
    index: Number(next.item_index),
    number: Number(next.item_index) + 1,
    topic_prompt: next.topic_prompt,
    essay: next.essay,
  };
  if (phase.startsWith("rationale")) {
    const source = phase === "rationale_a"
      ? next.first_source
      : (next.first_source === "api" ? "model" : "api");
    state.submitted_scores = Object.fromEntries(AXES.map((axis) => [axis, Number(next[`${axis}_score`])]));
    state.rationale = {
      label: phase === "rationale_a" ? "A" : "B",
      texts: JSON.parse(source === "api" ? next.api_rationale_json : next.model_rationale_json),
    };
  }
  return state;
}

async function login(request, env, payload) {
  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  await Promise.all([
    applyRateLimit(env.LOGIN_RATE, `login:${ip}`, "로그인 요청이 너무 많습니다. 잠시 뒤 다시 시도해 주세요."),
    applyRateLimit(env.LOGIN_GLOBAL_RATE, "login:global", "로그인 요청이 일시적으로 제한됐습니다."),
  ]);
  const name = typeof payload.name === "string" && payload.name.length <= 64 ? payload.name : "";
  const password = typeof payload.password === "string" && payload.password.length <= 256 ? payload.password : "";
  const challengeOk = await verifyTurnstile(payload.turnstile_token, request, env);
  const passwordOk = challengeOk && await authenticatePassword(password, env);
  if (!USERS.has(name) || !passwordOk) {
    throw new ApiError(401, "평가자 이름 또는 공통 비밀번호가 올바르지 않습니다.");
  }

  const token = randomToken();
  const tokenHash = await sha256Hex(token);
  const csrfToken = randomToken(24);
  const now = new Date();
  const expires = new Date(now.getTime() + SESSION_TTL_SECONDS * 1000);
  await env.DB.batch([
    env.DB.prepare("DELETE FROM sessions WHERE expires_at<=?").bind(now.toISOString()),
    env.DB.prepare("INSERT INTO sessions(token_hash,user_name,csrf_token,created_at,expires_at) VALUES (?,?,?,?,?)")
      .bind(tokenHash, name, csrfToken, now.toISOString(), expires.toISOString()),
  ]);
  return json(await stateFor(env, name, csrfToken), 200, {
    "Set-Cookie": sessionCookie(token, SESSION_TTL_SECONDS),
  });
}

async function recordScores(env, userName, payload) {
  if (!Number.isInteger(payload.item_index)) throw new ApiError(400, "문항 번호가 올바르지 않습니다.");
  if (!validateScores(payload.scores)) throw new ApiError(400, "세 영역의 점수를 모두 선택해 주세요.");
  const reasons = cleanReasons(payload.reasons);
  const next = await nextRow(env, userName);
  if (!next || Number(next.item_index) !== payload.item_index || phaseFromRow(next) !== "score") {
    throw new ApiError(409, "채점 제출 순서가 바뀌었습니다. 화면을 새로고침해 주세요.");
  }
  const now = new Date().toISOString();
  const result = await env.DB.prepare(`
    INSERT OR IGNORE INTO responses(
      user_name,item_index,content_score,organization_score,expression_score,
      content_reason,organization_reason,expression_reason,score_submitted_at
    ) VALUES (?,?,?,?,?,?,?,?,?)
  `).bind(
    userName,
    payload.item_index,
    payload.scores.content,
    payload.scores.organization,
    payload.scores.expression,
    reasons.content,
    reasons.organization,
    reasons.expression,
    now,
  ).run();
  if (Number(result.meta?.changes || 0) !== 1) throw new ApiError(409, "이미 처리된 채점입니다.");
}

async function recordRationale(env, userName, payload) {
  if (!Number.isInteger(payload.item_index)) throw new ApiError(400, "문항 번호가 올바르지 않습니다.");
  if (!hasExactKeys(payload.verdicts, AXES) || !AXES.every((axis) => VERDICTS.has(payload.verdicts[axis]))) {
    throw new ApiError(400, "세 영역의 적절성을 모두 판단해 주세요.");
  }
  const reasons = cleanReasons(payload.reasons);
  const next = await nextRow(env, userName);
  const phase = phaseFromRow(next);
  if (!next || Number(next.item_index) !== payload.item_index || !["rationale_a", "rationale_b"].includes(phase)) {
    throw new ApiError(409, "평가 설명 제출 순서가 바뀌었습니다. 화면을 새로고침해 주세요.");
  }
  const suffix = phase === "rationale_a" ? "a" : "b";
  const source = suffix === "a" ? next.first_source : (next.first_source === "api" ? "model" : "api");
  const prerequisite = suffix === "a"
    ? "score_submitted_at IS NOT NULL AND rationale_a_submitted_at IS NULL"
    : "rationale_a_submitted_at IS NOT NULL AND rationale_b_submitted_at IS NULL";
  const result = await env.DB.prepare(`
    UPDATE responses SET
      rationale_${suffix}_source=?,
      rationale_${suffix}_content_verdict=?,
      rationale_${suffix}_organization_verdict=?,
      rationale_${suffix}_expression_verdict=?,
      rationale_${suffix}_content_reason=?,
      rationale_${suffix}_organization_reason=?,
      rationale_${suffix}_expression_reason=?,
      rationale_${suffix}_submitted_at=?
    WHERE user_name=? AND item_index=? AND ${prerequisite}
  `).bind(
    source,
    payload.verdicts.content,
    payload.verdicts.organization,
    payload.verdicts.expression,
    reasons.content,
    reasons.organization,
    reasons.expression,
    new Date().toISOString(),
    userName,
    payload.item_index,
  ).run();
  if (Number(result.meta?.changes || 0) !== 1) throw new ApiError(409, "이미 처리된 평가 설명입니다.");
}

async function handleApi(request, env) {
  const url = new URL(request.url);
  if (request.method === "GET" && url.pathname === "/api/config") {
    return json({ turnstile_site_key: env.TURNSTILE_SITE_KEY || null });
  }
  if (request.method === "GET" && url.pathname === "/api/state") {
    const session = await sessionFor(request, env);
    if (!session) return error(401, "로그인이 필요합니다.");
    return json(await stateFor(env, session.user_name, session.csrf_token));
  }
  if (request.method !== "POST") return error(405, "허용되지 않은 요청 방식입니다.", { Allow: "GET, POST" });
  if (!requestOriginAllowed(request)) return error(403, "허용되지 않은 요청 출처입니다.");
  const payload = await readJson(request);
  if (url.pathname === "/api/login") return login(request, env, payload);

  const ip = request.headers.get("CF-Connecting-IP") || "unknown";
  await applyRateLimit(env.API_RATE, `api:${ip}`, "요청이 너무 많습니다. 잠시 뒤 다시 시도해 주세요.");
  const session = await sessionFor(request, env);
  if (!session) return error(401, "로그인이 필요합니다.");
  if (!constantTimeEqual(request.headers.get("X-CSRF-Token") || "", session.csrf_token)) {
    return error(403, "요청 확인 토큰이 올바르지 않습니다.");
  }
  if (url.pathname === "/api/logout") {
    await env.DB.prepare("DELETE FROM sessions WHERE token_hash=?").bind(session.token_hash).run();
    return json({ ok: true }, 200, { "Set-Cookie": sessionCookie("", 0) });
  }
  if (url.pathname === "/api/score") {
    await recordScores(env, session.user_name, payload);
    return json(await stateFor(env, session.user_name, session.csrf_token));
  }
  if (url.pathname === "/api/rationale") {
    await recordRationale(env, session.user_name, payload);
    return json(await stateFor(env, session.user_name, session.csrf_token));
  }
  return error(404, "찾을 수 없습니다.");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    try {
      if (request.method === "GET" && url.pathname === "/healthz") return json({ status: "ok" });
      if (url.pathname.startsWith("/api/")) return await handleApi(request, env);
      return env.ASSETS.fetch(request);
    } catch (cause) {
      if (cause instanceof ApiError) return error(cause.status, cause.message, cause.headers);
      return error(500, "서버 처리 중 오류가 발생했습니다.");
    }
  },
};
