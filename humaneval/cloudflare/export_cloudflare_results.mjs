import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const [outArg, cloudflareArg, pnpmArg] = process.argv.slice(2);
if (!outArg || !cloudflareArg || !pnpmArg) {
  throw new Error("usage: node export_cloudflare_results.mjs OUT_DIR CLOUDFLARE_DIR PNPM_PATH");
}

const outDir = resolve(outArg);
const cloudflareDir = resolve(cloudflareArg);
mkdirSync(dirname(outDir), { recursive: true, mode: 0o700 });
mkdirSync(outDir, { mode: 0o700 });

const query = `
SELECT r.*, i.source_id, i.split,
  i.target_content_band, i.target_organization_band,
  i.target_expression_band, i.first_source
FROM responses AS r
JOIN study_items AS i USING(item_index)
ORDER BY r.user_name, r.item_index;
SELECT COUNT(*) AS study_items FROM study_items;
`;

const command = spawnSync(
  pnpmArg,
  ["exec", "wrangler", "d1", "execute", "DB", "--remote", "--json", "--command", query],
  {
    cwd: cloudflareDir,
    encoding: "utf8",
    env: { ...process.env, WRANGLER_LOG_PATH: "/private/tmp/mal2026-final-export.log" },
    maxBuffer: 16 * 1024 * 1024,
  },
);
if (command.status !== 0) {
  throw new Error(`wrangler export query failed: ${command.stderr || command.stdout}`);
}

const resultSets = JSON.parse(command.stdout);
const rows = resultSets.at(0)?.results;
const studyItems = Number(resultSets.at(1)?.results?.at(0)?.study_items);
if (!Array.isArray(rows) || studyItems !== 20) {
  throw new Error("unexpected D1 export result shape");
}

const allowedUsers = ["명훈", "찬희", "정호", "지민"];
const allowedSources = new Set(["api", "model"]);
const allowedVerdicts = new Set(["appropriate", "partial", "inappropriate"]);
const seen = new Set();
for (const row of rows) {
  const key = `${row.user_name}\u0000${row.item_index}`;
  if (seen.has(key)) throw new Error(`duplicate response key: ${key}`);
  seen.add(key);
  if (!allowedUsers.includes(row.user_name)) throw new Error(`unexpected user: ${row.user_name}`);
  if (!Number.isInteger(row.item_index) || row.item_index < 0 || row.item_index >= studyItems) {
    throw new Error(`invalid item index: ${row.item_index}`);
  }
  for (const axis of ["content", "organization", "expression"]) {
    if (!Number.isInteger(row[`${axis}_score`]) || row[`${axis}_score`] < 1 || row[`${axis}_score`] > 5) {
      throw new Error(`invalid ${axis} score for ${key}`);
    }
    if (!Number.isInteger(row[`target_${axis}_band`]) || row[`target_${axis}_band`] < 1 || row[`target_${axis}_band`] > 5) {
      throw new Error(`invalid target ${axis} band for ${key}`);
    }
    for (const suffix of ["a", "b"]) {
      if (!allowedVerdicts.has(row[`rationale_${suffix}_${axis}_verdict`])) {
        throw new Error(`invalid rationale verdict for ${key}`);
      }
    }
  }
  if (!row.score_submitted_at || !row.rationale_a_submitted_at || !row.rationale_b_submitted_at) {
    throw new Error(`partial response row: ${key}`);
  }
  if (row.score_submitted_at > row.rationale_a_submitted_at || row.rationale_a_submitted_at > row.rationale_b_submitted_at) {
    throw new Error(`non-monotonic response timestamps: ${key}`);
  }
  if (!allowedSources.has(row.rationale_a_source) || !allowedSources.has(row.rationale_b_source)) {
    throw new Error(`invalid rationale source: ${key}`);
  }
  if (row.rationale_a_source === row.rationale_b_source) {
    throw new Error(`duplicate blinded rationale source: ${key}`);
  }
}

const canonical = (value) => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  }
  return value;
};
const responseText = rows.map((row) => JSON.stringify(canonical(row))).join("\n") + (rows.length ? "\n" : "");
const responsePath = resolve(outDir, "responses.jsonl");
writeFileSync(responsePath, responseText, { encoding: "utf8", mode: 0o600, flag: "wx" });
chmodSync(responsePath, 0o600);
const responseSha256 = createHash("sha256").update(responseText).digest("hex");

const participantCounts = Object.fromEntries(allowedUsers.map((name) => [name, 0]));
const participantCompleted = Object.fromEntries(allowedUsers.map((name) => [name, 0]));
const sourceVerdicts = {
  api: { appropriate: 0, partial: 0, inappropriate: 0 },
  model: { appropriate: 0, partial: 0, inappropriate: 0 },
};
let exact = 0;
let withinOne = 0;
let absoluteError = 0;
let scoreReasonSlots = 0;
let scoreReasonsFilled = 0;
let rationaleReasonSlots = 0;
let rationaleReasonsFilled = 0;
for (const row of rows) {
  participantCounts[row.user_name] += 1;
  participantCompleted[row.user_name] += Number(Boolean(row.rationale_b_submitted_at));
  for (const axis of ["content", "organization", "expression"]) {
    const delta = row[`${axis}_score`] - row[`target_${axis}_band`];
    exact += Number(delta === 0);
    withinOne += Number(Math.abs(delta) <= 1);
    absoluteError += Math.abs(delta);
    scoreReasonSlots += 1;
    scoreReasonsFilled += Number(Boolean(String(row[`${axis}_reason`] ?? "").trim()));
    for (const suffix of ["a", "b"]) {
      const source = row[`rationale_${suffix}_source`];
      sourceVerdicts[source][row[`rationale_${suffix}_${axis}_verdict`]] += 1;
      rationaleReasonSlots += 1;
      rationaleReasonsFilled += Number(Boolean(String(row[`rationale_${suffix}_${axis}_reason`] ?? "").trim()));
    }
  }
}
const axisJudgments = rows.length * 3;
const summary = canonical({
  axis_judgments: axisJudgments,
  completed_response_rows: rows.length,
  participant_completed: participantCompleted,
  participant_counts: participantCounts,
  rationale_reason_filled_pct: rationaleReasonSlots ? 100 * rationaleReasonsFilled / rationaleReasonSlots : null,
  rationale_source_verdict_counts: sourceVerdicts,
  score_agreement: {
    exact_pct: axisJudgments ? 100 * exact / axisJudgments : null,
    mean_absolute_error: axisJudgments ? absoluteError / axisJudgments : null,
    within_one_pct: axisJudgments ? 100 * withinOne / axisJudgments : null,
  },
  score_reason_filled_pct: scoreReasonSlots ? 100 * scoreReasonsFilled / scoreReasonSlots : null,
  study_items: studyItems,
});
const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
const summaryPath = resolve(outDir, "summary.json");
writeFileSync(summaryPath, summaryText, { encoding: "utf8", mode: 0o600, flag: "wx" });
chmodSync(summaryPath, 0o600);
const summarySha256 = createHash("sha256").update(summaryText).digest("hex");

const exportedAt = new Date().toISOString();
const manifest = canonical({
  collection_status: "point-in-time snapshot; public Worker was not stopped or modified",
  database_id: "dca817ac-ccea-4581-a080-818a100aaa77",
  excluded: ["essay text", "topic prompt", "API rationale text", "model rationale text", "sessions", "cookies", "password material", "CSRF tokens"],
  exported_at: exportedAt,
  files: {
    "responses.jsonl": { rows: rows.length, sha256: responseSha256 },
    "summary.json": { sha256: summarySha256 },
  },
  included: ["reviewer name", "item index", "human scores and optional reasons", "rationale verdicts and optional reasons", "submission timestamps", "opaque source ID", "split", "hidden target bands", "blinded source mapping"],
  schema_version: 1,
  source: "Cloudflare D1 production database for mal2026-human-validation",
  worker: "mal2026-human-validation",
});
const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
const manifestPath = resolve(outDir, "manifest.json");
writeFileSync(manifestPath, manifestText, { encoding: "utf8", mode: 0o600, flag: "wx" });
chmodSync(manifestPath, 0o600);
const manifestSha256 = createHash("sha256").update(manifestText).digest("hex");

const checksums = `${responseSha256}  responses.jsonl\n${summarySha256}  summary.json\n${manifestSha256}  manifest.json\n`;
const checksumPath = resolve(outDir, "SHA256SUMS");
writeFileSync(checksumPath, checksums, { encoding: "utf8", mode: 0o600, flag: "wx" });
chmodSync(checksumPath, 0o600);

process.stdout.write(JSON.stringify({ out_dir: outDir, rows: rows.length, study_items: studyItems, response_sha256: responseSha256 }) + "\n");
