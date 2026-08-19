# Terminal aggregate evaluation and next-stage ledger: 20260719-001

**Scope:** aggregate-only terminal evaluation of the completed train-only
judge-v2 pilot and the three completed scalar score heads. No raw essay,
identifier, explanation, request, response, prediction, credential, or
checkpoint content is recorded here.

## Observed judge-v2 result

- Run: `qwen36-judge-v2-pilot-20260720-001`; terminal status:
  `executed_failed_gates`.
- Reconciliation: 7,776 requests and 7,776 raw responses. A streaming audit
  retained only aggregation metadata, produced the same aggregate report, and
  retained no raw payloads.
- Provenance/isolation: the v2 config hash, report/manifest hashes, and
  localhost GPU-0 attestation agreed; the manifest records 128 train essays,
  zero validation source rows, zero validation requests, and no selection
  artifact.
- Passed gates: split isolation, sample size, zero transport/schema failures,
  pointwise repeat stability (1.0), pairwise repeat stability (1.0), raw
  pairwise abstention (0.0), and identity neutrality (0 non-neutral choices).
- Failed preregistered gates (thresholds unchanged): factorial consistency
  0.615885 < 0.90; label imbalance 0.362454 > 0.05; first-position imbalance
  0.195167 > 0.05; two-lane consensus 0.0 < 0.60; invalid-control abstention
  0.0 < 0.99.

**Decision:** fail. The v2 artifact is preserved; no high-confidence
selection artifact, SFT, DPO, or GRPO is authorized.

## Versioned judge remediation and durable next stage

The aggregate failure mode is calibration/presentation bias, not transport or
schema failure: deterministic repeats succeeded while invalid controls,
factorial invariance, and cross-lane consensus failed. The targeted v3
amendment retains every v2 gate threshold, preserves train-only isolation, and
changes only the ambiguous score-conditioning and invalid-evidence
instructions. It is capped at 32 train essays, uses the benchmark-validated
four-server-slot/four-client-request configuration, and remains GPU-0-only.

Before scheduling, the v3 GGUF byte/hash, llama.cpp revision/tag,
split-scoped train artifact, config isolation, shell syntax, Python compile,
and watchdog dry-run gates passed. The watchdog attempted
`qwen36-judge-v3-pilot-20260720-001` once, but the runner refused before
creating a run directory because physical GPU 0 was not idle. No v3 request,
source row, response, selection artifact, or training stage started. This is
a GPU-ownership/resource conflict; no retry is authorized until that ownership
or a new approved GPU-0 window is confirmed.

## Score-head status and next stage

Completed scalar-head artifacts are present for content, organization, and
expression and share Qwen3-Embedding-8B revision, prepared-manifest hash,
seed, LoRA architecture, and length. Their completion-time selection metrics
are not external metrics and were not used for promotion.

No valid score-head external aggregate existed at terminal inspection. The
queued evaluator was invalid for this task because it targeted `selection_dev`
and referenced a nonexistent content completion lineage. It was therefore not
run. A versioned frozen-validation-only evaluator/config now binds the three
actual completion artifacts, checks each model checksum, computes per-axis
RMSE/MAE/Spearman plus the external arithmetic average, and persists only an
aggregate output. The frozen-validation aggregate then completed for 400 rows:
content RMSE/MAE/Spearman 2.322682/2.229000/null; organization
2.446777/2.286875/null; expression 2.761934/2.676875/null; external average
2.484930/2.397583/null. It is materially worse than the compatible baseline
and its rank correlations are undefined, so it is an invalid candidate
architecture. This negative result is preserved and nothing is promoted.

The next required score stage is a versioned bounded replication/ablation,
then the same frozen-validation aggregate-only check. It is not launched:
replacing or sharing the active watchdog's GPU-local backfill would conflict
with its current ownership. GPUs 4--7 remain untouched.

Compatible frozen-validation baseline for comparison: Qwen3 final content RMSE/MAE/Spearman
0.659371/0.537459/0.457325; organization 0.876352/0.695508/0.445892;
expression 0.554379/0.430474/0.548509; average
0.603776/0.482735/0.519270.

## Evidence and remaining work

- Aggregate recomputation:
  `outputs/aggregate-reports/qwen36-judge-v2-pilot-20260720-001.recomputed.json`
  (ignored; aggregate-only).
- V3 config/runner: `configs/qwen36_gguf_judge.v3.pilot.json`,
  `scripts/run_qwen36_judge_v3_checked.sh`, and
  `scripts/run_qwen36_judge_v3_pilot.sh`.
- Frozen score evaluator:
  `scripts/evaluate_pre_sft_score_ensemble_validation.py` and the versioned
  watchdog config under the ignored reservation runtime.
- Aggregate evaluation is complete. Further judge-v3 and score remediation is
  blocked only by current GPU ownership; no SFT/DPO/GRPO stage is authorized.
