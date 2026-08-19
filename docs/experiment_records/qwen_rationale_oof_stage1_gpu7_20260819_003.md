# Qwen rationale-aware Stage1 OOF on GPU7 — 2026-08-19-003

- **Status:** stopped by user after the raw-input arm completed exact five-fold
  OOF. The interrupted rationale fold has no admitted result, and the dropout
  arm did not start.
- **Question:** determine whether score-blind rationales or deterministic
  rationale dropout add score-prediction signal beyond `prompt + essay` for a
  Qwen3-Embedding-8B score encoder.
- **Run ID:** `qwen-rationale-oof-stage1-gpu7-v1-20260819-003`.
- **Launch Git SHA:** `98672f9677ff7d8516e05c7302e573d01bccc615`; the
  experiment was launched from a dirty worktree and every effective input was
  checksum-bound by the durable runner.
- **Authorization and GPU scope:** the user explicitly authorized GPU7 for this
  rerun. Only the repository-owned process titled
  `mal2026:qwen-oof:<phase>:<arm>:<fold>:gpu7` was stopped. A later unrelated
  GPU7 process was neither terminated nor modified.
- **Environment:** existing `.venv-standard`; no package installation or
  environment creation.

## Immutable inputs

- Config: `configs/qwen_rationale_oof_stage1_gpu7.v3.json`, SHA-256
  `061e566aec37ea4e0ac8e2724ce37266ee86f2dbc79bd4eae24c3d5bbb4f8579`.
- Train: `eval/train.jsonl`, SHA-256
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Fold map: SHA-256
  `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16`.
- Rationale handoff: SHA-256
  `5c33995cee94d7b5d79d7e10db6daa4c45a63a1e65d0e3f46eb36ac5f7cd70ad`.
- Score-input prompt: `rationale_to_score.txt`, SHA-256
  `0bec7d13a109522413226a3fc155bc0b3ec5f7838b3de61008f2cb9eacf6e9f2`.
- Model: `Qwen/Qwen3-Embedding-8B`, revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
- Initialization: AI-Hub full backbone and matched bounded score head, then a
  fresh LoRA adapter.
- Seed: `2026081201`; five folds; seven epochs; learning rate `1e-4`; LoRA
  rank/alpha `16/32`; maximum length `2560`; effective batch size `32`.
- `average` was never a target. Validation content was not parsed or used for
  selection.

## Execution and recovery

The durable runner first passed six CPU contract tests, dependency preflight,
the GPU7 ownership gate, and a two-step GPU7 smoke. The smoke peak allocation
was 25,773.95 MiB.

Two earlier lineages produced no scientific result:

1. `-001` stopped at preflight because the frozen config referenced an older
   score-input prompt hash.
2. `-002` stopped before its first optimizer step because the prompt-routing
   manifest still referenced older tracked prompt hashes.

The `-003` repair bound the routing manifest to the current tracked prompts and
added a rendering regression test. The raw arm then completed all five folds.
The user requested GPU release while `rationale_mse` fold 0 was at optimizer
step 79/350; that incomplete state was excluded. No rationale or dropout metric
was inferred from partial training.

Canonical restart command, after an authorized GPU7 idle check:

```bash
PYTHONPATH=src .venv-standard/bin/python \
  scripts/run_qwen_rationale_oof_matrix.py \
  --config configs/qwen_rationale_oof_stage1_gpu7.v3.json \
  --phase stage1
```

The scheduler skips only fold directories containing a checksum-bound complete
`result.json`. In the original filesystem it therefore preserves the five raw
folds and restarts the interrupted rationale fold from the beginning.

## Completed raw-input result

All values below cover 2,000 unique train essays with exactly one held-out
prediction per essay.

| Metric | New raw arm | Frozen R0 exact OOF | Delta |
| --- | ---: | ---: | ---: |
| Continuous macro RMSE vs raw decimal gold | 0.592219 | **0.568780** | +0.023439 |
| Half-up integer macro RMSE vs raw decimal gold | 0.651869 | **0.635718** | +0.016151 |
| Half-up integer macro Spearman | 0.504953 | **0.513527** | -0.008574 |

Axis-level integer RMSE was `0.600704` for content, `0.753056` for
organization, and `0.601846` for expression. The predicted integer
distributions were:

- content: `{1: 0, 2: 93, 3: 1193, 4: 714, 5: 0}`;
- organization: `{1: 0, 2: 162, 3: 908, 4: 930, 5: 0}`;
- expression: `{1: 0, 2: 63, 3: 657, 4: 1280, 5: 0}`.

Recall was zero for both rounded-gold scores 1 and 5 on every axis. Score-2
recall was `0.1931`, `0.2540`, and `0.2727` for content, organization, and
expression. Thus the raw arm did not improve R0 and retained severe 3/4 central
compression, with organization remaining the weakest axis.

## Decision and artifacts

Do not promote the new raw arm. The rationale-input hypothesis remains unknown,
not negative: its exact OOF arm did not finish. If resumed, use a predeclared
paired futility gate on rationale folds 0 and 1 before spending the remaining
five-fold/dropout budget. Keep the existing R0 score path unless a candidate
beats its train-only exact OOF metrics.

Aggregate-only stop report:

```text
outputs/qwen-rationale-oof-stage1-gpu7-v1/
  qwen-rationale-oof-stage1-gpu7-v1-20260819-003/
  stage1/stopped_by_user_report.json
sha256:c63f73442a82bd0998865a77f08c8ce9c5a92cdf1c4fa8ee7f575efae79dd988
```

Row predictions, trainable states, rationales, labels, identifiers, telemetry,
and runner logs remain only in ignored restricted/output locations.
