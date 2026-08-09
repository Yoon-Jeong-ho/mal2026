# Human-audited decoder score-prompt validation — 2026-08-09-001

- **Status:** completed; negative calibration result, not promoted.
- **Run ID:** `decoder-human-audited-prompt-validation-v1-20260809-001`.
- **Question:** does a zero-shot scoring prompt informed by reliable train-split
  human reasons improve Qwen3.5-9B decoder scoring on the fixed 400-row
  validation split relative to the exact `evaluation.txt` prompt and the prior
  public score-band prompt?
- **Privacy:** individual writings, identifiers, human reasons, audit notes,
  model requests, responses, and predictions remain under ignored restricted
  or output roots. This record contains aggregate evidence only.

## Authorized task card

- **Stage and deliverable:** derive a new rule-only prompt, run a deterministic
  vLLM validation ablation, preserve restricted predictions, and report
  aggregate metrics and negative results.
- **Canonical inputs:** `eval/train.jsonl` only for binding the human-feedback
  derivation rows; all 400 rows of `eval/validation.jsonl` for evaluation;
  `evaluation.txt`; the privacy-reduced Cloudflare human response export; and
  its AI-assisted reason audit.
- **Human-feedback filter:** retain a direct-score reason only when its response
  row is from `train`, the human integer score is within ±1 of that row's
  canonical half-up target band, and the secondary reason audit is `match` or
  `partial`. The expected selection is 18 axis reasons. Three otherwise
  eligible validation-axis reasons are excluded before rule derivation.
- **Prompt arms:** exact `evaluation.txt` (`official_p0`), the prior public
  axis-band prompt (`public_band_p1`), and the new train-human-audited rule-only
  prompt (`human_audit_p2`). No individual human example or reason is inserted
  into a model request.
- **Model and decoding:** `Qwen/Qwen3.5-9B` revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, deterministic temperature 0,
  structured three-axis score-plus-rationale JSON, 512 initial output tokens,
  and a 2,048-token integration retry only for exact length truncations.
- **Primary comparison:** P2 minus P0 macro raw RMSE and macro raw Spearman.
  P2 minus P1 is secondary. A paired 10,000-replicate essay bootstrap is fixed
  for raw-RMSE deltas; score histograms and exact band recall are diagnostics.
- **Scientific boundary:** validation has already been observed by earlier
  experiments. This run is locked/descriptive and cannot provide an unbiased
  model-selection estimate or justify automatic promotion of P2.
- **Resource scope:** **GPU 6 only**, tensor parallel size 1. The current user
  explicitly requested on 2026-08-09: “vllm을통해 6번서버에서 빠르게 진행”.
  This is recorded as authorization to expand this named task card to GPU 6;
  no GPU 4, 5, or 7 process may be changed or displaced.
- **Environment:** existing `.venv-standard`; no package installation or new
  environment.
- **Test ladder:** focused synthetic unit tests and canonical prepare preflight,
  then a three-request real smoke on GPU 6, then the fixed 1,200-request full
  validation run without routine phase confirmation.
- **Output paths:** restricted derivation manifest and predictions under
  `data/processed/restricted/decoder_human_audited_prompt_validation_v1/`;
  aggregates under `outputs/analysis/`; append-only runtime ledger under
  `outputs/decoder-human-audited-prompt-validation-v1/`.
- **Escalation:** stop for a GPU 6 conflict, non-length parse failure, changed
  input checksum, unapproved data transfer or external cost, destructive
  overwrite, or any proposed change to the fixed prompt/data/model/decoding or
  comparison protocol.

## Reproducibility bindings

- `evaluation.txt` SHA-256:
  `1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145`.
- public band prompt SHA-256:
  `1c126b26b4ff8b99a6dd4d14235a9e8eaa07b7e9f3d8f28040e2fdccd35a4e4e`.
- human-audited prompt SHA-256:
  `690a7c58c60240c9da49904bda895f868a7d070c3f6276aaf198ab23a88df250`.
- human response export SHA-256:
  `471a13750fcb71bb6b8683b7839803fb8b4198532cb4abe7c66c4886c8cc48a8`.
- reason-audit SHA-256:
  `cf985f375ffbfa3e4431a74fdf70268ecc64dde1b94e6aad26e3259c45e2daa0`.
- canonical train/validation SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737` /
  `0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4`.
- launch Git SHA before experiment-specific changes:
  `243f18742b46adb551a0b4ffe31130eaf6d8e46d` with unrelated pre-existing
  worktree changes preserved.

Fixed commands:

```text
PYTHONPATH=src .venv-standard/bin/python scripts/run_decoder_human_audited_prompt_validation.py --config configs/decoder_human_audited_prompt_validation.v1.json --stage prepare
PATH="$PWD/.venv-standard/bin:$PATH" CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src .venv-standard/bin/python scripts/run_decoder_human_audited_prompt_validation.py --config configs/decoder_human_audited_prompt_validation.v1.json --stage run
```

## Preflight and execution evidence

- Focused synthetic unit tests: 3/3 passed.
- Canonical prepare gate selected 18 train-only axis reasons: content 3,
  organization 6, expression 9; audit labels were 8 `match` and 10 `partial`;
  anonymized reviewer contributions were 17 and 1. Three otherwise eligible
  validation-axis reasons were recorded and excluded before prompt derivation.
- GPU 6 immediately before launch: 0 MiB used, 0% utilization, no listed
  compute process. No other GPU was used or altered.
- Real smoke: one request per arm, 3/3 parsed successfully.
- Full run: 1,200 requests (400 validation rows × 3 prompts). Fourteen initial
  responses ended at the fixed 512-token ceiling; all were preserved and all
  14 resolved under the preapproved same-prompt 2,048-token length-only retry.
  Final parse coverage was 1,200/1,200.
- Runtime: existing `.venv-standard`, vLLM 0.25.1, one H100 on physical GPU 6,
  TP=1, CUDA graphs, prefix caching, 12,288-token context, up to 64 sequences,
  and 32,768 batched tokens. GPU 6 returned to 0 MiB and 0% utilization after
  clean shutdown.

## Validation results

`raw RMSE` compares integer decoder scores with the canonical continuous
labels. `int RMSE` compares against half-up integer labels. Lower RMSE and
higher Spearman are better.

| Prompt | Raw RMSE | Raw Spearman | Int RMSE | Score 3 | Score 4 | Triplet acc. |
|---|---:|---:|---:|---:|---:|---:|
| P0 exact `evaluation.txt` | 0.9481 | 0.4198 | **1.0043** | 38.33% | 37.83% | 5.75% |
| P1 public band rules | **0.9119** | 0.3979 | 1.0155 | 66.67% | 10.92% | **6.00%** |
| P2 train-human-audited rules | 0.9885 | **0.4366** | 1.0843 | 49.50% | 17.25% | 4.00% |

P2 minus P0 macro raw-RMSE was `+0.0404` (worse), with paired 10,000-essay
bootstrap 95% interval `[-0.0031, +0.0837]`. P2 minus P1 was `+0.0765`
(worse), interval `[+0.0435, +0.1095]`. P2 improved rank correlation by
`+0.0168` over P0 and `+0.0387` over P1, but this did not compensate for its
calibration error.

### Axis diagnosis

| Axis | P0 raw RMSE | P1 raw RMSE | P2 raw RMSE | P2−P0 |
|---|---:|---:|---:|---:|
| Content | 0.8479 | **0.7542** | 0.7949 | -0.0530 |
| Organization | 0.9762 | 0.9456 | **0.9147** | -0.0615 |
| Expression | **1.0202** | 1.0360 | 1.2558 | +0.2356 |

The P2 failure is concentrated in expression. Relative to P0, P2 moved the
mean score down by 0.420 content points, 0.628 organization points, and 0.365
expression points. The expression prediction mean fell from 3.005 to 2.640;
199/400 expression predictions became score 2, while P2 produced no expression
score 5. This direction is consistent with the derivation sample's imbalance:
9/18 accepted reasons concern expression, nearly all accepted reasons come from
one reviewer, and the small human study intentionally over-represents low
bands.

P2 improved pooled exact score-2 recall from 26.72% (P0) to 50.86%, but score-4
recall fell from 43.51% to 21.56% and score-5 recall fell from 19.05% to 0%.
Score-1 recall remained 0% in all three prompts. P1 and P2 therefore exhibit
different versions of the same prompt-prior/tail-calibration problem.

## Decision and limitations

1. Do **not** replace `evaluation.txt` or the prior P1 prompt with P2.
2. Preserve P2 as a negative result: the human-derived boundary rules improve
   rank signal and content/organization RMSE but overcorrect expression and
   erase the score-5 tail.
3. The filter itself is not sufficient evidence of representative expertise.
   It retained only 18 axis reasons, 17 from one reviewer, and was intentionally
   concentrated on low-band examples. “Within ±1 of the canonical band” plus a
   non-mismatching reason audit does not remove this sampling prior.
4. A future prompt should not be retuned on these validation results. If the
   idea is revisited, build balanced train-only or out-of-fold human evidence
   across axes, reviewers, and bands, then reserve an untouched confirmation
   set. This run's validation is descriptive and already observed.

Aggregate evidence:

- protocol SHA-256:
  `54c6c4d749c9df7b13e114932993c01de307078e2fa2aa9f573faf29b69d3d5a`;
- smoke SHA-256:
  `7c4d9e29fa443684f3d4b5e00f4e4d4b8578197680c9493198496216cda7ad4c`;
- aggregate SHA-256:
  `e18592c80d54de08b4b6e5822064f0d519124cc810d8879308ffa3b38e2fb261`;
- restricted final prediction SHA-256:
  `9288269727a86e7a8271fda4027ea656971ab00986acb2f5e55a4e88654107ec`;
- append-only ledger:
  `outputs/decoder-human-audited-prompt-validation-v1/decoder-human-audited-prompt-validation-v1-20260809-001/ledger.jsonl`.
