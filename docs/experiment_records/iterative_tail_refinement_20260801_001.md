# Iterative tail refinement, 20-round train-OOF study

Run ID: `iterative-tail-refinement-v1-20260801-001`

Status: completed, no candidate promoted; exact R0 OOF baseline retained

Date: 2026-08-01 (Asia/Seoul)

## Decision

The fixed 20-round train-only program completed. No round passed every
predeclared promotion gate, so the final selection is the unchanged exact R0
five-fold OOF baseline. This study does **not** establish a validation,
generalization, deployment, or leaderboard improvement.

The best exploratory challenger was round 17, a score-blind evidence-hash
ridge projection. Its macro continuous RMSE was `0.564401`, compared with
`0.568780` for R0 (`+0.004379` improvement). It was not promoted because the
required improvement was `0.005`, equal-group RMSE slightly worsened, and the
low-tail `{1,2}` RMSE worsened materially. The existing baseline remains the
only protocol-valid selection.

## Immutable inputs and isolation

- Git SHA at launch: `a3763c1fc918c80c80663f20f2abc1620bd44f73`
- Config SHA-256: `25508fc5fb510251cf94dc1b6a7cd798b3ff102efceaef4cbc0ac5a47943c7a5`
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Exact R0 OOF predictions SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- Frozen Qwen3-Embedding-8B rows SHA-256:
  `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16`
- Fold-assignment fingerprint:
  `8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db`
- Embedding model: `Qwen/Qwen3-Embedding-8B`, immutable revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
- Score-blind rationale train SHA-256 values:
  `1a10524f...10223`, `f5cce405...e3c`, `d4a2be9a...8e55`, and
  `45dc9bfd...f347`. Their manifests state that no human/reference score was
  read or prompted.
- Records: 2,000 train rows, five fixed OOF folds of 400.
- Targets: only `content`, `organization`, and `expression`; no `average`
  target was loaded or predicted.
- Validation: not loaded and not used for fitting, round selection, or
  calibration.
- External API: not used. Existing local score-blind artifacts were enough.

All row-level predictions, feature caches, task queues, and logs remain under
ignored `data/processed/restricted/` or `outputs/` paths. This tracked record
contains aggregate values only.

## Execution

Environment: existing `.venv-standard`; Python 3.12.3; PyTorch
`2.11.0+cu130`; CUDA 13.0; Linux 6.8.0. Hardware was four NVIDIA H100 80GB
GPUs, restricted to GPU 0--3. Solar was stopped under the user's explicit
resource-transition authorization before encoder work.

Commands:

```bash
PYTHONPATH=src .venv-standard/bin/python scripts/run_iterative_tail_refinement.py --prepare
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  .venv-standard/bin/python scripts/run_iterative_tail_refinement.py --smoke --device cuda:0
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
  .venv-standard/bin/python scripts/run_iterative_tail_refinement.py --full
```

The full stage ran in tmux session
`mal2026-iterative-tail-20260801-001`. GPU0 passed the neural, ridge,
threshold, prompt-group, and evidence smoke checks before the full stage.
The four full workers completed 135 fold/variant jobs for rounds 2--16, then
10 jobs for rounds 17--18. Rounds 19--20 were aggregate ensemble/calibration
stages. The ledger records all four GPU IDs and the two worker-stage
completions.

## Aggregate results

Macro RMSE is the arithmetic mean of the three continuous axis RMSE values.
Each candidate is a five-fold OOF prediction on the fixed train population.

| Round | Method / selected subvariant | Macro RMSE |
| ---: | --- | ---: |
| 1 | exact R0 OOF baseline | **0.568780** |
| 2 | ridge residual, alpha 10 | 0.569296 |
| 3 | MLP residual, hidden 256 | 0.615356 |
| 4 | CORAL proxy | 0.658773 |
| 5 | Huber + ordinal | 0.605931 |
| 6 | uncertainty gate, 25% coverage | 0.570251 |
| 7 | low-tail weighting, 1.5 | 0.614983 |
| 8 | high-tail weighting, 1.5 | 0.606532 |
| 9 | equal-band weighted replay proxy | 0.650424 |
| 10 | prompt-equal grouped loss | 0.605996 |
| 11 | 3-vs-4 auxiliary head | 0.615023 |
| 12 | four monotonic thresholds | 0.569935 |
| 13 | adjacent contrastive proxy | 0.599848 |
| 14 | score-blind content evidence | 0.607412 |
| 15 | score-blind organization/expression evidence | 0.610503 |
| 16 | four-agent consensus/disagreement | 0.623540 |
| 17 | evidence-hash ridge projection | **0.564401** |
| 18 | text/structured concatenation proxy | 0.649346 |
| 19 | eligible-member ensemble fallback (R1 only) | 0.568780 |
| 20 | five-fold bounded affine calibration of R1 | 0.569874 |

### Exploratory round 17 versus baseline

| Metric | R1 baseline | R17 | Improvement direction |
| --- | ---: | ---: | ---: |
| Macro RMSE | 0.568780 | 0.564401 | +0.004379 |
| Macro Spearman | 0.600288 | 0.608892 | +0.008603 |
| Equal-group RMSE | 0.691549 | 0.691684 | -0.000135 |
| Low-tail `{1,2}` RMSE | 0.923335 | 0.984277 | -0.060942 |
| Score-5 RMSE | 0.884190 | 0.836521 | +0.047669 |
| True-gold 3/4 balanced accuracy | 0.643313 | 0.660647 | +0.017334 |
| 3→4 error rate | 0.305836 | 0.320312 | -0.014476 |
| 4→3 error rate | 0.356965 | 0.331492 | +0.025472 |
| Score-1 descriptive RMSE | 1.164567 | 1.312709 | -0.148142 |

Round 17 reduced all three axis RMSE values: content by `0.003473`,
organization by `0.007516`, and expression by `0.002148`. Its low-tail RMSE
worsened on all three axes. Its score-5 continuous RMSE improved, but
score-5 integer recall remained zero on all three axes, so this is a modest
continuous shift rather than categorical score-5 recovery.

A post-selection, descriptive 10,000-resample paired row bootstrap gave a
macro-RMSE improvement interval of `[0.001271, 0.007469]`. This interval is
not selection-adjusted: round 17 was inspected after 20 methods were compared
on the same OOF population. It is therefore not confirmatory or external
generalization evidence.

## Promotion and final gate

Each incumbent replacement required all of the following: macro RMSE
improvement at least `0.005`; equal-group RMSE improvement at least `0.010`;
both low-tail and score-5 RMSE improvement; 3/4 balanced-accuracy improvement
at least `0.01`; no axis RMSE more than `0.01` worse; and macro Spearman fall
no larger than `0.005`. Score 1 was descriptive only because the axis counts
were 11, 26, and 4.

No round passed the complete AND gate. Round 17 passed score-5, 3/4,
no-axis-regression, and Spearman checks, but failed macro RMSE, equal-group,
and low-tail checks. Round 19 consequently had no promoted challenger to
ensemble and evaluated the baseline with weight 1. Round 20 worsened that
baseline. The final superiority requirement of at least `0.01` macro RMSE
gain and a candidate-minus-baseline bootstrap upper bound below zero failed.
The final baseline-to-itself bootstrap interval is the tautological zero
interval and must not be interpreted as evidence of equivalence.

Because no candidate was promotion-eligible, the fail-closed baseline fallback
was applied. There is no positive final candidate for an independent outer
refit. The task card's stronger nested-cross-fitting aspiration was therefore
not demonstrated; same-OOF variant/method selection still creates exploratory
selection optimism, although the no-promotion conclusion is conservative.

## Multi-agent aggregate review

Three independent subagents reviewed aggregate metrics, modeling behavior,
and protocol compliance. Their consensus was:

1. retain R1; label R17 only as an exploratory challenger;
2. interpret R17 as regularized evidence projection/denoising, not as proof of
   a temperature/KL distillation method;
3. do not trade the consistent low-tail loss for the smaller overall RMSE;
4. do not claim validation or deployment improvement from this train-only OOF
   search.

The modeling review attributed the neural losses chiefly to an untrained
fixed 50:50 regression/ordinal inference blend, strong ordinal central
regression, and a short full-batch optimization schedule. The evidence ridge
appears to preserve repeatable high/central-band corrections while shrinking
much of the weaker neural teacher, but it also inherits the teacher's upward
low-tail bias.

## Deviations and negative evidence

These results evaluate the implemented proxies, not idealized names:

1. Round 9 used deterministic inverse-frequency loss weighting rather than a
   literal replay sampler.
2. Round 13 used adjacent-label supervised contrastive positives rather than
   an explicit margin implementation.
3. Round 16 consumed four pre-existing score-blind sources although the round
   method description names the v2 artifact; all four input hashes are bound
   globally in the config, but round aggregates do not repeat consumed SHAs.
4. Round 17 did not use temperature/KL logits. It fit ridge residuals to the
   round-16 OOF continuous pseudo-target, so the accurate name is regularized
   evidence projection.
5. Round 18 used plain feature concatenation followed by the joint model, not
   a learned cross-view gate.
6. Round 3 interpreted `[256,128]` as two hidden-width candidates rather than
   a two-layer architecture.
7. Round 20 used positive-slope bounded affine calibration; it is monotonic
   and bounded, but narrower than a general monotonic calibrator.
8. Variant selection, method comparison, and the descriptive bootstrap use the
   same OOF population. No selection-adjusted nested result is available.

Two integration failures were preserved in the append-only ledger: an
over-strict `1e-12` float32 baseline identity tolerance and a bootstrap local
variable typo. Both repairs were mechanical and changed no model, data,
metric, or scientific threshold.

## Verification and artifacts

Focused test command:

```bash
PYTHONPATH=src .venv-standard/bin/python -m unittest -v \
  tests.test_iterative_tail_metrics \
  tests.test_iterative_tail_models \
  tests.test_iterative_tail_protocol \
  tests.test_iterative_tail_runner
```

Result: 16 tests passed. `py_compile` and `git diff --check` also passed.

Ignored runtime evidence:

- Aggregate/runtime root:
  `outputs/iterative-tail-refinement-v1/iterative-tail-refinement-v1-20260801-001`
- Restricted row-level root:
  `data/processed/restricted/iterative_tail_refinement_v1/iterative-tail-refinement-v1-20260801-001`
- Promotion summary SHA-256:
  `d3e0e2f7871518bf9123e554ad19afc764a5e257a7a9a087a9cdd1e466e3d0f7`
- Completion marker SHA-256:
  `432c24066c4fef3c3b6c09c638104378d85bd2717dc44fbeb5c3c28d4b2d7262`
- Final retained restricted prediction SHA-256:
  `aaf17f93fdad2d93d137d1f3afe0feac2f8e72db8957168febf3e992c4e708c6`
- Round-17 descriptive bootstrap SHA-256:
  `f51906d87b23b37b0f63ad83582f3bb8bbb0ee5e796acbfe691441113908ba4e`

The negative outcome is retained: this experiment does not replace the
current scoring model.
