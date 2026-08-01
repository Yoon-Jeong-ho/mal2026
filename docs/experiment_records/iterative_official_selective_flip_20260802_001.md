# Official Terra selective boundary flips (V9)

Run ID: `iterative-official-selective-flip-v9-20260802-001`

Status: completed; strict final gate failed, exact R0 retained

Date: 2026-08-02 (Asia/Seoul)

## Decision

V9 tested a materially different use of the frozen V7 official Terra score
features after V8's smooth boundary nudges failed. It made a discontinuous
3/4 flip only when a fresh classifier disagreed confidently with the fresh
alpha-10/cap-0.5 residual prediction and that prediction was within a fixed
window around 3.5. The protocol was committed before the smoke or full
train-OOF run, and no V9 full-OOF prestudy was performed.

The nested development result improved exact R0 macro continuous RMSE from
`0.568780` to `0.563067`. The 10,000-resample paired candidate-minus-R0
interval was `[-0.009640, -0.001937]`, and every axis plus both tails improved.
However, macro gain was only `0.005714` rather than the required `0.010`, and
true-gold 3/4 balanced-accuracy gain was `0.004620` rather than `0.010`.
The final gate therefore failed and the protocol-valid retained prediction is
the unchanged exact R0 OOF baseline.

This is same-train nested development evidence only. It is not independent
validation, hidden-test, leaderboard, generalization, or deployment evidence.

## Frozen protocol and inputs

- Preregistration commit: `79793d7`.
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`.
- Fixed fold fingerprint:
  `8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db`.
- Official Terra manifest SHA-256:
  `960ae42ac19e79bd8cff747844ee58a5c724abe30d46d6f3865cb92e22b9de53`.
- Official Terra rows SHA-256:
  `a1791c418c79c0b76399ddb993e862f34209c2da95b0c13f7cda87f403a24e4c`.
- V8 aggregate SHA-256:
  `97dc01637c16a3448eb6971b03d266b38f7c1aee339add7b474db65866ecae4e`.
- Records: 2,000 train rows, five original folds of 400.
- Official inputs: three score-blind GPT-5.6-Terra participant outputs per
  essay. Their manifest states that no human/reference score was read or
  prompted.
- V9 loaded neither validation nor the `average` target and made zero new API
  calls. Rationale text was not used by V9.

The exact three candidates were:

1. adjacent 3-vs-4, confidence `0.60`, boundary window `0.20`;
2. adjacent 3-vs-4, confidence `0.65`, boundary window `0.15`;
3. average of adjacent and cumulative-`>=4` probabilities, confidence `0.60`,
   boundary window `0.20`.

Each selected cell was moved only to `3.499` or `3.501`. Every residual ridge
system and every logistic head started fresh for every candidate and fold;
no checkpoint or learned state was reused.

## Nested execution

For each outer fold `O`, every other fold served once as inner validation `D`
and the remaining three folds formed `S`. All three candidates completed all
four inner predictions before the unchanged seven-gate selector ran. The
selected specification was then freshly refit on all four outer-train folds
and predicted `O` once. No outer-fold gold was used before selection freeze
or prediction.

GPU0 passed the required real 96-train/32-predict smoke. Outer folds 0--3
then ran concurrently on physical GPUs 0--3; outer fold 4 followed on GPU0.
Observed snapshots during concurrent work showed all four GPUs active at
roughly 28--36% utilization and about 913 MiB each. These jobs are tiny
39-feature float64 ridge/LBFGS problems, so low memory occupancy is expected;
allocating artificial batches would not improve the computation.

## Inner selections

| Outer fold | Selected | Macro gain | Equal gain | Low gain | High gain | 3/4 BA gain | Outcome |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | dual flip c0.60/w0.20 | 0.012594 | 0.018116 | 0.020281 | 0.035866 | 0.015337 | eligible |
| 1 | adjacent flip c0.65/w0.15 | 0.013387 | 0.021454 | 0.024981 | 0.053187 | 0.018351 | eligible |
| 2 | R0 fallback | 0.010314 | 0.014078 | 0.015614 | 0.030153 | 0.007102 | failed only 3/4 BA |
| 3 | R0 fallback | 0.011351 | 0.019884 | 0.054153 | 0.024466 | 0.001637 | failed only 3/4 BA |
| 4 | dual flip c0.60/w0.20 | 0.016416 | 0.025384 | 0.062672 | 0.027515 | 0.010072 | eligible |

Folds 2 and 3 again failed only the frozen 3/4 balanced-accuracy gate. The
selector did not lower the threshold after seeing those outcomes. V9 did
improve over V8's nested development RMSE (`0.565515`) and BA gain (`0.003018`),
but did not solve the cross-fold 3/4 instability.

## Final nested metrics

| Metric | Exact R0 | V9 nested selected | Improvement |
| --- | ---: | ---: | ---: |
| Macro continuous RMSE | 0.568780 | 0.563067 | +0.005714 |
| Equal-band RMSE | 0.691549 | 0.682969 | +0.008580 |
| Low-tail `{1,2}` RMSE | 0.923335 | 0.907099 | +0.016236 |
| Score-5 RMSE | 0.884190 | 0.868330 | +0.015860 |
| True-gold 3/4 balanced accuracy | 0.643313 | 0.647933 | +0.004620 |
| Macro Spearman | 0.600288 | 0.605904 | +0.005615 |

Axis RMSE gains were content `0.002553`, organization `0.006546`, and
expression `0.008041`; no axis worsened. Score-1 remained descriptive only,
as required by the original plan. The final gate passed the paired-bootstrap,
both-tail, no-axis-regression, and Spearman checks, but failed the macro-gain
and 3/4-BA-gain checks.

## Verification and artifacts

Fourteen focused V9 tests passed before launch; Python compilation, config
parsing, shell syntax, launcher progress parsing, and scoped
`git diff --check` also passed.

- Task card SHA-256:
  `6fd1568c995bbd1b141915e20443bb4bc541991a9e4248208ddee9fa4f5c2f10`.
- Smoke SHA-256:
  `0cacbd0da4f2fa9af14fb0c04c8ccc1a2b089b822309c16eebc1babd0f65b1f4`.
- Aggregate SHA-256:
  `17b71c86c2d638cb8c531130844c56d6243932fb24cd5016f4225bcfad8d4fef`.
- Completion SHA-256:
  `5fc657d4a37d5c7e9b0d9e6570e0706f6d9f228497808105272843abd99acdf9`.
- Retained restricted prediction SHA-256:
  `36059045be7b2824e8e2f775495e4dda84a817dcdcc65eb19ac7d7696956bcc9`.

Public aggregate artifacts are under
`outputs/iterative-official-selective-flip-v9/`; row predictions remain only
under ignored `data/processed/restricted/iterative_official_selective_flip_v9/`.
