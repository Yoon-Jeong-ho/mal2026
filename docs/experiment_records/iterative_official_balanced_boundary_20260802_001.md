# Class-balanced Terra + Luna 3/4 boundary head (V11)

Run ID: `iterative-official-balanced-boundary-v11-20260802-001`

Status: completed; strict final gate failed, exact R0 retained

Date: 2026-08-02 (Asia/Seoul)

## Decision

V11 tested the preapproved dedicated 3-vs-4 auxiliary-head round after V10.
It retained the frozen score-blind Terra+Luna 96-feature matrix and residual
ridge, but trained every adjacent 3/4 classifier with equal total weight for
gold 3 and gold 4. The three specifications and all thresholds were committed
before the run. No V11 OOF prestudy, validation access, API call, checkpoint
continuation, or `average` target was used.

The nested development prediction improved exact R0 macro continuous RMSE
from `0.568780` to `0.563918`. The 10,000-resample paired interval excluded
zero and both tails, all axes, equal-band RMSE, 3/4 balanced accuracy, and
Spearman moved favorably. The final macro gain (`0.004862`) and 3/4 balanced-
accuracy gain (`0.002545`) were nevertheless below the required `0.010`.
Exact R0 remains the protocol-valid model.

V11 is same-train nested development evidence, not independent validation or
a hidden-test, leaderboard, generalization, or deployment result.

## Frozen model and protocol

- Preregistration and sealed-runner commit: `7810dfe`.
- Input feature dimensions: 96 (39 Terra, 39 Luna, 18 cross-source).
- Feature-matrix SHA-256:
  `eae934ac8f79e38da9a57556368e1e5f90b7785b396494c0a583f9487b32e708`.
- Common residual: fresh alpha-10, cap-0.5 ridge for every candidate/fold.
- Boundary objective: gold 3 vs gold 4 only, equal total class weight per
  axis, fresh zero initialization, LBFGS, and hard near-boundary flips only.
- Fixed candidates: L2 `0.01`/confidence `0.50`/window `0.25`; L2 `0.10`/
  confidence `0.50`/window `0.25`; L2 `0.01`/confidence `0.55`/window `0.20`.
- Selection: unchanged original seven-gate AND rule inside sealed 5-outer by
  4-inner cross-fitting; all candidates completed before selection.
- Outer gold remained locked until selected-spec freeze, fresh refit, and one
  prediction of that outer fold.

## Inner selections

The narrow-window candidate (L2 `0.01`, confidence `0.55`, window `0.20`)
was selected in outer populations 0 and 1. Outer populations 2--4 fell back
to exact R0 because every candidate missed only the 3/4 balanced-accuracy
gate. For the narrow-window candidate, inner macro/BA gains by outer were:

| Outer | Macro RMSE gain | Equal gain | Low gain | High gain | 3/4 BA gain | Outcome |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.010231 | 0.017797 | 0.018884 | 0.043086 | 0.013959 | eligible |
| 1 | 0.011340 | 0.021449 | 0.018458 | 0.064879 | 0.019286 | eligible |
| 2 | 0.007788 | 0.013216 | 0.006411 | 0.040559 | 0.002515 | BA fail |
| 3 | 0.009154 | 0.020285 | 0.052109 | 0.033807 | 0.003222 | BA fail |
| 4 | 0.012484 | 0.024275 | 0.057315 | 0.037192 | 0.006000 | BA fail |

Compared with V10's plain dual-source ridge, class balancing reduced rather
than rescued the 3/4 BA gains in the three problematic outer populations
(V10: `0.005024`, `0.008848`, `0.008362`). This is negative evidence against
further threshold/L2 retuning of the same hard-flip family on this train set.

## Final nested metrics

| Metric | Exact R0 | V11 nested selected | Improvement |
| --- | ---: | ---: | ---: |
| Macro continuous RMSE | 0.568780 | 0.563918 | +0.004862 |
| Equal-band RMSE | 0.691549 | 0.683582 | +0.007967 |
| Low-tail `{1,2}` RMSE | 0.923335 | 0.907438 | +0.015897 |
| Score-5 RMSE | 0.884190 | 0.871454 | +0.012736 |
| True-gold 3/4 balanced accuracy | 0.643313 | 0.645858 | +0.002545 |
| Macro Spearman | 0.600288 | 0.604690 | +0.004401 |

Axis RMSE gains were content `0.001733`, organization `0.003980`, and
expression `0.008873`. The candidate-minus-R0 macro-RMSE interval was
`[-0.008286, -0.001458]`. The final gate passed the bootstrap, both-tail,
no-axis-regression, and Spearman checks, but failed macro-gain and 3/4-BA-gain.
Score 1 remained descriptive only.

## Execution and artifacts

Fourteen scoped tests passed before launch, together with Python compilation,
shell syntax, bound-input validation, and launcher progress parsing. GPU0
then passed a real 96-train/32-predict smoke. Outer folds 0--3 ran concurrently
on physical GPUs 0--3; outer fold 4 followed on GPU0. The append-only ledger
records the exact scope. The workload is a small float64 ridge/LBFGS study, so
GPU allocation was valid but sustained utilization was not expected.

- Task-card SHA-256:
  `79ca80fdd344b979a52cbbd1b211cfb7ed20c29672b64a03a40f74d5ed6d767c`.
- Smoke SHA-256:
  `c6f8cc4a26719d713a51a946468e53ae6250d8dfb0315c1269bb1034c0966858`.
- Aggregate SHA-256:
  `0a8cd65cfe6e688641abd1ed72fd9bbcc1ba5b5c6ecb0273eda1d513f94e2af2`.
- Completion SHA-256:
  `a7e75d8829955233c943933da9edf5c4b3e60f08eaf882254b30fd9ada21faf1`.
- Retained restricted prediction SHA-256:
  `36059045be7b2824e8e2f775495e4dda84a817dcdcc65eb19ac7d7696956bcc9`.

