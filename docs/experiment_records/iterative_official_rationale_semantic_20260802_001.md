# Terra + Luna rationale-semantic falsification (V12)

Run ID: `iterative-official-rationale-semantic-v12-20260802-001`

Status: completed; all outer selections failed closed to exact R0; terminal
same-train adaptive stop

Date: 2026-08-02 (Asia/Seoul)

## Decision

V12 tested the last materially distinct information channel authorized by the
twenty-round plan: actual frozen Qwen3 embeddings of the score-blind Terra and
Luna rationale text. V7--V11 had used their participant scores but not their
rationale semantics. The exact feature transform, three candidates, nested
protocol, gates, and terminal failure action were committed before feature
generation. The generated artifact was then attached by a checksum-only
binding commit.

None of the three candidates passed the unchanged seven-gate conjunction in
any of the five outer populations. The semantic-only learner worsened mean
inner macro RMSE by `0.040285`; adding the prior 96 structured features still
worsened it by `0.024304`. The balanced 3/4 head also failed. Therefore every
outer fold used exact R0, final macro RMSE remained `0.568780`, and the final
paired interval was `[0,0]`. Exact R0 remains the protocol-valid model.

This is strong negative same-train nested evidence against further tuning of
the current score-derived or rationale-semantic families. In accordance with
the preregistered terminal stop, no additional threshold, projection, alpha,
prompt, or same-train candidate search is warranted. Independent new labels
or a genuinely untouched evaluation population would be required to resume a
confirmatory model-selection claim.

V12 is not independent validation or a hidden-test, leaderboard,
generalization, or deployment result.

## Target-blind semantic artifact

Artifact run ID: `iterative-official-rationale-embeddings-v12-20260802-001`

- Input: 2,000 train essays x Terra/Luna x three candidates x three axes =
  36,000 rationale texts.
- Rendered text: rationale alone. Candidate score, essay, writing prompt,
  human/reference score, and gold label were not included.
- Encoder: frozen `Qwen/Qwen3-Embedding-8B` at revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
- Pooling: last non-padding token, float32 L2 normalization; max length 2,048.
- Token audit: 5,439,405 tokens, maximum 282, zero truncated texts.
- For each source/axis, the three candidate embeddings were averaged and
  renormalized. A fixed data-independent Rademacher matrix projected
  4,096 dimensions to 32 (seed `2026080212`; matrix SHA-256
  `8095c0e0cfcae32002b7e0d0f1c690336612cc7012ef18e0b6b13fa1c2dfd234`).
- Each axis contributed projected pooled-centroid mean (32), projected signed
  Terra-minus-Luna centroid difference (32), and three agreement cosines, for
  67 dimensions per axis and 201 dimensions total.
- Feature matrix SHA-256:
  `a23ac6a02b1941474e6f40ea82739a48671f698a0b0f45b881e5d7025c7f9fb3`.
- Restricted feature rows SHA-256:
  `76a9329995d3e5352ddff5f42ff20e9d8bc0e0aba1670d968e61341714392c6d`.
- Public/restricted manifest SHA-256:
  `7a8dc097bde73ad8aa263d48a805cf8cdb24c35d35cc4820482669d59c8a6f1f`.

GPU0 passed a two-essay/36-text smoke. The four 500-essay shards then ran on
physical GPUs 0--3 with batch 64. A runtime sample showed 20.3--24.8 GiB and
55--100% utilization per GPU. All four shards exited zero in about 89 seconds;
the CPU merge then completed without truncation or population drift.

## Frozen candidates and protocol

1. 201-dimensional semantic residual ridge, alpha 10 and correction cap 0.5.
2. 297-dimensional fusion of V10's structured 96 features and semantic 201,
   with the same ridge and cap.
3. The same fusion ridge plus a fresh equal-total-class-weight gold-3-vs-4
   logistic head (L2 0.01, confidence 0.55, window 0.20, hard 3.499/3.501
   flip).

Every ridge, feature standardizer, and optional head was fresh per candidate
and fold. Standardization was fit on the current fit partition only. The
5-outer x 4-inner isolation, exact-three completion barrier, original seven
inner gates, fallback rule, and final 10,000-resample bootstrap gate were
unchanged. Validation and the `average` target were not loaded. V12 itself
made zero API calls and reused no checkpoint or historical row prediction.

## Inner results

Values below average baseline-relative improvements over the five 1,600-row
outer-train populations. Positive is better.

| Candidate | Macro RMSE | Equal band | `{1,2}` | Score 5 | 3/4 BA | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Semantic 201 ridge | -0.040285 | -0.019872 | -0.001960 | +0.046847 | -0.036248 | -0.053825 |
| Fusion 297 ridge | -0.024304 | -0.003041 | +0.017679 | +0.060629 | -0.030387 | -0.035960 |
| Fusion + balanced 3/4 | -0.025000 | -0.004191 | +0.014530 | +0.059250 | -0.036279 | -0.038648 |

Every candidate failed macro RMSE, 3/4 balanced accuracy, the axis guard, and
the Spearman guard in all five populations. Semantic 201 also failed the low
tail in three populations; both fusion variants failed it in two. The new
semantic channel consistently helped score 5, and the fusion often helped the
low tail, but these gains came with large center/ranking degradation. The
balanced boundary head reduced 3/4 accuracy further rather than rescuing it.

## Final result and artifacts

Because all outer selections fell back, the final nested prediction equals
exact R0:

- Macro continuous RMSE: `0.5687802169918456`.
- Equal-band RMSE: `0.6915490515990893`.
- Low-tail `{1,2}` RMSE: `0.9233348764858816`.
- Score-5 RMSE: `0.8841899261902623`.
- True-gold 3/4 balanced accuracy: `0.6433125889620798`.
- Macro Spearman: `0.6002884386652662`.
- Candidate-minus-R0 bootstrap interval: `[0,0]`.

GPU0 passed the real 96-train/32-predict learner smoke. Outer folds 0--3 then
ran concurrently on physical GPUs 0--3; outer fold 4 followed on GPU0. This
stage used small float64 ridge/LBFGS matrices, so low sustained GPU memory and
utilization were expected after the embedding stage.

- Preregistration commit: `1c2fd79`.
- Checksum-only binding commit: `ecd5c0e`.
- Post-binding test adjustment commit: `e360bc3`.
- Task-card SHA-256:
  `fa58dc5b88f5665000bc7046013cc52a8edc4adc2d385becaa92be857add80e4`.
- Learner smoke SHA-256:
  `942594f807c410be5ab5e3f9f743e303d7367743683ac65a1413074e9cc09d25`.
- Aggregate SHA-256:
  `e81b7b15f192933a400c58f97aad3496c3411df126904b811b67abc1aa23a487`.
- Completion SHA-256:
  `04bc189e7fdb6802b9fcbb810178efed213de96191f5ea097ed9249a221f3d94`.
- Retained restricted prediction SHA-256:
  `36059045be7b2824e8e2f775495e4dda84a817dcdcc65eb19ac7d7696956bcc9`.

Twenty-two scoped V12 tests passed after binding, together with Python
compilation, JSON and shell validation, canonical input validation, launcher
progress parsing, and checksum verification.
