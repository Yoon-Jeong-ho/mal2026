# Official Terra agent-score stack (V7)

Run ID: `iterative-official-agent-stack-v7-20260802-001`

Status: completed; strict final gate failed, exact R0 retained

Date: 2026-08-02 (Asia/Seoul)

## Why this study was allowed after V6

V6 froze further tuning on the same train population **and the same feature
sources**. V7 did not reopen any V4--V6 candidate or learned state. It added a
materially new source: three previously completed GPT-5.6-Terra participant
outputs per train essay, produced with the official scoring prompt. Their
manifest states that no human or reference score was read or prompted. V7
used only the three model-predicted axis-score vectors and deterministic
consensus/disagreement summaries; rationale text was not used.

The user's original fixed 20-round program had already completed as V1,
including exact R0, residual, ordinal, tail weighting, 3/4, score-blind agent
evidence, ensemble, and calibration rounds. V7 is a separately named adaptive
follow-up, not a replacement or continuation of a V1 checkpoint. Every V7
ridge system was solved afresh from the same raw feature construction.

## Adaptive prestudy disclosure

Before V7 preregistration, a full five-fold train-OOF grid inspected six ridge
penalties by five correction caps. The full 30-cell aggregate is preserved at
the ignored path
`outputs/iterative-official-agent-stack-v7-prestudy/iterative-official-agent-stack-v7-prestudy-20260802-001/aggregate.json`
(SHA-256 `4b66a91344195a5b1e8f40d4816052ea8b6fbc032780dbaa3f0721efac1083eb`).
It wrote no row predictions and did not load validation.

The grid's best macro RMSE was `0.554917`. The preregistered primary cell,
alpha `10` and correction cap `0.5`, reached `0.554928` versus exact R0
`0.568780`. On that adaptively inspected OOF population it improved macro
RMSE by `0.013853`, equal-band RMSE by `0.020806`, low-tail RMSE by
`0.036087`, score-5 RMSE by `0.034626`, and 3/4 balanced accuracy by
`0.012349`; all three axis RMSE values and macro Spearman also improved.

This prestudy is why V7 cannot be independent confirmation even though V7
subsequently used sealed nested fitting. Its role is candidate-design evidence
only.

## Frozen V7 protocol

- Preregistration commit: `04fc3a8`; mechanical smoke-audit repair:
  `c6dbec6`.
- Train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`.
- Official Terra candidate manifest SHA-256:
  `960ae42ac19e79bd8cff747844ee58a5c724abe30d46d6f3865cb92e22b9de53`.
- Candidate rows SHA-256:
  `a1791c418c79c0b76399ddb993e862f34209c2da95b0c13f7cda87f403a24e4c`.
- Official system prompt SHA-256:
  `ea0454665da8e13ffb606c1b0fc7f8323bd62dbad65a9a363e463df888bbe5e9`.
- Model: `gpt-5.6-terra`; 6,000 valid outputs, three per each of 2,000
  essays; no human/reference score prompted.
- V7 made zero new external API calls and loaded neither validation nor the
  `average` target.

The fixed 39-dimensional feature order was three candidates x three axis
scores, per-axis mean/std/min/max, three pairwise equality vectors, and the
nine candidate-minus-R0 differences. Its complete feature-matrix SHA-256 was
`fada96b9bf4829ee649e059e35d7a341ef377a31ca91832975e4d3abddbb765f`.

Three candidates were frozen: alpha 10/cap 0.5 primary, alpha 0.1/cap 0.3
diverse, and alpha 10/cap 0.1 conservative. For outer fold `O`, each of the
other four folds served once as inner validation `D`; the remaining three
folds were training `S`. All three candidates completed four inner predictions
before the original seven-gate selection. Only the selected candidate was
freshly refit on all 1,600 outer-train rows and predicted `O` once. No
checkpoint was reused.

The exact original inner gate was retained: macro gain at least `0.005`,
equal-band gain at least `0.010`, both tails improve, 3/4 balanced-accuracy
gain at least `0.010`, no axis worse by more than `0.010`, and Spearman not
down by more than `0.005`. Score-5 recall was reported but was not added as an
eighth gate.

## Execution

GPU0 passed a 96-train/32-predict real smoke. The full run launched outer
folds 0--3 concurrently on physical GPUs 0--3, then outer 4 on GPU0. Ledger
times were `1785601850.50` for the four launches, `1785601860.52--.53` for
their completion, `1785601860.54` for outer 4 launch, and `1785601868.54` for
outer 4 completion. These are 39-by-39 float64 closed-form solves, so using
more GPU memory or padding the batch would add waste rather than throughput.

The first smoke stopped before full execution because its audit incorrectly
required unique coefficient hashes. Cycles 1 and 3 intentionally share alpha
10 and differ only in post-solve cap, so identical coefficients are correct.
Commit `c6dbec6` changed only that audit from uniqueness to a valid SHA-256
shape check; data, candidates, parameters, gates, and predictions were
unchanged. The failure and successful replay remain in the append-only ledger.

## Inner selections

| Outer fold | Selected | Primary inner macro gain | Primary equal-band gain | Primary low gain | Primary high gain | Primary 3/4 BA gain | Failed primary gates |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | primary | 0.012346 | 0.017978 | 0.020600 | 0.035202 | 0.012514 | none |
| 1 | primary | 0.013206 | 0.021284 | 0.025272 | 0.052579 | 0.015983 | none |
| 2 | R0 fallback | 0.010166 | 0.014021 | 0.015747 | 0.030089 | 0.005649 | 3/4 BA |
| 3 | R0 fallback | 0.011210 | 0.019769 | 0.054153 | 0.024025 | 0.000752 | 3/4 BA |
| 4 | primary | 0.016511 | 0.025449 | 0.063005 | 0.026717 | 0.010909 | none |

All primary cells improved macro, equal-band, both tails, all axes, and
Spearman. Folds 2 and 3 failed solely because their 3/4 BA gains were below the
frozen `0.010` threshold. The protocol therefore used exact R0 on those two
outer folds rather than relaxing the gate after seeing results.

## Final nested result

The three candidate-selected outer folds plus two exact-R0 fallbacks produced:

| Metric | R0 | Nested selected | Improvement |
| --- | ---: | ---: | ---: |
| Macro continuous RMSE | 0.568780 | 0.563253 | +0.005528 |
| Equal-band RMSE | 0.691549 | 0.683114 | +0.008435 |
| Low-tail `{1,2}` RMSE | 0.923335 | 0.907099 | +0.016236 |
| Score-5 RMSE | 0.884190 | 0.868571 | +0.015619 |
| True-gold 3/4 BA | 0.643313 | 0.646583 | +0.003271 |
| Macro Spearman | 0.600288 | 0.605431 | +0.005143 |

Axis RMSE gains were content `0.002328`, organization `0.006344`, and
expression `0.007910`; no axis regressed. The paired 10,000-resample
candidate-minus-baseline RMSE CI was `[-0.009417, -0.001648]`, so the observed
gain was statistically direction-consistent on this train population.

Nevertheless, the final gate failed because macro gain was below `0.010` and
3/4 BA gain was below `0.010`. Equal-band gain was also only `0.008435`, though
the final gate did not separately require it. Score-5 integer recall gain was
zero. The retained protocol-valid prediction is therefore exact R0, not the
`0.563253` nested development stack.

## Interpretation

This is the first new feature source in the iterative series to give a clear,
same-direction global signal: the CI excludes zero, every axis improves, and
both tails improve without a material tail-risk fold. The remaining weakness
is highly specific: the linear residual stack does not make its 3/4 boundary
gain stable across fold populations. A subsequent study may test a separately
registered 3-vs-4 or `>=4` auxiliary head on the frozen official-agent
features, but may not reopen these three V7 cells or lower the gate.

No validation, hidden-test, deployment, or leaderboard improvement is claimed.
The Terra API features would also require a later local distillation or an
allowed inference-time API contract before deployment.

## Verification and artifacts

Fifteen focused tests passed before launch; Python compilation, JSON parsing,
and `git diff --check` passed. Runtime checksums:

- task card: `f329c07408feab167c0c4c4678934fed0f825a9988a99759aba8920f1e6205b8`;
- smoke: `e3bd7e393df9456ae4761cb1c887d4e34f904d67543f7bf1969d722ae33ca9dc`;
- aggregate: `2cbbfde7a07db8f9844458a722541a50731193a6fd9632e6f2633282509ed68c`;
- completion: `7ac54a9f24f564e84ba2c9f4fd135aadf1586917113e7876560ff817da4cc371`;
- final retained restricted prediction:
  `36059045be7b2824e8e2f775495e4dda84a817dcdcc65eb19ac7d7696956bcc9`.

Each of five restricted outer files contains exactly 400 rows and the final
restricted file contains 2,000. A recursive audit found no identifiers,
essays, prompts, rationale text, or writing text in the public artifact tree.
