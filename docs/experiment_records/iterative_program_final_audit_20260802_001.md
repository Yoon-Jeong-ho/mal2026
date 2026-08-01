# Fixed train-only iterative score program: final audit

Status: completed; no challenger passed the fixed final gate; exact R0 retained

Date: 2026-08-02 (Asia/Seoul)

## Scope

This record closes the authorized fixed-initialization train-only improvement
program. It covers the exact R0 reproduction, the original twenty-candidate
learner study, subsequent leakage-safe falsifications, new score-blind Terra
and Luna evidence, dedicated 3/4 heads, and actual rationale-semantic Qwen3
features. Every named follow-up used a new preregistration rather than
continuing a prior checkpoint or retuning a failed inventory.

All model selection used the canonical 2,000-row train population with sealed
5-outer x 4-inner cross-fitting. Validation was not loaded for these runs and
the `average` target was never trained or evaluated. Score 1 was descriptive
only; `{1,2}` was the low-tail gate. Row-level essays, rationales, embeddings,
and predictions remain in ignored restricted storage.

## Fixed reference and gate

- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`.
- Fold fingerprint:
  `8c6eed86944f3f75666da746bb5f43d5e4d435fc781a923a349fd58a88a2c6db`.
- Exact R0 macro continuous RMSE: `0.5687802169918456`.

The unchanged inner AND gate required macro gain at least 0.005, equal-band
gain at least 0.010, improvement in both tails, 3/4 balanced-accuracy gain at
least 0.010, no axis worsening above 0.010, and Spearman fall at most 0.005.
The final gate required macro gain at least 0.010 and a paired 10,000-resample
candidate-minus-R0 95% interval strictly below zero, plus the safety gates.

## Result chronology

| Phase | Material change | Nested selected RMSE | Macro gain | 3/4 BA gain | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| V1--V6 | frozen essay/rationale embedding, hash evidence, twenty fresh learners, tail routers | 0.568780 | 0 | 0 | R0 fallback |
| V7 | new score-blind Terra participant scores | 0.563253 | +0.005528 | +0.003271 | final gate fail |
| V8 | smooth Terra boundary heads | 0.565515 | +0.003265 | +0.003018 | final gate fail |
| V9 | selective Terra boundary flips | **0.563067** | **+0.005714** | **+0.004620** | final gate fail |
| V10 | new score-blind Luna source + Terra/Luna disagreement | 0.563717 | +0.005063 | +0.004180 | final gate fail |
| V11 | class-balanced Terra/Luna 3/4 head | 0.563918 | +0.004862 | +0.002545 | final gate fail |
| V12 | actual Terra/Luna rationale-only Qwen3 semantics | 0.568780 | 0 | 0 | all outer folds R0 fallback |

V9 produced the best sealed nested development RMSE, and its bootstrap
interval excluded zero, but it still achieved only about half the required
macro and 3/4 gains. It was not promoted. V10 showed that a second independent
agent source improved all core metrics directionally but not enough across
folds. V11's class balancing reduced the problematic folds' 3/4 signal. V12's
new semantic ridge improved score 5 but substantially worsened macro RMSE,
3/4 separation, and Spearman, so no outer population selected it.

The disclosed V7 full-OOF prestudy near `0.5549` is adaptive exploration and
is not a sealed nested result or a valid final candidate. It must not be
reported as confirmed performance.

## Interpretation

The repeated bottleneck is not absence of a score-5 continuous signal. Several
families improved score 5 and sometimes the low tail. The failure is stable
cross-fold center/boundary separation: improvements in tails traded against
3/4 balanced accuracy, macro RMSE, or rank correlation. Adding more score
sources, hard flips, class balancing, and rationale semantics did not break
that trade-off under the fixed gates.

Therefore the scientifically valid output of this program is a negative
result and the retained exact R0 reference, not a weaker challenger selected
because it looks better on an already reused population. The local
`validation.jsonl` may still be used for clearly labelled descriptive
analysis, but repeated prior exposure means it cannot serve as an untouched
independent confirmation set. A hidden benchmark result or newly isolated
labels are required for such a claim.

## Completion evidence

- All fixed candidates started from the same declared initialization within
  their round; no checkpoint continuation was used.
- GPU0 smoke preceded every multi-GPU stage. GPU 0--3 scope and authorization
  were recorded in append-only ledgers; no pre-existing process was displaced.
- The last new feature build processed 36,000 rationale-only texts on GPUs
  0--3 and the sealed V12 evaluation completed all 15 outer-candidate jobs.
- V12 completion SHA-256:
  `04bc189e7fdb6802b9fcbb810178efed213de96191f5ea097ed9249a221f3d94`.
- V12 aggregate SHA-256:
  `e81b7b15f192933a400c58f97aad3496c3411df126904b811b67abc1aa23a487`.

Detailed metrics, negative results, commands, inputs, and checksums remain in
the per-run records linked from the repository README.
