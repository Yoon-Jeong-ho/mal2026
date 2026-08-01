# Tail score remediation v4: final nested twenty-router study

Run ID: `iterative-tail-router-v4-20260801-001`

Status: completed; 20/20 routes were fit inside each of five sealed outer
folds (100/100 route fits); no route passed the inner gate in any outer fold;
the exact R0 OOF baseline was retained and same-train model search was frozen
by the preregistered stop rule

Date: 2026-08-02 (Asia/Seoul)

## Scope and decision

V4 was the final adaptive train-only attempt after observing v1--v3. It tested
five score-only routing families with four fixed variants each. Three agent
roles separately reviewed metrics, model design, and split isolation. Their
aggregate recommendations were used only to preregister the experiment; agent
text and judgments were not model features, pseudo-targets, rewards, or
weights.

The result is negative. Every outer fold selected the exact R0 baseline because
none of its 20 inner-OOF routes passed all seven gates. Consequently the five
outer predictions exactly equal R0, final macro RMSE remains `0.568780`, the
10,000-resample candidate-minus-baseline interval is `[0, 0]`, and the final
gate fails. The preregistered action is therefore
`freeze_all_same_train_model_search`.

This is adaptive same-train descriptive evidence, not independent
confirmation. It supports no validation, generalization, deployment, or
leaderboard claim.

## Immutable execution contract

- Git SHA at launch: `138aa0f590f87f44c311ed3bd08237d25fb04240`
- Config SHA-256:
  `0955ad8a1c03aff51057600dfc782d6f1c71b4580375f018bd088cca583ceb60`
- Model source SHA-256:
  `fd42e537a97833e46cfade5f67a0b0dce1ad305ee2af044bbfe91e798b54ff69`
- Protocol source SHA-256:
  `c26222a3b2d373041c2285b53082767aafe861a097baac91be70ad2513cee1c4`
- Runner source SHA-256:
  `8dab4b42ab2a7da128141d2923e98d4946d22c560a791377f59b75b576df145d`
- Launcher source SHA-256:
  `b7ab3dd89b83af46dffa0aec129854d2572fa190b0f881348bc22619883e9de1`
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- Historical v2 aggregate SHA-256:
  `bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f`
- Historical v3 aggregate SHA-256:
  `bc44123673b512ae20212a94f7a3e98b4d227823cd1d74d2fddf8f4a07e3667f`
- Records: 2,000 train rows, five fixed folds of 400.
- Targets: content, organization, expression only; average forbidden.
- Evidence: local score-blind feature cache; historical v1--v3 row
  predictions, validation, and external APIs forbidden.

## Nested isolation

For outer fold `O`, each inner validation fold `D` used `S`, the other three
folds. A fresh R16 teacher was cross-fit 2-of-3 inside `S`; neither `D` nor `O`
entered that training. Fresh alpha-100 R17 and direct-evidence ridges, v3
`selective_hurdle-v1`, and v3 `soft_routed_residual-v4` were fit on `S` and
predicted `D`. The four `D` outputs formed exactly-once component OOF banks over
the 1,600 outer-train rows.

All twenty routes were fit and gated against R0 using only those banks. The
selection rule was eligible minimum macro RMSE with route-number tie-break and
exact-baseline fallback. Only after the route and aggregate weights froze
could fresh 3-of-4 R16/R17/direct/hurdle/soft outer components be fit and `O`
predicted once. In practice every fold fell back, so no non-baseline outer
refit was needed. Outer gold was opened only after prediction.

The inner gate was an AND of macro gain at least `.005`, equal-group gain at
least `.010`, both `{1,2}` and score-5 RMSE improvement, 3/4 balanced-accuracy
gain at least `.010`, no axis worsening over `.010`, and Spearman fall at most
`.005`. Score 1 was descriptive only. The final gate strengthened macro gain
to `.010` and also required the candidate-minus-baseline RMSE bootstrap CI
upper bound below zero.

## Execution

The existing `.venv-standard` environment was used without installation. GPU0
passed a real two-epoch neural/component smoke and one fit/apply smoke for each
router family. A read-only preflight found GPUs 0--3 empty; outer folds 0--3
then ran concurrently on GPUs 0--3, and fold 4 ran on the first freed GPU
(GPU1). No process was terminated or displaced.

Commands:

```bash
.venv-standard/bin/python scripts/run_iterative_tail_router.py --smoke
.venv-standard/bin/python scripts/run_iterative_tail_router.py --launch
```

The full preflight-to-aggregate stage took about 177 seconds. GPU memory was
about 1,023 MiB per active worker and sampled utilization was often zero even
while workers consumed roughly 1.7--2 CPU cores. This is expected for the
small R16 networks followed by dominant NumPy ridge/router searches; it is not
a vLLM-style memory-bound workload. Four GPUs still reduced the first four
outer folds to concurrent execution.

## Twenty-route inner evidence

The table averages baseline-relative inner-OOF improvements across the five
overlapping 1,600-row outer-train populations. Positive values mean better.
These are selection diagnostics, not five independent estimates.

| Cycle | Router | RMSE | Equal group | `{1,2}` | Score 5 | 3/4 BA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | low-protected sigmoid v1 | +0.000210 | +0.000272 | +0.000136 | +0.000857 | +0.000763 |
| 2 | low-protected sigmoid v2 | +0.000256 | +0.000360 | +0.000029 | +0.001311 | +0.001078 |
| 3 | low-protected sigmoid v3 | +0.000281 | +0.000396 | -0.000025 | +0.001526 | +0.001156 |
| 4 | low-protected sigmoid v4 | +0.000324 | +0.000434 | +0.000116 | +0.001537 | +0.000972 |
| 5 | four-zone hard v1 | +0.000275 | +0.000625 | +0.001171 | +0.003192 | +0.001147 |
| 6 | four-zone hard v2 | +0.000291 | +0.000727 | +0.001889 | +0.002922 | +0.001177 |
| 7 | four-zone hard v3 | +0.000321 | +0.000750 | +0.002100 | +0.002874 | +0.001589 |
| 8 | four-zone hard v4 | **+0.000341** | **+0.000774** | **+0.002299** | +0.002838 | **+0.001685** |
| 9 | boundary overlay v1 | +0.000117 | +0.000227 | +0.000475 | +0.000475 | +0.001106 |
| 10 | boundary overlay v2 | +0.000177 | +0.000339 | +0.000646 | +0.000773 | +0.001275 |
| 11 | boundary overlay v3 | +0.000208 | +0.000439 | +0.000907 | +0.000953 | +0.001261 |
| 12 | boundary overlay v4 | +0.000205 | +0.000438 | +0.000842 | +0.001058 | +0.001217 |
| 13 | sigmoid four-expert v1 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 14 | sigmoid four-expert v2 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 15 | sigmoid four-expert v3 | +0.000059 | +0.000092 | +0.000063 | +0.000262 | +0.000765 |
| 16 | sigmoid four-expert v4 | +0.000111 | +0.000171 | +0.000155 | +0.000459 | +0.000795 |
| 17--20 | formal-gate lattice; identity fallback | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

Cycle 8 had the largest mean macro gain, but only three of five outer-train
populations improved low-tail RMSE and only two improved score-5 RMSE. Cycle
11 was directionally more stable: macro, low-tail, high-tail, and 3/4 BA all
improved in all five populations, but its gains were only `0.000208`,
`0.000907`, `0.000953`, and `0.001261`. Neither pattern approached the frozen
macro/equal-group/BA thresholds.

Across all 100 route fits, the best single-fold gains were macro `0.000547`,
equal-group `0.001298`, and 3/4 BA `0.003615`, still below `.005`, `.010`, and
`.010`. Large isolated tail gains existed (low `0.008845`, high `0.012177`),
but never with the full joint gate. Strict eligible route count was zero.

## Scores 1, 2, 3, 4, and 5

Cycle 8's mean band diagnostics make the remaining trade-off explicit:

| Gold score | RMSE improvement | Recall change |
| ---: | ---: | ---: |
| 1 | +0.003384 | +0.000000 |
| 2 | +0.002386 | +0.013003 |
| 3 | -0.000048 | -0.002498 |
| 4 | -0.001992 | +0.005868 |
| 5 | +0.002838 | +0.000000 |

Thus it slightly improved continuous error at scores 1, 2, and 5 and improved
score-2 recall, but did not recover score-1 or score-5 integer recall. It also
worsened continuous score-4 error and score-3 recall. Its mean 3/4 BA gain was
only `0.001685`, about one sixth of the required `0.010`, and the tail
directions were not fold-stable.

Because all outer folds correctly fell back, the actually selected final
model has no band change. R0's macro band RMSE/recall remains: score 1
`1.164567/0.030303`, score 2 `0.896752/0.241179`, score 3
`0.504166/0.647305`, score 4 `0.454505/0.639320`, and score 5
`0.884190/0.000000`. Score-1 counts remain only 11/26/4 by axis, so no
standalone score-1 promotion claim is made.

## Final metrics and artifacts

- R0 and nested-selected macro RMSE: `0.5687802169918456`
- Equal-group RMSE: `0.6915490515990893`
- Low `{1,2}` RMSE: `0.9233348764858816`
- Score-5 RMSE: `0.8841899261902623`
- 3/4 balanced accuracy: `0.6433125889620798`
- Spearman: `0.6002884386652662`
- 3-to-4 rate: `0.3058362751501927`
- 4-to-3 rate: `0.35696465039277264`
- Final improvement: exactly zero on every metric.
- Candidate-minus-baseline 95% bootstrap CI: `[0, 0]`; required upper `< 0`.

Ignored artifacts:

- Runtime root:
  `outputs/iterative-tail-router-v4/iterative-tail-router-v4-20260801-001`
- Restricted predictions:
  `data/processed/restricted/iterative_tail_router_v4/iterative-tail-router-v4-20260801-001`
- Aggregate SHA-256:
  `5c34e53706b1e2bc90ea24a402c9f9efc444549ba2ce12f01e7ac481bd978279`
- Completion SHA-256:
  `2e52992e9dd1dc80ff15f1179ff2f59228cd733eeb4f86ef8f4f50ebcf54c720`
- Smoke SHA-256:
  `c6bd466a38ead26e127056a14771af592c72307743cdabf2639d6a69d2ac9e42`

All five restricted files contained 400 unique rows and covered all 2,000
records exactly once. All result checksums matched, the public tree contained
no `source_id`, and 15 focused v4 tests plus `py_compile` and `git diff
--check` passed after execution. No validation data or external API was used.

The negative result is preserved. Further tuning on these same train folds is
stopped; a scientifically meaningful next comparison requires an untouched
evaluation set and a separately authorized protocol.
