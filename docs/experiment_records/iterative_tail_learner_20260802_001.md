# Tail score learner v5: fresh nested twenty-candidate study

Run ID: `iterative-tail-learner-v5-20260802-001`

Status: completed; 20/20 candidates were evaluated in each of five sealed
outer folds (100 candidate OOF evaluations, 400 fresh inner fits). No
candidate passed the seven-gate inner conjunction in any outer fold, so exact
R0 was retained in all five folds and the final gate failed.

Date: 2026-08-02 (Asia/Seoul)

## Scope and authorization

This is a separately named adaptive train-only experiment authorized after
V4. The authorization clarified that V4 failure froze the V4 candidate
inventory and learned state, not the whole project. V4 artifacts were not
changed or used as row-level inputs. V5 tested five new GPU learner families,
four fixed variants per family, over the frozen Qwen3-Embedding-8B train-OOF
embedding, exact R0 OOF score, and fixed score-blind rationale hash features.

The result is negative. It is descriptive same-train nested evidence, not an
independent validation, generalization, deployment, or leaderboard result.
No validation file, external API, `average` target, historical row prediction,
historical learned weight, checkpoint, or pseudo-target was used.

## Immutable execution contract

- Git SHA at launch: `40d7e05280f3461a9cca790d4afef770142d547b`
- Config SHA-256:
  `1132157d9e722c3db6920f695786a7788d1774012c86f822cafa448c21ea7a68`
- Model source SHA-256:
  `f54172c2b5d136da28f9e93e287eb4267d624b4d8c53bdc9982e9e3fa4585e1a`
- Protocol source SHA-256:
  `14a1f1ef93dfeb4de4f61117ead7d25a16b833239069ee7a8f035724ddb1c63d`
- Runner source SHA-256:
  `5794fa7226184ab4c42ce46053286b1a2e78b8f2d964dce9c465cf5c86f27799`
- Selection source SHA-256:
  `95e5fc47d7f0ae4fc378d2b37baa8627f29d09aad3f78a34c454273f57c58852`
- Launcher source SHA-256:
  `cd233c9ed03469b557e19efd21723714381a7f6275042b65ef804e5b20bf39a6`
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- Score-blind cache SHA-256:
  `c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db`
- Historical V4 aggregate SHA-256:
  `5c34e53706b1e2bc90ea24a402c9f9efc444549ba2ce12f01e7ac481bd978279`
- Seed: `2026080205`.
- Environment: existing `.venv-standard`; Python 3.12.3, PyTorch
  2.11.0+cu130, NumPy 2.3.5, cuDNN 91900.
- Hardware: four NVIDIA H100 80 GB GPUs, physical scope 0--3.
- Targets: `content`, `organization`, and `expression` only.

## Nested isolation and selection

For each outer fold `O`, each candidate was freshly initialized and fit four
times. For inner validation fold `D`, training set `S` was the other three
folds, so neither `D` nor `O` entered fitting. Concatenating the four `D`
predictions produced exactly-once OOF predictions for all 1,600 outer-train
rows. All 20 candidates completed before selection.

The inner gate required all of the following against exact R0: macro RMSE
gain at least `.005`, equal-group RMSE gain at least `.010`, improvement in
both pooled score `{1,2}` and score-5 RMSE, gold-3/4 balanced-accuracy gain at
least `.010`, no axis RMSE worsening over `.010`, and macro Spearman fall at
most `.005`. Score 1 remained descriptive only. No candidate passed, so no
outer learner refit occurred and every outer prediction was exact R0.

The final concatenation was evaluated once. Its gate additionally required a
macro RMSE gain of at least `.010` and a 10,000-resample paired
candidate-minus-baseline RMSE interval with upper bound below zero. No
selection was performed after outer predictions were concatenated.

## Candidate results

The table averages baseline-relative inner-OOF improvements over the five
overlapping 1,600-row outer-train populations. Positive values are better.

| Cycle | Candidate | Macro RMSE | Equal group | `{1,2}` | Score 5 | 3/4 BA | Spearman |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | anchored multitask v1 | -0.010025 | -0.002381 | +0.012248 | +0.018402 | -0.007562 | -0.018159 |
| 2 | anchored multitask v2 | -0.019155 | -0.010386 | -0.000243 | +0.022401 | -0.014037 | -0.030475 |
| 3 | anchored multitask v3 | -0.029094 | -0.018527 | -0.002397 | +0.017699 | -0.019400 | -0.042386 |
| 4 | anchored multitask v4 | -0.040692 | -0.029750 | -0.018099 | +0.015912 | -0.029415 | -0.056695 |
| 5 | R0-anchored distributional v1 | **-0.000404** | -0.003184 | -0.028734 | +0.012986 | **+0.005119** | **-0.003398** |
| 6 | R0-anchored distributional v2 | -0.002425 | -0.005027 | -0.038002 | +0.018948 | +0.004527 | -0.007701 |
| 7 | R0-anchored distributional v3 | -0.006012 | -0.009043 | -0.051031 | +0.023333 | +0.003529 | -0.012567 |
| 8 | R0-anchored distributional v4 | -0.012126 | -0.015225 | -0.066398 | **+0.027048** | +0.003493 | -0.021540 |
| 9 | joint tail-boundary hurdle v1 | -0.012907 | -0.008959 | +0.002438 | +0.005751 | -0.007988 | -0.025981 |
| 10 | joint tail-boundary hurdle v2 | -0.023741 | -0.018469 | -0.007194 | +0.001883 | -0.013088 | -0.041929 |
| 11 | joint tail-boundary hurdle v3 | -0.033876 | -0.026596 | -0.013060 | +0.000319 | -0.021319 | -0.054854 |
| 12 | joint tail-boundary hurdle v4 | -0.044776 | -0.036476 | -0.019241 | -0.000253 | -0.033241 | -0.069348 |
| 13 | axis-coupled low-rank MoE v1 | -0.009226 | -0.006371 | -0.006681 | +0.011288 | -0.004114 | -0.019364 |
| 14 | axis-coupled low-rank MoE v2 | -0.017546 | -0.013367 | -0.008732 | +0.009410 | -0.011760 | -0.032129 |
| 15 | axis-coupled low-rank MoE v3 | -0.026600 | -0.019738 | -0.011341 | +0.009933 | -0.015858 | -0.043380 |
| 16 | axis-coupled low-rank MoE v4 | -0.035836 | -0.026716 | -0.019297 | +0.013145 | -0.024020 | -0.055169 |
| 17 | band-risk Pareto residual v1 | -0.010834 | -0.004466 | +0.011834 | +0.011242 | -0.010667 | -0.020910 |
| 18 | band-risk Pareto residual v2 | -0.018238 | -0.010124 | +0.001883 | +0.018528 | -0.011330 | -0.030185 |
| 19 | band-risk Pareto residual v3 | -0.028349 | -0.018910 | -0.001492 | +0.014038 | -0.016114 | -0.041991 |
| 20 | band-risk Pareto residual v4 | -0.037949 | -0.027475 | -0.000499 | +0.001401 | -0.027844 | -0.054709 |

The best macro candidate was cycle 5 in every outer population, but its mean
macro score still worsened by `0.000404`; it improved macro RMSE in only two
of five populations. It improved score-5 continuous RMSE and 3/4 balanced
accuracy consistently, but worsened the low tail by `0.028734`. Cycle 1 made
the opposite trade: both tails improved in all five populations, while macro
RMSE worsened by `0.010025` and 3/4 balanced accuracy by `0.007562`.

Across 100 candidate evaluations, failures by inner gate were: macro threshold
100, equal-group threshold 100, 3/4 balanced-accuracy threshold 100, Spearman
guard 95, axis guard 86, low-tail direction 70, and high-tail direction 15.
These counts are dependent diagnostics, not 100 independent trials.

No candidate changed score-5 integer recall in any of the 300
candidate/population/axis cells. Continuous score-5 gains therefore did not
recover actual band-5 predictions. The bounded corrections were only
`.15--.30`, while band 5 requires crossing `4.5`; this explains why small
continuous gains did not alter recall. Score-1 counts remain only 11/26/4 by
axis, so score 1 is not used for promotion.

## Final metrics

All selected predictions equal exact R0:

- Macro RMSE: `0.5687802169918456`
- Content / organization / expression RMSE:
  `0.509024 / 0.688392 / 0.508925`
- Equal-group RMSE: `0.6915490515990893`
- Low `{1,2}` RMSE: `0.9233348764858816`
- Score-5 RMSE: `0.8841899261902623`
- Gold-3/4 balanced accuracy: `0.6433125889620798`
- Macro Spearman: `0.6002884386652662`
- 3-to-4 / 4-to-3 rates: `0.305836 / 0.356965`
- Paired candidate-minus-baseline 95% interval: `[0, 0]`
- Final gate: failed; final selection: exact R0 fallback.

## Execution and artifacts

GPU0 completed the real five-family smoke before the full run. Outer folds
0--3 then ran concurrently on GPUs 0--3, and fold 4 ran on GPU3 after it
finished first. The run took about 426 seconds from preflight to final
aggregate. Active workers used about 0.85--0.89 GiB each; sampled utilization
was commonly 20--35%. This workload is many small 1,200-row dense fits rather
than a memory-bound decoder batch, so four-way outer parallelism reduced wall
time even though it did not fill H100 memory.

Commands:

```bash
.venv-standard/bin/python scripts/run_iterative_tail_learner.py --launch
.venv-standard/bin/python scripts/run_iterative_tail_learner.py --progress
```

Ignored artifacts:

- Public runtime root:
  `outputs/iterative-tail-learner-v5/iterative-tail-learner-v5-20260802-001`
- Restricted row outputs:
  `data/processed/restricted/iterative_tail_learner_v5/iterative-tail-learner-v5-20260802-001`
- Aggregate SHA-256:
  `eb7906883fbe91d93ab0928848c91ffa8448cd8fc278033caea3c0c06dd99705`
- Completion SHA-256:
  `37b96297e552c80793fa78a9dc2557b5b287e931061f88b545ae1c98cafbc34b`
- Smoke SHA-256:
  `d95d5494e3fc508b401de6964e09ecd0d63aab06aec51a50a24ad134b9f8f20c`
- Task-card SHA-256:
  `4544c04dc082bc954a590f321484d35b93c62bf278a82f675f035059ea64ceee`

All five restricted files contain 400 rows and together cover 2,000 records.
The public JSON tree contains no `source_id`. Twenty-two focused V5 tests,
Python compilation, and scoped diff checks pass. The negative result and exact
R0 fallback are preserved. Further same-data candidate search is not supported
by the observed signal; a materially different, separately preregistered
falsification or new independent information is required.
