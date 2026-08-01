# Tail score remediation v3: twenty-cycle adaptive discovery

Run ID: `iterative-tail-cycle-v3-20260801-001`

Status: completed; 20/20 cycles and 100/100 fold-cycle predictions produced;
no strict candidate; exact R0 OOF baseline retained

Date: 2026-08-01 (Asia/Seoul)

## Scope and decision

V3 implemented the requested five-stage loop as 20 fixed train-only cycles:
hypothesis/preregistration, implementation/smoke, heldout cross-fit, frozen
seven-gate review, and result freeze. Three agent roles supplied aggregate
preregistration evidence only; their text, consensus, and judgments were not
model features, pseudo-targets, rewards, or weights.

No cycle passed every frozen gate. The protocol-valid selection remains the
exact R0 OOF baseline with macro RMSE `0.568780`. The best exploratory cycle
was cycle 4, `soft_routed_residual-v4`, with RMSE `0.568611` (gain
`0.000169`). Its post-selection 10,000-resample paired interval for improvement
was `[-0.000819, 0.001140]`, crossing zero. It is not promoted.

V3 was designed after the v2 outer summaries were observed. It is explicitly
adaptive train-only descriptive discovery, not new confirmation. It cannot
support validation, generalization, deployment, or leaderboard claims.

## Immutable execution contract

- Git SHA at launch: `2de57d0d2b06d14a6fe3d4a1e2a96be06fac8fc7`
- Config SHA-256:
  `4acb0fc70ea310a4a6e0a2ba38894c4ee62e42a13d455f2500e741c1fb2b05f9`
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- Historical v2 aggregate SHA-256:
  `bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f`
  (preregistration evidence only; forbidden as a model feature).
- Records: 2,000 train rows in five fixed folds of 400.
- Targets: content, organization, expression only; average forbidden.
- Feature evidence: local score-blind `evidence_hash`; validation and API
  disabled.
- Same seed/fresh initialization for every cycle and fold; checkpoint reuse
  forbidden.

For heldout fold D, only the other four folds were used. A fresh R16 3-of-4
cross-fit teacher generated an R17 challenger; the direct evidence ridge used
fixed alpha `100`. Historical v1/v2 predictions were not used. Each of the 20
cycles fit freshly on the other four folds and predicted D once. Metrics and
selection were not computed until all five folds had emitted all 100
fold-cycle predictions.

## Execution

Environment: existing `.venv-standard`. GPU0 passed a real neural integration
smoke plus one smoke from each of the five cycle families. Full execution used
only GPUs 0--3 after a read-only conflict check reported no existing compute
processes. Folds 0--3 ran concurrently, and fold 4 then ran on GPU0. No process
was terminated or displaced.

Commands:

```bash
.venv-standard/bin/python scripts/run_iterative_tail_cycle.py --smoke
tmux new-session -d -s mal2026-tail-cycle-v3 \
  '.venv-standard/bin/python scripts/run_iterative_tail_cycle.py --full'
```

The first four folds completed in about 40 seconds, fold 4 in another 20
seconds, and aggregation/bootstrap in about 10 seconds. GPU computation was
concentrated in R16 feature regeneration; the fixed NumPy linear/router cycle
fits were CPU-bound. This explains low instantaneous GPU utilization during
the latter portion without indicating a stalled run.

## Twenty-cycle results

Positive deltas mean improvement over R0. `Eq`, `Low`, `High`, and `BA` are
equal-group RMSE, `{1,2}` RMSE, score-5 RMSE, and true-gold 3/4 balanced
accuracy improvement.

| Cycle | Fixed variant | RMSE | Delta | Eq | Low | High | BA |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | soft route v1 | 0.572433 | -0.003652 | -0.002820 | -0.002443 | +0.002165 | -0.004976 |
| 2 | soft route v2 | 0.574866 | -0.006086 | -0.004506 | -0.001212 | +0.001732 | -0.005814 |
| 3 | soft route v3 | 0.571119 | -0.002339 | -0.000698 | +0.002133 | +0.004982 | -0.002608 |
| 4 | soft route v4 | **0.568611** | **+0.000169** | +0.001104 | +0.002562 | +0.004125 | +0.000091 |
| 5--8 | Pareto stacks; identity fallback | 0.568780 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| 9 | Group-DRO ridge v1 | 0.574420 | -0.005640 | -0.003197 | +0.001948 | +0.003190 | -0.002341 |
| 10 | Group-DRO ridge v2 | 0.581426 | -0.012646 | -0.007748 | +0.005075 | +0.002734 | -0.007094 |
| 11 | Group-DRO ridge v3 | 0.588306 | -0.019525 | -0.010734 | +0.012253 | +0.006670 | -0.013938 |
| 12 | Group-DRO ridge v4 | 0.587327 | -0.018547 | -0.006307 | +0.024954 | +0.016583 | -0.018568 |
| 13 | selective hurdle v1 | 0.575113 | -0.006333 | +0.003314 | +0.011919 | +0.028984 | +0.004186 |
| 14 | selective hurdle v2 | 0.575213 | -0.006433 | +0.005580 | +0.018158 | +0.034037 | -0.005957 |
| 15 | selective hurdle v3 | 0.571442 | -0.002661 | +0.002506 | +0.019116 | +0.010190 | -0.007837 |
| 16 | selective hurdle v4 | 0.568694 | +0.000086 | +0.000515 | +0.002599 | +0.000000 | -0.000313 |
| 17 | final ordinal stack v1 | 0.571827 | -0.003047 | +0.000189 | -0.012866 | +0.032293 | +0.001101 |
| 18 | final ordinal stack v2 | 0.577917 | -0.009137 | -0.002827 | -0.012035 | +0.038406 | +0.000455 |
| 19 | final ordinal stack v3 | 0.581638 | -0.012858 | -0.005551 | -0.016572 | +0.044750 | -0.004223 |
| 20 | final ordinal stack v4 | 0.581784 | -0.013003 | -0.005865 | -0.018171 | +0.048160 | -0.004077 |

The Pareto cycles rejected their non-identity train fits because at least one
axis-by-band MSE degraded, so they correctly emitted identity predictions.
Group-DRO and selective hurdle models show that large low/score-5 gains are
possible, but their middle-distribution error erased the macro benefit. The
ordinal stack repeated the historical high-versus-low trade-off.

## Best exploratory cycle: score bands and axes

Cycle 4 improved both continuous tails while largely preserving the middle,
but by far less than the frozen thresholds:

- macro RMSE gain `+0.000169` versus required `+0.005`;
- equal-group gain `+0.001104` versus required `+0.010`;
- low `{1,2}` gain `+0.002562` and score-5 gain `+0.004125`;
- 3/4 BA gain only `+0.000091` versus required `+0.010`;
- Spearman change `-0.000149` (within the guardrail).

Band-level descriptive changes for cycle 4:

| Gold band | Baseline RMSE | Cycle 4 RMSE | Baseline recall | Cycle 4 recall |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.164567 | 1.162590 | 0.030303 | 0.030303 |
| 2 | 0.896752 | 0.894313 | 0.241179 | 0.254460 |
| 3 | 0.504166 | 0.505196 | 0.647305 | 0.648068 |
| 4 | 0.454505 | 0.455747 | 0.639320 | 0.638740 |
| 5 | 0.884190 | 0.880065 | 0.000000 | 0.000000 |

Score 1 remains descriptive only because its axis counts are 11, 26, and 4.
Cycle 4's low-tail gain is mainly a small score-2 improvement. Score-5
continuous error improved slightly, but integer recall stayed zero. Content
RMSE improved `0.000611`, organization improved `0.000022`, and expression
worsened `0.000127`, all within the axis guardrail.

## Verification and artifacts

Before execution, 13 v3 focused tests passed. They cover the exact 20-cycle
inventory, determinism, every family, bounds, average rejection, Pareto
fallback, exact lineage, historical isolation, seven-gate conjunction, and
direct-alpha binding. `py_compile` and `git diff --check` passed. Afterward,
all five fold invariants, all 100 predictions, and a public row-identifier
privacy scan passed.

Ignored artifacts:

- Runtime/aggregate root:
  `outputs/iterative-tail-cycle-v3/iterative-tail-cycle-v3-20260801-001`
- Restricted predictions:
  `data/processed/restricted/iterative_tail_cycle_v3/iterative-tail-cycle-v3-20260801-001`
- Aggregate SHA-256:
  `bc44123673b512ae20212a94f7a3e98b4d227823cd1d74d2fddf8f4a07e3667f`
- Completion SHA-256:
  `238c6074de6abf35aed02c1d95e76b8b1789edf88f78377d56eb101b506cd9d3`
- Smoke SHA-256:
  `199d4112b36b009a7521e3e15a2abd377c6cc189c7d0ee0e3ac050d382fc4f62`

The strict negative result is preserved. Cycle 4 is exploratory evidence for
future routing work, not a promoted model.
