# Iterative tail remediation v2: strict nested outer result

Run ID: `iterative-tail-remediation-v2-20260801-001`

Status: completed; no inner candidate passed all seven gates in any outer
fold; exact R0 OOF baseline retained

Date: 2026-08-01 (Asia/Seoul)

## Decision

Five leakage-guarded outer folds completed. Every fold selected
`base-identity`, so the concatenated selected prediction is exactly the fixed
R0 OOF baseline: macro continuous RMSE `0.568780`, macro Spearman `0.600288`,
equal-group RMSE `0.691549`, low-tail `{1,2}` RMSE `0.923335`, score-5 RMSE
`0.884190`, and true-gold 3/4 balanced accuracy `0.643313`. The final macro
gain is `0.000000`; the candidate-minus-baseline bootstrap interval is the
tautological `[0,0]` because the fail-closed selection is the baseline itself.
It is not equivalence evidence.

The final requirements (macro gain at least `0.01` and paired 95% bootstrap
upper bound below zero) both failed. There is no validation, generalization,
deployment, or leaderboard improvement claim.

## Pre-execution review and repaired blockers

Three independent aggregate/code/protocol reviews found and repaired five
execution blockers before any v2 outer result existed:

1. Inner-selection R16 teachers were changed from a shared 3-of-4 teacher to
   a split-specific 2-of-3 cross-fit teacher, excluding both the inner
   validation fold and outer fold. The outer-refit 3-of-4 teacher is generated
   only after selection freezes.
2. Candidate selection now evaluates every candidate against the same
   identity baseline and selects the global eligible macro-RMSE minimum;
   sequential incumbent tournaments are forbidden.
3. An unregistered standalone convex-blend candidate was removed.
4. The fixed five-knot piecewise model now solves the exact bounded monotone
   weighted least-squares problem by enumerating 64 active faces. Tail cutoff,
   center, kernel, and radius constants were registered exactly.
5. Row-derived isotonic knots are removed from public output. Public records
   contain only counts and deterministic parameter digests; full row
   predictions remain below the restricted ignored root.

Synthetic regression tests cover the teacher isolation, baseline-relative
selection, exact piecewise fit, candidate inventory, and public scrubbing.

## Immutable inputs and execution

- Git SHA at launch: `40a28c758020b356e1d86e9790e55ea08a2ea69c`
- Config SHA-256:
  `62256203bc43315e7cf18abf10377db4d4af802e185418634c5ab3b2c1c2bd93`
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- Records: 2,000 train rows; five fixed folds of 400.
- Targets: exactly content, organization, expression; average forbidden.
- Validation/API: not loaded/not called.
- Environment: existing `.venv-standard`.
- GPU scope: GPUs 0--3 under the user's explicit authorization. GPU0 passed
  a 128-train/32-predict real integration smoke before the full run.

Commands:

```bash
.venv-standard/bin/python scripts/run_iterative_tail_remediation.py --smoke
tmux new-session -d -s mal2026-tail-v2 \
  '.venv-standard/bin/python scripts/run_iterative_tail_remediation.py --full'
```

Outer folds 0--3 ran concurrently on GPUs 0--3. Outer fold 4 started on GPU0
after the first wave completed. The GPU stage took about 35 seconds after
preflight; aggregate/bootstrap completed about seven seconds later. No
pre-existing process was terminated or displaced.

## Inner diagnostic result

The table averages improvement deltas across the five outer-specific inner
OOF populations. Positive is better.

| Family | Macro RMSE | Equal-group | Low `{1,2}` | Score 5 | 3/4 BA | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nested rebuilt R17 | -0.002195 | -0.006376 | -0.064913 | +0.042552 | +0.011324 | -0.003996 |
| conditional R17 delta | +0.000095 | -0.000406 | -0.032596 | +0.035621 | +0.005449 | -0.001817 |
| weighted isotonic/piecewise | -0.000261 | -0.003316 | -0.011943 | -0.000923 | +0.005952 | -0.005205 |
| tail boundary | -0.002477 | -0.005465 | -0.057857 | +0.043188 | +0.011324 | -0.004277 |
| direct evidence ridge | +0.001153 | +0.000105 | -0.002659 | +0.003298 | +0.003710 | -0.001011 |

All evaluated family representatives missed both required thresholds (macro
`+0.005`, equal-group `+0.010`) in every outer-specific selection. R17 and
tail-boundary consistently improved score 5 and 3/4 separation but paid for
it with a much larger low-tail loss. Direct evidence ridge was safest but its
gain was too small and its two tail directions were inconsistent. This is the
reason for the baseline fallback, not an execution failure.

Score 1 remains descriptive only: train counts are 11 content, 26
organization, and 4 expression rows. The combined `{1,2}` metric remains the
promotion target.

## Verification and artifacts

Focused verification: 20 unittests passed; `py_compile`, `git diff --check`,
five outer binding checks, and a public `x_knots`/`y_knots`/`source_id` privacy
scan passed.

Ignored runtime evidence:

- Public aggregate/runtime root:
  `outputs/iterative-tail-remediation-v2/iterative-tail-remediation-v2-20260801-001`
- Restricted predictions:
  `data/processed/restricted/iterative_tail_remediation_v2/iterative-tail-remediation-v2-20260801-001`
- Aggregate SHA-256:
  `bc64443aafc48c45f9a882cf861c11eba9b22d6d417b62d90c43a1fbf81af81f`
- Completion SHA-256:
  `101474406ef9c8400859c52cc6ffcc4c4d2d2dc237626cd2f04e17bf8fa1e3ea`
- Smoke SHA-256:
  `953d80436b4cb7bc40d98403b26c6afe584dbb1e045ec861f062c2560ee73ddb`

The negative result and fallback are preserved. The next 20-cycle program is
adaptive train-only discovery motivated by this trade-off and cannot convert
these same rows into a new independent confirmatory set.
