# KURE Stage3 phase-1 direct CORAL exact-OOF result

- Run ID: `kure-phase1-direct-oof-v1-20260803-001`
- Status: completed negative result; candidate frozen
- Scientific result Git SHA: `95dfbc3b721c4e1edbe1a547405ed4476c90be34`
- Exact command sequence:
  - `bash scripts/run_kure_phase1_direct_oof_gpu0_3.sh smoke`
  - `bash scripts/run_kure_phase1_direct_oof_gpu0_3.sh full`
- Seed: `2026080302`
- GPU scope: physical GPUs 0--3; GPU0 smoke, then fold mapping
  `0→0, 1→1, 2→2, 3→3, 4→0`
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- Authorized task-card SHA-256:
  `16f9e8aa4a7d4921bf8d38530d2216838fb1be699e4be9ef47c7f538826c425c`
- Runtime config-file SHA-256:
  `10337e1de67c96e8385475e484653736b89fdde41083e94fdc972fffadb41c60`
- Aggregate path:
  `outputs/kure-phase1-direct-oof-v1/kure-phase1-direct-oof-v1-20260803-001/aggregate.json`
- Aggregate SHA-256:
  `4624146639b9b7c93a13850b8114169b49f6bec3997356219d000f4875fce94d`
- Restricted predictions: five immutable 400-row fold artifacts under
  `data/processed/restricted/kure_phase1_direct_oof_v1/kure-phase1-direct-oof-v1-20260803-001/`.

## Result

The saved phase-1 CORAL expected-score decoder failed every frozen promotion
condition except coverage, finiteness, and tail-support checks.

| Metric | exact R0 | direct phase-1 | improvement (positive is better) |
|---|---:|---:|---:|
| Macro RMSE | 0.568780 | 0.673147 | -0.104367 |
| Content RMSE | 0.509024 | 0.582712 | -0.073688 |
| Organization RMSE | 0.688392 | 0.777333 | -0.088942 |
| Expression RMSE | 0.508925 | 0.659395 | -0.150470 |
| Low-tail RMSE | 0.923335 | 1.220393 | -0.297059 |
| Score-5 RMSE | 0.884190 | 1.186600 | -0.302410 |
| Gold-3/4 balanced accuracy | 0.643313 | 0.543297 | -0.100015 |
| Macro Spearman | 0.600288 | 0.331714 | -0.268574 |

All 6,000 axis predictions rounded into bands 3 or 4. Content predicted band
3 for all 2,000 rows; organization predicted 1,587 band-3 and 413 band-4;
expression predicted 238 band-3 and 1,762 band-4. No band 1, 2, or 5 was
predicted. The clustered 10,000-resample 95% interval for `R0 RMSE - candidate
RMSE` was `[-0.115584, -0.092772]`, confirming degradation rather than an
uncertain tie. Exact R0 remains protected.

## Runtime evidence and deviations

- Smoke: eight label-free held rows, content only, no held-gold metric.
- Full telemetry: 29 samples per GPU; peak utilization was 100%, 85%, 59%,
  and 100% on GPUs 0--3. The low time-averaged utilization is expected because
  model loading and the CPU-only 10,000-resample aggregate dominated elapsed
  time after brief GPU inference.
- `setproctitle` exposed `mal2026:direct:*` titles with stage/fold/axis. The
  public smoke report records `mal2026:direct:smoke:f0:persist`.
- `validation_rows_loaded=false`, `average_target_used=false`, and no training,
  calibration, selection, refit, or deployment occurred.
- The first smoke launcher attempt stopped before output creation because the
  concurrently updated delayed scheduler state had bounded negative mtime age.
  Integration-only commit `95dfbc3` accepts at most 300 seconds of negative
  clock skew while retaining the stale-state and named-run gates. The retry
  passed. This did not change the scientific protocol or data.

Per the preregistered stop rule, this direct candidate is permanently frozen.
Its failed gate is the sole entry condition for the separately registered
Gaussian-LDS weighted-CORAL experiment.
