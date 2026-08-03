# KURE Gaussian-LDS weighted-CORAL exact-OOF result

- Run ID: `kure-coral-lds-oof-v1-20260803-001`
- Candidate: `coral-lds-gaussian-s025-cap4`
- Status: completed negative result; candidate frozen
- Scientific result Git SHA: `e041a15b0ff0e78c87af6f8a395fdc2090162062`
- Exact commands:
  - `.venv-standard/bin/python scripts/prepare_kure_lds_inputs.py --config configs/kure_lds_oof.v1.json`
  - `bash scripts/run_kure_lds_oof_gpu0_3.sh smoke`
  - `MAL2026_ATTEMPT_TAG=full-001 bash scripts/run_kure_lds_oof_gpu0_3.sh full`
- Seed: `2026080302`
- GPU scope: physical GPUs 0--3; fold mapping `0→0, 1→1, 2→2,
  3→3, 4→0`
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Fit-label manifest SHA-256:
  `c38d21fc9ec54c1d7140a03e74287f22316c1ab8e6e05217d8f89372061a7e58`
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- Failed direct entry-condition aggregate SHA-256:
  `4624146639b9b7c93a13850b8114169b49f6bec3997356219d000f4875fce94d`
- Scientific task-card SHA-256:
  `80e4a966d083f662f07cb70c742d2463cef9a82a6fff0b079ac6aa9ef19000cb`
- Runtime config-file SHA-256:
  `0a1a93ddd0ed881f1fdaa88df3a548e22df42b2dd0385b4e9296cc1db3d5a771`
- Aggregate path:
  `outputs/kure-coral-lds-oof-v1/kure-coral-lds-oof-v1-20260803-001/aggregate.json`
- Aggregate SHA-256:
  `96447f15dc07d68727adddfafc84ffca88148fb28daaa2289dfb69ec195f07bd`

## Frozen result

| Metric | exact R0 | LDS candidate | improvement (positive is better) |
|---|---:|---:|---:|
| Macro RMSE | 0.568780 | 0.675829 | -0.107049 |
| Content RMSE | 0.509024 | 0.589541 | -0.080517 |
| Organization RMSE | 0.688392 | 0.778225 | -0.089833 |
| Expression RMSE | 0.508925 | 0.659722 | -0.150797 |
| Low-tail RMSE | 0.923335 | 1.102386 | -0.179051 |
| Score-5 RMSE | 0.884190 | 1.209207 | -0.325017 |
| Gold-3/4 balanced accuracy | 0.643313 | 0.561241 | -0.082072 |
| Macro Spearman | 0.600288 | 0.367982 | -0.232306 |

The candidate failed every performance gate. The clustered 10,000-resample
95% interval for `R0 RMSE - candidate RMSE` was
`[-0.118112, -0.095718]`, so degradation is decisive. All 6,000 axis
predictions again rounded into bands 3 or 4. Content predicted band 3 for all
2,000 rows; organization produced 1,563 band-3 and 437 band-4; expression
produced 500 band-3 and 1,500 band-4. No band 1, 2, or 5 was predicted.

Relative to the already negative phase-1 direct decoder, LDS improved the
low-tail RMSE from 1.220393 to 1.102386 and Spearman from 0.331714 to
0.367982, but worsened overall RMSE from 0.673147 to 0.675829 and score-5
RMSE from 1.186600 to 1.209207. Reweighting alone therefore did not repair
the representation/decoder collapse.

## Runtime evidence

- GPU0 smoke passed with ten fit-only rows, two optimizer steps, finite nonzero
  LoRA/CORAL gradients, bounded mean-one weights, 292-tensor checkpoint, and
  eight label-free held predictions.
- Full training completed once with five immutable 400-row outer folds and 15
  independent axis checkpoints. No fold was resumed or overwritten.
- Full telemetry had 231 samples per GPU. Peak utilization was 100% on all
  four GPUs. Mean utilization was 84.16%, 42.27%, 42.25%, and 42.23% on
  GPUs 0--3: folds 0--3 ran concurrently, then the preregistered fold-4 mapping
  kept only GPU0 active. Mid-run axis/fold remapping was intentionally rejected
  because it would change the scientific protocol after partial results.
- Every long-running process used `setproctitle` with `mal2026:lds:*`; live
  titles exposed fold and axis.
- `validation_rows_loaded=false`, `average_target_used=false`; no calibration,
  selection, refit, deployment, or validation evaluation occurred.

Per the frozen stop rule, this LDS candidate cannot be retuned or rerun. Exact
R0 remains protected. A RankSim-only CORAL/raw-MSE candidate would be a new
scientific protocol requiring a new independent train-only evidence partition
and explicit authorization; this result does not authorize it.
