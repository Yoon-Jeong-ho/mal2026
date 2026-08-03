# KURE cRT fresh-head integration recovery — 2026-08-03

## Status and authority

- Run ID: `kure-ordinal-crt-recovery-v1-20260803-001`
- Status: pre-registered; GPU execution pending the GPU0 smoke gate.
- Authorized scope: GPUs 0--3, with physical GPU0 used first for the smoke gate.
- Scientific scope: a narrow integration recovery of the failed Stage3
  `coral-natural` cRT head. The failed Stage3 artifacts are preserved and are
  not overwritten.
- Selection boundary: train-only exact five-fold OOF. The exposed validation
  split is forbidden for fitting, tuning, selection, early stopping, or gate
  decisions. The `average` target is forbidden throughout.

## Fixed hypothesis and protocol

Stage3's fresh 5-way cRT head received only 120 optimizer updates at the shared
`5e-5` phase-1 learning rate and collapsed every prediction to rounded score 3.
This run changes only the head-integration procedure:

1. Replay the pinned KURE/AI-Hub `coral-natural` phase 1 independently for each
   outer fold and each of `content`, `organization`, and `expression`: six
   epochs, learning rate `5e-5`, weight decay `0.01`, batch 20, gradient
   accumulation 2, maximum length 1536, seed 2026080302 with deterministic
   derived per-fold/per-axis seeds.
2. Freeze the phase-1 backbone and LoRA representation and cache fit-fold-only
   1024-dimensional CLS-L2 features.
3. Initialize a new `Linear(1024, 5)` head with zero weights and the logarithm
   of the fit-fold empirical rounded-label prior as bias.
4. Train only that head with AdamW, constant learning rate `5e-3`, weight decay
   `0.01`, no warmup, batch 20 with two-microbatch accumulation equivalence,
   20 epochs and exactly 800 optimizer updates. Loss is 5-way cross entropy
   plus `0.25` times raw-score MSE.
5. Before held-fold inference, require a fit-label-only bias optimizer sanity
   gate: max PMF error at most 0.02, ordinal rounded-label mean error at most
   0.02, and cross entropy at most empirical entropy plus 0.005.
   This is a separately disclosed integration diagnostic using AdamW for 160
   steps at learning rate `0.05` and weight decay `0.01`; it is not a cRT
   hyperparameter or a source of model weights.
6. Held rows and held labels cannot enter head fitting or the sanity gate.
   Held features are computed only after the fitted head passes the gate;
   held labels are read only after the restricted prediction artifact is
   durably written.

There is one candidate and no retry, hyperparameter sweep, validation-based
choice, or post-result retuning in this run.

## Frozen inputs

- Git SHA before implementation commit: `70df5e29a355973361bd35494c56edc5fa2fb4e5`
- Stage3 config: `configs/kure_ordinal_oof.v1.json`
  - SHA-256: `5108e37c490482fbbf45c043f2ad4d0154048d432b4a4a163b3d102f8c6d756e`
- Training rows: `eval/train.jsonl`
  - SHA-256: `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Fold manifest SHA-256:
  `9ccb63c857f80cefb1ed15ac4f60ca75f5570d044c273030d5eb0185c756e938`
- Fold-row artifact SHA-256:
  `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16`
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- KURE revision: `d14c8a9423946e268a0c9952fecf3a7aabd73bd9`
- KURE config SHA-256:
  `852d42e020c7f989c2acaf30fc683b7f768e8c6d1ab17166e835442162bd825d`
- AI-Hub warm-start tensor SHA-256:
  `ffdc985d56c655c03e8964927b127b24f0c5bb7fdde8d89e944941f5419cf25a`

The reviewed executable bundle before launch is:

- `configs/kure_crt_recovery.v1.json`:
  `238e5d18ed390a839b2df517b457169676e54806ab1015421c2413e8116479d3`
- `src/mal2026/kure_crt_recovery.py`:
  `03869f3d33e8d5dbb8dfe02e50067e75fca29113019062fbaecc0861ec401e00`
- `scripts/run_kure_crt_recovery.py`:
  `cba31024bb076c53fb9b31b4f5776fa2cfa91e36c2fd9b973b02cf1c73c86451`
- `scripts/run_kure_crt_recovery_gpu0_3.sh`:
  `7ad0ec024e1069d41223ea8c125a2e1b3a0439812a74862af2a0f1139f435cd1`
- `tests/test_kure_crt_recovery.py`:
  `f99bdf0a613b0c976ae6ba41f800082759cab3cecc39e47d6aaf4bc703b4e64e`

The implementation commit SHA is recorded below after the scoped commit and
before GPU execution.

Independent code review found no code or protocol blocker after two high
findings were repaired: the frozen 10,000-resample common gate is now executed,
and exact command/environment/hardware evidence is now materialized. The
reviewer passed 10 focused tests, `py_compile`, `bash -n`, and validate-only.
No LSP/type-checker executable exists in the mandated environment; this is a
tooling limitation rather than a remaining code finding.

## Runner and output boundaries

```bash
bash scripts/run_kure_crt_recovery_gpu0_3.sh smoke
bash scripts/run_kure_crt_recovery_gpu0_3.sh full
```

- Public aggregate-only output:
  `outputs/kure-ordinal-crt-recovery-v1/kure-ordinal-crt-recovery-v1-20260803-001`
- Restricted row predictions and fit artifacts:
  `data/processed/restricted/kure_crt_recovery_v1/kure-ordinal-crt-recovery-v1-20260803-001`
- GPU telemetry interval: 30 seconds.
- Hardware: four NVIDIA H100 80GB GPUs for the full run; GPU0 only for smoke.
- Environment: existing `.venv-standard`; no environment creation or package
  installation is permitted.

## Evaluation and promotion gate

The primary report is exact train-only five-fold OOF three-axis macro RMSE,
with Spearman, low-tail `{1,2}` RMSE, score-5 RMSE, gold-3/4 balanced accuracy,
and per-axis RMSE reported alongside exact R0. The exact 10,000-resample
Stage6 common promotion criteria are reported, but the protected output
remains exact R0. Recovery artifacts are outside the frozen Stage6 trust
chain, so even a gate pass is not automatic deployment eligibility and
requires an explicitly authorized, newly hash-bound decision. This recovery
does not claim that RMSE 0.4 is achieved; it tests whether the known cRT
integration failure can be removed without changing the scientific hypothesis.

## Results and deviations

Pending. Smoke and full-run hashes, exact commands, utilization, metrics,
negative outcomes, and any integration-only deviation will be appended here.
