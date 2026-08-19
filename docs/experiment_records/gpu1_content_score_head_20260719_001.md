# GPU1 content scalar-head pre-SFT launch: 2026-07-19-001

**Status:** running in the user-owned tmux session
`mal2026-gpu1-content-20260719-001`.

This aggregate-only run trains one `content` scalar head on physical GPU 1.
It uses `selection_train` for training and the isolated `selection_dev` only
for selection/checkpointing. The model has no learned average; a compatible
aggregate-only evaluator may calculate `(content + organization + expression) / 3`
only after all three completion artifacts exist.

- Run ID: `pre-sft-20260719-content-gpu1-001`
- Config and launch ledger: `outputs/reservations/gpu1-content-now-20260719-001/`
- Output/checkpoint root: `outputs/standard-encoder-runs/pre-sft-20260719-content-gpu1-001/`
- GPU scope: physical GPU 1 only (`CUDA_VISIBLE_DEVICES=1`)
- Environment: `.venv-standard`; W&B and Hugging Face offline; no external
  upload
- Git SHA: `86902f1e3a077b1178d1297a1dcccf10e929453d`
- Seed: 2026; manifest SHA-256:
  `12222a8df5b41c9d05a3362e97302600e2ccf8b5928baf360baa6a2a04355f53`
- Hardware: GPU 1, NVIDIA H100 80GB HBM3, driver 580.105.08
- Launch command: `.venv-standard/bin/python
  scripts/train_pre_sft_score_head.py --config
  outputs/reservations/gpu1-content-now-20260719-001/content-primary.json`

GPU-free validation passed for the config and for equivalence with the active
organization/expression configs (only run ID, scalar target, and output path
differ). The pre-launch GPU1-only health check found 0 MiB allocated at 31 C.
The job log confirmed offline tracking and local model-weight loading. Initial
monitoring found the tmux session alive at training step 16/48,040, with GPU1
at 40% utilization, 26,267 MiB allocated, and 38 C. No completion artifact is
present yet.

No raw essays, identifiers, prompts, explanations, predictions, credentials,
or frozen validation rows are recorded here.
