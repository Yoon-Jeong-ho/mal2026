# vLLM synthetic GPU0--3 48-hour soak — 2026-07-25-001

- **Status:** stopped early by direct user request on 2026-07-26 20:23 KST.
- **Purpose:** keep the local vLLM inference path saturated for a bounded
  48-hour systems soak. This is not a scientific model comparison, selection,
  training, or evaluation result.
- **Authorization:** the current user explicitly authorized physical GPUs
  0--3 for two days and requested maximum utilization.
- **Data/privacy:** synthetic prompts only. No MAL2026 source, train,
  validation, evaluation, writing, identifier, or generated rationale is
  opened. Raw prompts and completions are not persisted.
- **Runtime:** existing `.venv-standard`; vLLM 0.25.1; local pinned
  `Qwen/Qwen3.6-35B-A3B-FP8` revision
  `95a723d08a9490559dae23d0cff1d9466213d989`; DP=4/TP=1 on physical GPUs
  0,1,2,3; 192 sequences and 65,536 batched tokens per DP rank; 768 client
  requests; 512 forced output tokens; 0.90 GPU memory target.
- **Duration/safety:** 172,800 seconds after server readiness. Selected-GPU
  utilization, memory, and temperature are sampled every 10 seconds; the
  workload fails closed above 85 C or if safety telemetry fails.
- **Artifacts:** ignored legacy root
  `outputs/legacy/vllm-synthetic-soak/vllm-soak-gpu0-3-48h-20260725-001/`.
  It contains server logs, aggregate-only client metrics, selected-GPU
  telemetry, attestation, and an append-only run ledger.
- **Command:** durable tmux session
  `mal2026-vllm-soak-gpu0-3-48h-20260725-001` runs
  `scripts/run_vllm_synthetic_soak_gpu0_3_48h.sh
  vllm-soak-gpu0-3-48h-20260725-001`.
- **Reproducibility:** record the launch Git SHA and config SHA-256 in the
  append-only runtime ledger. The worktree was already dirty and is preserved.
  The repository does not contain `scripts/research-pipeline-runner.sh`, so
  this bounded run uses the project-approved durable `mal2026-` tmux fallback.

## Integration recovery

- Attempt `...-001` failed before server readiness and before any request
  because the launcher did not expose the already-installed
  `.venv-standard/bin/ninja` to FlashInfer JIT subprocesses. The failed log and
  ledger are preserved under its ignored legacy run directory.
- The bounded repair exports the existing environment's bin directory on
  `PATH`; no package was installed and no runtime, request, model, duration,
  data, or GPU variable changed. The replay uses fresh run ID `...-002` and
  tmux session `mal2026-vllm-soak-gpu0-3-48h-20260725-002`.

## One-hour live gate

- `...-002` reached health and began the timed workload at approximately
  2026-07-25 10:30 KST. The automatic 172,800-second deadline is approximately
  2026-07-27 10:30 KST.
- At 2026-07-25 11:32 KST it remained health-ready after 3,663 seconds:
  134,898 completed requests, 0 failed requests, 69,067,776 output tokens, and
  18,853 output tokens/s. No error category was recorded.
- Ten-second telemetry had 362 samples per GPU. Utilization medians were
  95%, 95%, 91%, and 90% on GPUs 0--3; p95 was 99% on every GPU. The live
  verification sample was 99% on all four GPUs. Mean utilization, which
  includes startup/routing troughs, was 88.6%, 90.0%, 79.1%, and 76.2%.
  Maximum temperatures were 54, 53, 55, and 54 C.
- The user conditionally authorized `Qwen/Qwen3-32B` if this workload did not
  fill utilization. The one-hour gate did not switch models: all four live
  samples were at the device-reported 99% ceiling, medians were at least 90%,
  and the current model had zero request failures. `Qwen3-32B` is not present
  in the local project or Hugging Face cache, so avoiding the switch also
  avoided an unnecessary external download and preserved the uninterrupted
  48-hour lineage.

## User-requested stop

- The user requested termination before the planned 48-hour deadline. The
  agent-created tmux session was terminated at 2026-07-26 20:23 KST and its
  cleanup trap released the vLLM server and client.
- Last aggregate snapshot: 121,961 seconds (33 h 52 min), 4,376,335 completed
  requests, 0 failed requests, and 2,240,683,520 output tokens. Per-GPU p95
  utilization was 99%; maximum temperatures were 54, 53, 66, and 54 C.
- Final release verification: the tmux session is absent and GPUs 0--3 each
  report 0 MiB used and 0% utilization. The aggregate-only stop record is
  `outputs/legacy/vllm-synthetic-soak/vllm-soak-gpu0-3-48h-20260725-002/user_stop_summary.json`.
