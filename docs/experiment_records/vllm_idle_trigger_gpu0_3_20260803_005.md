# Named-process delayed GPU0--3 idle-triggered vLLM soak — 2026-08-03-005

- **Status:** manually promoted before the idle gate, then stopped at the
  server-attestation gate; fresh integration-recovery replay is `...-006`.
- **Protocol:** stop lineage `...-004`, wait two hours, then monitor GPUs 0--3
  for the inherited 48-hour window. If all four remain at 0 MiB and 0%
  utilization for 30 continuous minutes, launch a fresh 120-hour synthetic
  vLLM soak.
- **Process-title amendment:** user requested that repository-owned Python
  processes explicitly use `setproctitle` with title `(D)_vllm`. The scheduler,
  synthetic client, and top-level vLLM CLI wrapper set that exact title. The
  existing environment already contains `setproctitle`; no package was
  installed.
- **Conflict policy:** never terminate, displace, signal, or identify another
  process. Any selected-GPU use after monitoring begins resets the idle timer;
  the launcher repeats idle/cool and localhost-port checks before allocation.
- **Privacy/runtime:** local pinned Qwen3.6 native-FP8 model, vLLM 0.25.1,
  synthetic prompts only, no MAL2026 data, and no persisted raw prompts or
  responses.
- **Scheduler/run IDs:** `vllm-idle-arm-gpu0-3-20260803-005` and
  `vllm-soak-gpu0-3-120h-20260803-005`.
- **Sessions:** `mal2026-vllm-idle-arm-gpu0-3-20260803-005` and eventual
  `mal2026-vllm-soak-gpu0-3-120h-20260803-005`.

## Launch evidence

- Static Python/shell checks and the vLLM CLI help gate passed. The existing
  `setproctitle` package was verified without environment mutation, and a live
  process-table check showed the exact title `(D)_vllm`.
- The durable scheduler started at 2026-08-03 13:57 KST with that title.
- Idle monitoring begins at **2026-08-03 15:57 KST** after the exact two-hour
  delay. The inherited 48-hour window ends at
  **2026-08-05 15:57 KST** if no qualifying idle interval occurs.
- Delayed state reports zero observations and zero accumulated idle seconds;
  activity before 15:57 KST cannot trigger vLLM.
- Live state:
  `outputs/legacy/vllm-idle-scheduler/vllm-idle-arm-gpu0-3-20260803-005/state.json`.

## Manual promotion and title integration recovery

- At 2026-08-03 16:41 KST, the user bypassed the idle gate after 793 seconds
  (13 min 13 sec) and requested immediate execution.
- The top-level vLLM process displayed the exact `(D)_vllm` title and reached
  HTTP health, but server attestation failed because `setproctitle` reused the
  environment memory and hid `CUDA_VISIBLE_DEVICES` from `/proc/<pid>/environ`.
  The server then shut down cleanly; no synthetic client request was started
  and GPUs 0--3 returned to 0 MiB.
- The bounded repair sets `SPT_NOENV=1` before importing/calling
  `setproctitle`. This preserves the exact title while keeping environment
  attestation readable. No model, request, duration, GPU, or data variable
  changed. The fresh run ID is `vllm-soak-gpu0-3-120h-20260803-006`.
