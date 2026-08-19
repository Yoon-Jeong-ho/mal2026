# Two-hour-delayed GPU0--3 idle-triggered five-day vLLM soak — 2026-08-03-004

- **Status:** triggered a five-day workload, then stopped by direct user
  request on 2026-08-03 13:55 KST and superseded by lineage `...-005`.
- **Protocol amendment:** direct user request stops the active five-day vLLM
  workload from lineage `...-003`, waits two hours, then starts a fresh
  48-hour monitoring window. If GPUs 0--3 are all at 0 MiB and 0% utilization
  for 30 continuous minutes, launch one fresh 120-hour synthetic vLLM soak.
- **Conflict policy:** never terminate, displace, signal, or identify another
  process. Any selected-GPU use after monitoring begins resets the idle timer.
  The launcher repeats idle/cool and localhost-port checks immediately before
  allocation.
- **Privacy/runtime:** existing local pinned Qwen3.6 native-FP8 model and vLLM
  0.25.1; synthetic prompts only; no MAL2026 data; no raw prompts or responses
  persisted; aggregate metrics and selected-GPU telemetry only.
- **Scheduler/run IDs:** `vllm-idle-arm-gpu0-3-20260803-004` and
  `vllm-soak-gpu0-3-120h-20260803-004`.
- **Sessions:** `mal2026-vllm-idle-arm-gpu0-3-20260803-004` and eventual
  `mal2026-vllm-soak-gpu0-3-120h-20260803-004`.

## Launch evidence

- Static checks passed and the durable scheduler started at
  2026-08-03 02:11 KST.
- Idle monitoring begins at **2026-08-03 04:11 KST** after the exact two-hour
  delay. The 48-hour monitoring window ends at
  **2026-08-05 04:11 KST** if no qualifying idle interval occurs.
- Delayed state reports zero observations and zero accumulated idle seconds;
  activity before 04:11 KST cannot trigger vLLM.
- Live state:
  `outputs/legacy/vllm-idle-scheduler/vllm-idle-arm-gpu0-3-20260803-004/state.json`.

## Trigger and stop

- The scheduler later observed 30 continuous idle minutes and launched the
  five-day workload at 2026-08-03 11:50 KST.
- Cleanup released GPUs 0--3; each reported 0 MiB and 0% utilization.
- Last workload snapshot: 7,265.8 seconds, 291,661 completed requests,
  0 failed requests, and 149,330,432 output tokens. The ignored stop record is
  `outputs/legacy/vllm-synthetic-soak/vllm-soak-gpu0-3-120h-20260803-004/user_stop_summary.json`.
