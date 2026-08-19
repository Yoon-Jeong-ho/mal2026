# Delayed GPU0--3 idle-triggered five-day vLLM soak — 2026-08-01-002

- **Status:** triggered a five-day workload, then stopped by direct user
  request on 2026-08-02 10:36 KST and superseded by lineage `...-003`.
- **Protocol amendment:** direct user request supersedes scheduler `...-001`
  before it launched any workload. Wait five hours without checking the idle
  gate, then monitor GPUs 0--3 for 48 hours. If all four remain at 0 MiB and
  0% utilization for 30 continuous minutes during that window, launch one
  120-hour synthetic vLLM soak.
- **Conflict policy:** never terminate, displace, signal, or identify another
  process. Any selected-GPU use after monitoring begins resets the idle timer.
  The vLLM launcher repeats the idle/cool and localhost-port gates immediately
  before allocation.
- **Privacy/runtime:** existing local pinned Qwen3.6 native-FP8 model and vLLM
  0.25.1; synthetic prompts only; no MAL2026 data; no raw prompts or responses
  persisted; aggregate metrics and selected-GPU telemetry only.
- **Scheduler/run IDs:** `vllm-idle-arm-gpu0-3-20260801-002` and
  `vllm-soak-gpu0-3-120h-20260801-002`.
- **Sessions:** `mal2026-vllm-idle-arm-gpu0-3-20260801-002` and eventual
  `mal2026-vllm-soak-gpu0-3-120h-20260801-002`.

## Launch evidence

- Static checks passed and the durable scheduler started at
  2026-08-01 23:01 KST.
- Idle monitoring begins at **2026-08-02 04:01 KST** after the exact five-hour
  delay. The 48-hour monitoring window ends at
  **2026-08-04 04:01 KST** if no qualifying interval occurs.
- Delayed state reports zero observations and zero accumulated idle seconds;
  therefore debugging activity before 04:01 KST cannot trigger vLLM.
- Live state:
  `outputs/legacy/vllm-idle-scheduler/vllm-idle-arm-gpu0-3-20260801-002/state.json`.

## Trigger and stop

- Monitoring began at the scheduled time and observed 30 continuous idle
  minutes. It launched the five-day workload at 2026-08-02 04:31 KST.
- The user later requested an immediate stop and a new two-hour delay. Cleanup
  released GPUs 0--3; each reported 0 MiB and 0% utilization.
- Last workload snapshot: 21,671.9 seconds, 773,952 completed requests,
  0 failed requests, and 396,263,424 output tokens. The ignored stop record is
  `outputs/legacy/vllm-synthetic-soak/vllm-soak-gpu0-3-120h-20260801-002/user_stop_summary.json`.
