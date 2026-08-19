# One-hour-idle-triggered five-day `(D)_vllm` soak — 2026-08-07-007

- **Status:** armed and monitoring; current workload is untouched.
- **Protocol:** begin monitoring GPUs 0--3 immediately for 48 hours. If all
  four remain at 0 MiB and 0% utilization for 60 continuous minutes, launch
  one fresh 120-hour synthetic vLLM soak.
- **Conflict policy:** never terminate, displace, signal, or identify another
  process. Any selected-GPU use resets the idle timer. The launcher repeats
  idle/cool and localhost-port checks immediately before allocation.
- **Process title:** repository-owned long-running Python scheduler, client,
  and top-level server use exact title `(D)_vllm` with `SPT_NOENV=1`.
- **Privacy/runtime:** existing local pinned Qwen3.6 native-FP8 model and vLLM
  0.25.1; synthetic prompts only; no MAL2026 data; no raw prompts or responses
  persisted; aggregate metrics and selected-GPU telemetry only.
- **Initial observation:** GPUs 0--3 currently have 22,659 MiB allocated and
  61--72% utilization, so the scheduler must begin with zero accumulated idle
  time and leave the current workload untouched.
- **Scheduler/run IDs:** `vllm-idle-arm-gpu0-3-20260807-007` and
  `vllm-soak-gpu0-3-120h-20260807-007`.
- **Sessions:** `mal2026-vllm-idle-arm-gpu0-3-20260807-007` and eventual
  `mal2026-vllm-soak-gpu0-3-120h-20260807-007`.

## Launch evidence

- Python/shell/config checks passed. A live preflight verified exact process
  title `(D)_vllm` while preserving the attested environment variable.
- Monitoring began at **2026-08-07 15:52 KST** and expires at
  **2026-08-09 15:52 KST** if no qualifying idle interval occurs.
- First observation retained zero idle seconds because all four GPUs had
  22,659 MiB allocated. No process was altered or identified.
- The durable scheduler itself displays `(D)_vllm`; live state is
  `outputs/legacy/vllm-idle-scheduler/vllm-idle-arm-gpu0-3-20260807-007/state.json`.
