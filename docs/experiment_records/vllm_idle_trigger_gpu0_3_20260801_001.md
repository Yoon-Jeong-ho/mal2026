# GPU0--3 idle-triggered five-day vLLM soak — 2026-08-01-001

- **Status:** superseded before workload launch by direct user request at
  2026-08-01 23:00 KST; current workloads remained untouched.
- **Interpreted authorization:** for the next 48 hours, observe only GPUs 0--3.
  If all four are simultaneously empty and at 0% utilization for 30 continuous
  minutes, launch one 120-hour synthetic vLLM soak. If no qualifying interval
  occurs within 48 hours, expire without launching.
- **Conflict policy:** never stop, displace, signal, or identify an existing
  process. Any selected-GPU memory use or utilization resets the idle timer.
  The vLLM launcher performs a second idle/cool check and a localhost-port
  check immediately before allocating resources.
- **Current observation:** GPUs 0--3 each have about 79.7 GiB allocated, so the
  trigger begins in busy/waiting state and does not alter the current workload.
- **Five-day workload:** existing local pinned Qwen3.6 native-FP8 model and
  vLLM 0.25.1; DP=4/TP=1; synthetic prompts only; raw prompts/responses are not
  persisted; aggregate metrics and selected-GPU telemetry go to ignored
  `outputs/legacy/` paths.
- **Scheduler/run IDs:** `vllm-idle-arm-gpu0-3-20260801-001` and
  `vllm-soak-gpu0-3-120h-20260801-001`.
- **Sessions:** scheduler
  `mal2026-vllm-idle-arm-gpu0-3-20260801-001`; eventual workload
  `mal2026-vllm-soak-gpu0-3-120h-20260801-001`.

## Launch evidence

- Static preflight passed with five-day config SHA-256
  `34983e2751da535948230c8c384f8499c08335c15690167d9034e5c75d8d17a4`.
- Scheduler armed at 2026-08-01 01:52 KST and expires at
  2026-08-03 01:52 KST if no qualifying idle interval occurs.
- First observation was busy: GPU memory use was 79,721, 79,721, 79,847,
  and 79,721 MiB. Consecutive idle time therefore remains zero; no process was
  signaled and the five-day vLLM session was not started.
- Live aggregate-only scheduler state is
  `outputs/legacy/vllm-idle-scheduler/vllm-idle-arm-gpu0-3-20260801-001/state.json`.

## Supersession

- The scheduler was stopped before its 30-minute idle gate completed. No vLLM
  workload session or five-day run artifact was created.
- Replacement lineage `...-002` adds the user-requested five-hour delay before
  beginning a fresh 48-hour monitoring window.
