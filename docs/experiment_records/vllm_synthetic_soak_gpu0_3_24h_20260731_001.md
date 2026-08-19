# vLLM synthetic GPU0--3 24-hour soak — 2026-07-31-001

- **Status:** paused by direct user request on 2026-07-31 20:17 KST.
- **Authorization:** the current user requested another 24-hour run, inheriting
  the prior synthetic vLLM soak's physical GPU 0--3 scope and privacy boundary.
- **Purpose/data:** aggregate-only systems soak using synthetic prompts. No
  MAL2026 source, train, validation, evaluation, writing, identifier, or
  generated rationale is opened or persisted.
- **Fixed runtime:** existing `.venv-standard`, vLLM 0.25.1, pinned local
  `Qwen/Qwen3.6-35B-A3B-FP8` revision
  `95a723d08a9490559dae23d0cff1d9466213d989`, DP=4/TP=1 on GPUs 0--3,
  768 client requests, and an 86,400-second workload deadline after health.
- **Safety:** ten-second telemetry; fail closed over 85 C or if telemetry
  fails. No external download, environment creation, or package installation.
- **Run ID/session:** `vllm-soak-gpu0-3-24h-20260731-001` /
  `mal2026-vllm-soak-gpu0-3-24h-20260731-001`.
- **Artifacts:** ignored legacy root
  `outputs/legacy/vllm-synthetic-soak/vllm-soak-gpu0-3-24h-20260731-001/`.

## Launch evidence

- Static preflight passed with config SHA-256
  `43c7543e59f817adf15810baa924d13c012bb5e7a2fa4f17735ddec0c535034e`;
  GPUs 0--3 were each at 0 MiB / 0% and at 31--44 C.
- The server reached health and the 86,400-second client deadline began at
  approximately 2026-07-31 04:08 KST, giving an expected automatic stop near
  2026-08-01 04:08 KST.
- First aggregate snapshot after 62.5 seconds: 1,643 completed requests,
  0 failures, 841,216 output tokens, and 99% p95 utilization on every selected
  GPU. Raw prompts and responses were not persisted.
- The first runtime-ledger launch line inherited stale prose saying 48 hours,
  but its recorded config hash resolves to this exact 86,400-second config.
  A correction was appended immediately; the launcher now states that duration
  is bound by config rather than embedding a fixed hour count.

## User-requested pause

- The agent-created tmux session and vLLM/client processes were stopped
  cleanly. GPUs 0--3 each report 0 MiB used and 0% utilization.
- Last aggregate snapshot: 58,109.8 seconds elapsed, 2,074,282 completed
  requests, 0 failed requests, and 1,062,032,384 output tokens.
- Remaining authorized workload time is 28,290.2 seconds (about 7 h 51 min).
  The ignored aggregate-only pause record is
  `outputs/legacy/vllm-synthetic-soak/vllm-soak-gpu0-3-24h-20260731-001/pause_summary.json`.
