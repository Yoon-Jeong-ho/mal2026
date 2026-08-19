# Manual five-day `(D)_vllm` soak replay — 2026-08-03-006

- **Status:** stopped early by direct user request on 2026-08-04 15:10 KST;
  server, attestation, client, title, and utilization gates had passed.
- **Authorization:** user requested immediate five-day execution on GPUs 0--3,
  bypassing the prior 30-minute idle gate.
- **Runtime/privacy:** unchanged local Qwen3.6 native-FP8, vLLM 0.25.1,
  DP=4/TP=1, synthetic prompts only, no MAL2026 data, and no persisted raw
  prompts or responses.
- **Process title:** repository-owned scheduler/client/top-level server uses
  exact title `(D)_vllm`. `SPT_NOENV=1` preserves resource-attestation
  environment variables while setting that title.
- **Run/session:** `vllm-soak-gpu0-3-120h-20260803-006` /
  `mal2026-vllm-soak-gpu0-3-120h-20260803-006`.

## Recovery and live evidence

- Focused preflight verified that `(D)_vllm` remained visible while
  `MAL2026_TITLE_ATTEST=present` remained readable in `/proc/<pid>/environ`.
  Python compilation, shell syntax, vLLM CLI help, selected-GPU idle/cool, and
  port-availability checks passed.
- The fresh server reached health and its environment/resource attestation
  passed. Both the top-level vLLM server and synthetic client display the exact
  `(D)_vllm` title; vLLM's internal child processes retain their upstream
  `VLLM::*` diagnostic titles.
- The 432,000-second client began at approximately 2026-08-03 16:50 KST, so
  automatic completion is expected near 2026-08-08 16:50 KST.
- First 62.5-second aggregate: 1,519 completed requests, 0 failures, 777,728
  output tokens, and 99% p95 utilization on every GPU. The simultaneous live
  sample was 99% on GPUs 0--3; temperatures were 51, 50, 66, and 51 C.
- Live ignored artifacts:
  `outputs/legacy/vllm-synthetic-soak/vllm-soak-gpu0-3-120h-20260803-006/`.

## User-requested stop

- Cleanup removed the tmux session, top-level `(D)_vllm` server/client, and
  vLLM child processes. GPUs 0--3 each reported 0 MiB and 0% utilization.
- Last aggregate snapshot: 80,441.9 seconds, 2,481,678 completed requests,
  0 failed requests, and 1,270,619,136 output tokens.
- Aggregate-only stop evidence is
  `outputs/legacy/vllm-synthetic-soak/vllm-soak-gpu0-3-120h-20260803-006/user_stop_summary.json`.
