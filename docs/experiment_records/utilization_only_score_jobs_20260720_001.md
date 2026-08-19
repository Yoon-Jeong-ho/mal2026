# Utilization-only score jobs: 20260720-001

**Status:** running; this is a resource-utilization workload, not a scientific
result, selection, or model-promotion run.  No metric from these jobs may be
reported or used as evidence.

## Scope and start state

- Runtime/aggregate-only ledger:
  `outputs/reservations/utilization-only-20260720-001/` (ignored).
- Durable session: `utilization-only-20260720-001`.
- The pre-launch selected-device checks at 2026-07-19 23:59:50 KST found GPUs
  0--3 respectively at 0 MiB / 0% and 31 C, 30 C, 33 C, and 30 C.  GPUs 4--7
  were not queried or used.
- Exact pinning is `CUDA_VISIBLE_DEVICES=N` and
  `MAL2026_RESERVED_PHYSICAL_GPU=N` for physical GPU `N`, for N in 0, 1, 2, 3.

## Workloads and boundaries

- GPU 0: existing bounded data-free GPU-local backfill, labelled
  `utilization_only`; no score-training workload is approved for GPU 0.
- GPU 1: prior split-isolated `content` scalar-head workload in explicit
  train-only utilization mode.
- GPU 2: prior split-isolated `organization` scalar-head workload in explicit
  train-only utilization mode.
- GPU 3: prior split-isolated `expression` scalar-head workload in explicit
  train-only utilization mode.
- Each score job has a hard maximum of 10,000 epochs, no validation split
  loading/evaluation/checkpoint selection, sparse 250,000-step checkpointing
  with one retained checkpoint, 5,000-step logging, disabled W&B, and a unique
  ignored output directory containing `utilization_only` in its name.
- No SFT, DPO, GRPO, ensemble evaluation, or results aggregation is queued or
  consumes these outputs.

## Yield and safety policy

- The supervisor checks selected-device temperature every 15 seconds and
  stops that GPU above 80 C or on a health-query failure.
- GPU 0 remains judge-v3-exclusive.  Its backfill process group receives
  SIGTERM when the separate aggregate-only
  `judge_v3_validated.ready.json` marker appears; the supervisor then starts
  only the already-validated v3 wrapper pinned to GPU 0.
- The ready marker was intentionally not created by this utilization run, so
  it cannot launch or alter judge-v3 preparation.  Concurrent frozen-results
  aggregation remains independent and unmodified.
