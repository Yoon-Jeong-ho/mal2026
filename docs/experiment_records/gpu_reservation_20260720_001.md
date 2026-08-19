# GPU 0--3 reservation launch record: 2026-07-20-001

**Status:** scheduled, not running. No GPU workload, GPU query, judge request,
validation-judging request, SFT, DPO, or GRPO ran during preparation.

## Reservation

- Start: **2026-07-20 00:00:00 Asia/Seoul**.
- Fixed end: 2026-07-21 00:00:00 Asia/Seoul.
- User-owned tmux session: `mal2026-resv-20260720-0000-kst`.
- Scope: physical GPUs 0, 1, 2, and 3 only. The scheduler has no bare GPU
  discovery path; all health checks are per selected physical ID.
- Ignored runtime/ledger: `outputs/reservations/gpu0-3-20260720-0000-kst-001/`.
  It contains aggregate-only plan/config/checkpoint/status/log references.

## Scheduled research

- GPU 0: judge-v2 pilot only after its midnight immutable preflight repeats the
  GGUF byte/hash gate, llama.cpp revision/tag gate, verified train-only derived
  artifact binding, GPU-0 CVD attestation, and existing judge hard gates.
  The train-only artifact is the completed 6,000-row derived artifact with
  checksum `d69dd9bc349c117a55b4bf652507135f084c9045d95d5939da3980223893aecf`.
  Validation is never loaded or requested by this job.
- GPUs 1--3: three independent Qwen3-Embedding-8B scalar score heads for
  `content`, `organization`, and `expression`, each trained through the
  maintained Hugging Face `Trainer` on selection-train and selected only on
  isolated selection-dev. No learned `average` head exists; the queued
  aggregate-only ensemble computes predicted average externally as
  `(content + organization + expression) / 3`.
- Queued follow-ons: content seed-2027 replication and organization LoRA-r8
  ablation, both conditional on their primary checkpoint artifact. The ensemble
  waits for all three primary completion artifacts.

## Watchdog / fallback controls

The watchdog accepts only the fixed plan, exact 0--3 queue, exact project
interpreter, and approved score/judge command schemas. It stops at the stated
reservation end, temperature/health fault, or bounded fallback budget. When a
GPU has no runnable research job, it can run only a separately labelled,
5-minute, GPU-local PyTorch backfill with `CUDA_VISIBLE_DEVICES` equal to that
one physical ID, one visible GPU, a 30% memory fraction cap, data access
prohibited, heartbeat logging, and termination when a queued research job
becomes runnable. It never substitutes a backfill for a runnable research job.

## Validation evidence before scheduling

- Python syntax checks passed for the reservation scheduler, watchdog,
  backfill, scalar-head trainer, and external-average evaluator.
- `tests.test_standard_encoder_stack` passed with `PYTHONPATH=src`.
- Both judge shell scripts passed `bash -n`.
- The scheduler dry-run generated and statically validated every planned job
  command/config/path without CUDA initialization or GPU queries, then
  revalidated the exact same plan before tmux scheduling.
- A final post-schedule GPU-free revalidation of the five scalar-head configs,
  queued ensemble plan, and seven-job watchdog plan passed after the final
  GPU-local CVD guard was added.
- Independent server-internal safety and experiment-validity reviews found the
  initial joint-four-output encoder and standard matrix unsuitable; the
  scheduled implementation uses scalar heads and does not invoke the matrix.

No result metric is available until the durable pipeline completes.
