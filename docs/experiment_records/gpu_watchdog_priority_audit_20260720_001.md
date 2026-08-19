# GPU 0--3 Watchdog Priority Audit — 2026-07-20

## Scope and boundary

This audit inspected only GPU 0--3 process metadata, aggregate reservation/heartbeat state, and watchdog/backfill scheduler code. It did not inspect raw writing data, identifiers, or response content; it did not query GPUs 4--7. No active score-training or results-aggregation process was terminated, restarted, or modified.

## Observed evidence

- The utilization-only supervisor owned active score-training processes on GPUs 1--3 and one bounded backfill on GPU 0.
- The reservation watchdog independently launched bounded backfills on GPUs 0--2 while those utilization workloads were active; after its frozen aggregate evaluation ended, it also launched a bounded backfill on GPU 3.
- Thus the existing watchdogs overlapped: GPU 0 had two data-free backfills, GPUs 1--2 had a score-training process plus a reservation backfill, and GPU 3 had score training plus a reservation backfill. GPU utilization remained high, but the overlap competes for compute and invalidates the "one owner" safety invariant.
- Both existing watchdogs had already loaded their pre-audit code. Altering source or plan files cannot safely change those resident processes. No process was signaled; their short bounded backfills are left to their documented duration.

## Cause

Each watchdog recorded only subprocesses it created. Neither had a shared launch lease or a process-metadata guard for a GPU owned by the other watchdog. The utilization supervisor also treated every utilization-only child as yieldable even when it was score training.

## Future coordination control

`scripts/gpu_watchdog_coordination.py` supplies an exclusive aggregate-only per-GPU lease and a priority-request marker. Both watchdogs now use it on every future launch.

- A launch first requires no existing compute process on the exact physical GPU, then an exclusive lease.
- A ready higher-priority research/judge job can request yield only from a data-free `backfill` lease. It never signals score training, aggregation, or research.
- Score training is explicitly non-yieldable. A higher-priority job remains queued until that owner exits; this is the only safe interpretation where no documented score-training yield mechanism exists.
- The reservation watchdog retains bounded backfill cycles for true idle time, so the autonomous queue can fill idle GPUs without competing with an existing measurement workload.

The aggregate-only activation state is `outputs/reservations/gpu0-3-watchdog-coordination-v1/future_schedule_state.json`. Both prior plan files remain unchanged and are referenced there.

## Validation

- Python compilation passed for the coordination module and both watchdogs.
- Both preserved plans passed their existing watchdog `--dry-run` validators.
- A metadata-only lease test verified: a second watchdog cannot acquire an occupied GPU lease; a priority request reaches a backfill lease; a different watchdog cannot clear that request; and higher-priority ownership can acquire after release.

## Current state and next action

The current resident legacy watchdogs can still create their old bounded backfills until they exit, so the live overlap is recorded rather than modified in place. Apply the new coordination on the next watchdog invocation after those legacy parents have ended; do not restart them solely to activate this change. Verify activation by checking only aggregate owner state and GPU 0--3 process metadata.
