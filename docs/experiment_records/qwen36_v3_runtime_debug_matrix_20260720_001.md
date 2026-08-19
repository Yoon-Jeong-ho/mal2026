# Qwen3.6 v3 Runtime Debug Matrix — 2026-07-20

**Status:** GGUF synthetic-runtime prerequisites passed; native-FP8 vLLM is
blocked. This is a debugging matrix only, not a judge pilot or a selection,
training, SFT, DPO, or GRPO result.

## Scope and isolation

- Run ID: `qwen36-v3-runtime-matrix-20260720-001`; Git SHA:
  `86902f1e3a077b1178d1297a1dcccf10e929453d`.
- Used only GPUs 4, 5, 6, and 7, each initially idle. No GPU 0--3 query,
  process, or artifact was touched.
- Inputs were literal fixed Korean synthetic structured controls. No student
  essay, source feedback/candidate, validation, or API data was opened.
- The pinned Q4 GGUF checksum and pinned llama.cpp revision/tag were verified
  for each GGUF lane. Servers bound only to `127.0.0.1`, used distinct ports,
  and were terminated after their bounded runs. Aggregate files contain no
  prompt or completion; they retain only counts, decision hashes, timing, and
  gate outcomes.

## Aggregate results

| Lane | Result | Aggregate evidence |
| --- | --- | --- |
| GPU 4, Q4 GGUF / llama.cpp `--parallel 4` | Ready for synthetic runtime prerequisite | Valid pointwise was eligible; invalid evidence ID was non-eligible with content gate false; identity pair was neutral; invalid pair abstained; all 16 responses schema-valid and semantically repeatable. |
| GPU 5, label-free independent scoring | Ready for synthetic runtime prerequisite | Four repeats each produced eligible aggregation for valid control, ineligible for invalid-ID control, and abstain for unavailable-evidence control; all 12 responses schema-valid and repeatable. |
| GPU 6, concurrency | Ready for synthetic runtime prerequisite | Eight fixed-schema requests passed at parallel 1 (2.737 req/s) and parallel 4 (4.313 req/s); the fixed synthetic output hash was unchanged. Both servers stopped cleanly. |
| GPU 7, vLLM native-FP8 audit | Blocked | Local metadata finds vLLM 0.25.1, torch 2.11.0, and flashinfer-python 0.6.13, but zero local native-FP8 weight files. No vLLM import or server launch occurred. |

The first GPU-4 artifact (`gpu4-pointwise/aggregate.json`) is retained as a
negative harness result: all semantic controls passed, but a full JSON hash
varied because the unconstrained explanatory `reason` field differed. The
versioned retry compares only the decision and declared hard gates—the actual
contract fields—and passed. This does not establish byte-identical free-text
reasons.

## Exact recommended next action

The active results-and-next-stage lead may mark the three GGUF synthetic
runtime prerequisites as satisfied, while keeping the full v3 judge pilot,
selection, and all training blocked pending their separate authorization and
the GPU 0 ownership protocol. Do not attempt native-FP8 vLLM serving: it needs
a separately authorized local native-FP8 model artifact and a fresh readiness
preflight. The machine-readable handoff is the ignored aggregate-only
[`matrix_aggregate_recommendation.json`](../../outputs/debug/qwen36-v3-runtime-matrix-20260720-001/matrix_aggregate_recommendation.json).
