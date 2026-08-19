# OpenAI explanation repeat-distribution v4 pilot — 2026-07-20-001

**Status:** failed preregistered gates; no candidate was chosen and no
selection, SFT, DPO, GRPO, or full rerun was constructed.

## Scope and provenance

- Run ID: `openai-repeat-v4-20260720-001`; Git SHA:
  `86902f1e3a077b1178d1297a1dcccf10e929453d`.
- Config: `configs/openai_explanation_repeat_distribution.v4.pilot.json`
  (SHA-256 `f668cea71e219e1ef5518c44735ada0992d42c1019afc428254324cf083a5909`).
- Input was only the verified derived train-only candidate artifact (SHA-256
  `d69dd9bc349c117a55b4bf652507135f084c9045d95d5939da3980223893aecf`),
  with a deterministic non-identifying sample of 96 essays and all three
  candidates. Validation source rows and validation requests were both zero.
- Four localhost Q4_K_M llama.cpp servers ran on physical GPUs 4, 5, 6, and 7
  only, each with `--parallel 4`. Their pinned GGUF checksum and llama.cpp
  revision/tag were preflight-verified. The GPU/memory/temperature watchdog
  recorded zero faults; all four server processes stopped at terminal.
- Restricted request and response payloads remain only under the ignored run
  root. This record contains no essays, explanations, identifiers, or raw
  outputs.

## Fixed request schedule

- 1,440 candidate-isolated deterministic calls: five rubric-order/prompt-layout
  permutations per essay/candidate at temperature 0.
- 1,440 candidate-isolated dispersion calls: five fixed seeds per
  essay/candidate at temperature 0.15.
- Controls: five duplicate-identity, five padded-verbosity, and five invalid
  evidence calls. No candidate labels or peer candidates were provided to the
  judge.

## Aggregate result

| Check | Observed aggregate | Gate |
| --- | ---: | --- |
| Candidate calls per candidate | 960 | recorded |
| Schema-valid rate (candidates 1 / 2 / 3) | 0 / 0 / 0.020833 | fail |
| Evidence-valid rate (all candidates) | 0 | fail |
| Invalid/abstain rate (all candidates) | 1.0 | fail-closed |
| Deterministic / cross-GPU agreement | not estimable | fail |
| Duplicate identity agreement | 0 | fail |
| Invalid-control abstain rate | 1.0 | pass |
| Padded-verbosity non-improvement | false | fail |
| Transport-or-schema failures | 2,860 / 2,895 calls | fail |
| Watchdog faults | 0 | pass |

No candidate/rubric score distribution was eligible for summary: each overall
distribution had `n=0`, so median, mean, IQR, standard deviation, and
length-score correlation are null rather than imputed. Aggregate server-log
diagnostics recorded cancellation/error events; this is an observed runtime
failure, not a scoring result.

The preregistered comparison rule required all controls and stability checks
to pass before a candidate's overall median could exceed each competitor by
combined uncertainty. It therefore returned `withhold_failed_gates` for all
three pairwise comparisons. The five-repeat strategy is **not viable for a
later full train-only selection with this runtime behavior**. A separately
versioned, bounded runtime repair/preflight must demonstrate schema-valid
pointwise outputs and all controls before any new selection pilot is proposed.

## Reproducibility pointers

- Command: `scripts/run_openai_explanation_repeat_distribution_v4.sh
  openai-repeat-v4-20260720-001`.
- Aggregate request checksum:
  `4f6f1337de33565cd0b405da2970d495ea8edd02f5a07b17c94c055934e0d058`.
- Restricted raw-response checksum:
  `7ee84c62af7ef185ec87584e2cfce07dedd0da2a62f59c434dc4ce3eabda6172`.
