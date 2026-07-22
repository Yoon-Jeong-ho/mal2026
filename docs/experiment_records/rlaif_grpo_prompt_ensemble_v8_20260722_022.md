# RLAIF-GRPO five-form reward study v8 — 2026-07-22

## Objective and fixed comparison

Continue each score-blind SFT rationale decoder with GRPO, comparing the mean
of all five fixed Qwen judge forms (`all5`) against the deterministic one-form
selector (`random1`).  Each completed arm is evaluated against its SFT decoder
on the held-out 400-essay validation set with the unchanged frozen-v6 five-form
by-ten-repeat judge.  Neither training nor evaluation prompts expose source
writing scores or candidate score fields.

## v7 runtime result and recovery decision

`20260722-021`, A.X-4.0-Light/content/`all5`, stopped at global update 84 of
480 when one Qwen reward response ended with vLLM `finish_reason=length`.
Its aggregate-only failure record reports 1,344 valid policy completions and
6,720 successful judge responses before the terminal unscorable response; no
adapter or post-RL evaluation was produced.  The failed run is preserved under
the ignored output root.

The v7 client correctly did **not** retry `length`: only explicit vLLM internal
`error` and HTTP/connection failures are retryable.  Assigning the policy's
invalid-completion reward (-1) to a valid rationale whose *judge* response was
incomplete would incorrectly turn an execution failure into a quality label.

v8 therefore preserves the judge model, five prompts, request body, sampling
seeds, 192-token ceiling, retry policy, data population, and optimizer.  Its
sole failure-handling change is explicit and aggregate-recorded: an unscorable
terminal judge response discards the entire four-candidate GRPO generation
group by assigning the same zero reward to all four candidates, producing no
relative-advantage update.  It never retries or substitutes a score.  Gates
limit unscorable judge responses to 0.1% of requested observations and
discarded groups to 1% of groups.  This prevents infrastructure failures from
becoming low-quality targets while preserving all successful judge scores.

## Declared full run

- Config: `configs/rlaif_grpo_prompt_ensemble.v8.json`
- Command: `MAL2026_RLAIF_CONFIG=configs/rlaif_grpo_prompt_ensemble.v8.json MAL2026_RLAIF_RUNTIME_ID=20260722-022 PYTHONPATH=src .venv-standard/bin/python scripts/run_rlaif_grpo_prompt_ensemble_v1.py all`
- Git source commit at launch: `54955a6fd7979d9ed1be7cd821a7ac4f85e59dcc`.
- Config SHA-256: `9528b01263d3e4bd2f5cbdf7f272beedd2fdbff697592a36462f1ec9d51fe375`.
- GPUs: rollout vLLM tensor-parallel GPUs 0–1; float32 LoRA GRPO GPU 2;
  Qwen3.6-35B-A3B FP8 reward judge GPU 3.  GPUs 4–7 are never queried or used.
- Full arm contract: 1,920 eligible train-only groups, four completions per
  group, 7,680 policy completions, 64-completion generation batches, 480 GRPO
  updates, and no checkpoint resume from the failed v7 arm.
- Evaluation contract: 400 validation essays × 5 prompt forms × 10 repeats =
  20,000 score observations per arm; zero abstentions and zero evaluator
  transport/schema failures are required.

## Pre-launch verification

`py_compile` and `tests/test_rlaif_grpo_contract.py` pass (17 tests).  The new
unit contract injects one `envelope_length` per candidate and verifies that the
corresponding four-candidate group receives equal zero reward, while the
successful observations remain counted and the terminal failure category is
recorded only in aggregate metadata.

Results, wall times, final frozen-judge deltas, and any deviation will be added
after the declared runner completes.  Generated rationales, essays, adapters,
server logs, and runtime artifacts remain ignored.

## v8 actual-update gate (passed)

At 2026-07-22T14:00:27Z the GPU0-first gate completed on the declared final
topology before Midm full training.  Both one-update arms passed all gates:

| arm | policy completions | parse-valid | Qwen requests / scored | unscorable | discarded groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all5` | 4 | 4 | 20 / 20 | 0 | 0 |
| `random1` | 4 | 4 | 4 / 4 | 0 | 0 |

No retry, terminal judge failure, policy parse failure, source-score access, or
raw text persistence occurred.  The runner then started the declared Midm full
two-arm continuation automatically.
