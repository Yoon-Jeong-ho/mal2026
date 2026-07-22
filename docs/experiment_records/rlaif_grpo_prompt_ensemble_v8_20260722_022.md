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

## Midm bundle full continuation and frozen evaluation (completed)

Both declared Midm bundle continuations reached all 480 updates.  `all5`
produced 7,680 parse-valid policy completions and 38,400/38,400 scored judge
requests.  `random1` produced 7,679 parse-valid completions and one canonical
policy-format invalid completion (therefore 7,679/7,679 scored judge
requests).  Both arms had zero unscorable judge outcomes, discarded groups,
transport retries, or judge failure categories; their training hard gates
passed.  The single policy-format failure in `random1` is distinct from a
judge failure and used the existing invalid-policy reward mapping.

Frozen-v6 evaluation used the same held-out 400-essay population and the
declared five forms by ten repeats for SFT, `all5`, and `random1`.  Every arm
has 20,000/20,000 schema-valid scores, zero abstentions, zero evaluator
failures, and all frozen-evaluation gates pass.  Aggregate-only results:

| decoder / arm | macro | content | organization | expression | paired macro Δ vs SFT (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| SFT baseline | 3.867967 | 3.839750 | 3.896100 | 3.868050 | — |
| GRPO `all5` | 4.174283 | 4.097300 | 4.268900 | 4.156650 | +0.306317 [+0.237733, +0.376617] |
| GRPO `random1` | 4.189100 | 4.104300 | 4.274600 | 4.188400 | +0.321133 [+0.252633, +0.392667] |

All three axes improve over SFT with positive paired 95% bootstrap intervals
in each arm.  This is target-proxy evidence only: the frozen evaluator is the
same Qwen family as the training judge (with a distinct fixed evaluation
protocol), so it is not human-quality evidence.  The two arm intervals above
compare each arm to SFT, not each other; one seed and no direct paired
`all5`-versus-`random1` confidence interval do not support declaring either
reward construction superior.
