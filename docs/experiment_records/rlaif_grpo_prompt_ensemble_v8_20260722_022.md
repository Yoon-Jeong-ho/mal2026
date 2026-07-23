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

## A.X bundle rollout-terminal recovery

The first A.X-4.0-Light bundle/`all5` continuation (`20260722-022`) stopped
at global update 204.  Its aggregate-only failure record shows 3,264/3,264
canonical policy completions, 16,320/16,320 successful Qwen judge calls, and
zero judge failures, retries, or discarded groups before a vLLM policy choice
returned a terminal reason other than `stop`.  No adapter or evaluation was
accepted from that partial arm.

The failure was in policy transport handling, not a Qwen reward label: the
client previously rejected a returned completion before canonical parsing just
because its `finish_reason` was not `stop`.  In vLLM the terminal reason is
transport metadata; a returned completion can still be canonical, while a
truncated or malformed completion is already handled by the score-blind
canonical policy parser and existing invalid-policy reward path.  Recovery
commit `d3e8c6e2e1324d4d068c385976f918a43552e875` therefore retains each
returned string, records aggregate terminal-reason counts, and leaves
canonical validity—not the terminal tag—as the acceptance boundary.  It does
not retry, resample, assign a judge-derived substitute, or change the model,
prompts, sampling, reward, data, or optimizer.

The failed partial output remains preserved under the ignored runtime root. A
fresh `20260722-023` run invokes `remaining` with the unchanged v8 config and
the repaired source.  Its initial Midm pair is reused only because both
previously completed under the stricter `stop` condition; the fresh A.X arm
starts from its SFT adapter rather than resuming the partial adapter.  All
subsequent records expose non-`stop` completion counts in aggregate metadata.

- Relaunch time: 2026-07-22T23:44:02Z.
- Command: `MAL2026_RLAIF_CONFIG=configs/rlaif_grpo_prompt_ensemble.v8.json MAL2026_RLAIF_RUNTIME_ID=20260722-023 PYTHONPATH=src .venv-standard/bin/python scripts/run_rlaif_grpo_prompt_ensemble_v1.py remaining`.
- Source Git SHA at relaunch: `f8f9dd621681eb72cf6f02bd6029d45823fa128e`.
- Unchanged v8 config SHA-256: `9528b01263d3e4bd2f5cbdf7f272beedd2fdbff697592a36462f1ec9d51fe375`.

## A.X bundle full continuation and frozen evaluation (completed)

The fresh A.X-4.0-Light bundle pair completed all 480 updates and passed the
training gates.  `all5` had 7,680 canonical completions and 38,400/38,400
successful judge calls.  `random1` had 7,678 canonical completions, two
canonical-policy invalid completions, and 7,678/7,678 successful judge calls.
The repaired non-`stop` path was exercised exactly twice in `random1`
(`length: 2`, `stop: 7,678`): both responses then failed the existing canonical
policy parser and received only its existing invalid-policy treatment.  Neither
was retried, substituted, or passed to Qwen.  There were zero judge failures,
unscorable outcomes, discarded groups, and transport retries in both arms.

Frozen-v6 evaluation again produced 20,000/20,000 schema-valid observations
per arm (400 essays × 5 forms × 10 repeats), with zero abstentions, evaluator
failures, or gate violations:

| decoder / arm | macro | content | organization | expression | paired macro Δ vs SFT (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| SFT baseline | 3.766183 | 3.685500 | 3.869800 | 3.743250 | — |
| GRPO `all5` | 4.184067 | 4.121450 | 4.141700 | 4.289050 | +0.417883 [+0.349500, +0.489667] |
| GRPO `random1` | 4.187033 | 4.142400 | 4.146650 | 4.272050 | +0.420850 [+0.349567, +0.492683] |

All requested axes improve with positive paired bootstrap intervals for both
arms.  As with Midm, the small point difference between arms is not a direct
between-arm statistical comparison and remains target-proxy evidence rather
than a human-quality claim.

## A.X content-only continuation and frozen evaluation (completed)

The A.X content-only pair reached all 480 updates and passed the v8 training
gates.  `all5` produced 7,680 parse-valid completions and requested 38,400
Qwen observations.  One terminal judge response was unscorable; exactly one
four-candidate generation group was assigned the declared equal zero reward,
leaving 38,399 successful Qwen calls.  Its unscorable rate and discarded-group
rate are both below the v8 ceilings.  `random1` produced 7,680 parse-valid
completions and 7,680/7,680 successful Qwen calls, with zero unscorable
observations, discarded groups, retries, or transport/schema failures.

Each SFT and RL arm again has 20,000/20,000 schema-valid frozen-v6 scores,
zero abstentions, zero evaluator failures, and all frozen-evaluation gates
passed.  The requested primary target for this task is `content`; the other
two axes are reported as transfer diagnostics rather than optimized targets.

| decoder / arm | macro | content | organization | expression | paired content Δ vs SFT (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| SFT baseline | 3.922283 | 3.867950 | 3.930350 | 3.968550 | — |
| GRPO `all5` | 3.995050 | 4.195400 | 3.815550 | 3.974200 | +0.327450 [+0.261950, +0.394800] |
| GRPO `random1` | 4.012033 | 4.223550 | 3.831800 | 3.980750 | +0.355600 [+0.290450, +0.425000] |

Both arms improve the declared content target.  Neither yields evidence that
the organization rationale transferred positively: its paired deltas are
`all5` −0.114800 [−0.153350, −0.076750] and `random1` −0.098550
[−0.136850, −0.059400].  Expression is statistically inconclusive in both
arms.  Thus a content-only continuation must not be interpreted as an
all-three-axis quality improvement; the separately trained bundle arms are
the appropriate primary comparison for that claim.

## A.X organization-only continuation and frozen evaluation (completed)

Both organization-only arms reached 480 updates with 7,680/7,680 canonical
policy completions, no parse failures, terminal non-`stop` completions,
unscorable judge observations, discarded groups, or transport retries.  The
`all5` arm scored all 38,400 requested Qwen observations (7,680 per form),
while `random1` scored all 7,680 and selected the five forms near uniformly
(1,540; 1,532; 1,535; 1,525; 1,548).  Both training gates passed.

Frozen-v6 evaluation also passed all gates for each arm, with 20,000/20,000
schema-valid score observations, zero abstentions, and zero evaluator
failures.  The requested primary target is `organization`.

| decoder / arm | macro | content | organization | expression | paired organization Δ vs SFT (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| SFT baseline | 3.922283 | 3.867950 | 3.930350 | 3.968550 | — |
| GRPO `all5` | 4.040150 | 3.927800 | 4.181000 | 4.011650 | +0.250650 [+0.176400, +0.321600] |
| GRPO `random1` | 3.994417 | 3.922950 | 4.040600 | 4.019700 | +0.110250 [+0.040850, +0.178150] |

Unlike content-only training, both organization-only arms have positive paired
intervals on all three diagnostic axes.  The `all5` point estimate is larger,
but the current confidence intervals are each versus SFT rather than a direct
between-arm comparison; one seed is insufficient to declare an ensemble
winner.  Prompt-form organization ranges (0.235000 and 0.287250) also exceed
the observed arm point difference (0.140400), reinforcing that caution.

## Framework-conformance audit during the declared run

The active environment is TRL 0.29.1, vLLM 0.25.1, Transformers 5.14.1,
PyTorch 2.11.0, Accelerate 1.14.0, and PEFT 0.19.1.  TRL's own import-time
compatibility warning does not list vLLM 0.25.1 among the versions supported
by its integrated vLLM path.  The v8 runner therefore deliberately uses the
publicly provided (but experimental) `GRPOTrainer.rollout_func` boundary
rather than falsely declaring the built-in `use_vllm=True` integration
compatible.

The custom path was compared with the current TRL, OpenRLHF, verl, and vLLM
interfaces.  Its intended properties are present: four samples per source are
coalesced into one `n=4` vLLM request; reward groups retain those four members;
the policy LoRA is snapshotted and loaded in place immediately before every
generation batch; and the rollout server (GPUs 0--1), trainer (GPU 2), and
reward judge (GPU 3) remain separate.  For example, the completed A.X
organization/`all5` arm records 1,920 vLLM requests, 7,680 completions, and
120 adapter syncs, which is exactly one synchronization per four-update
generation batch (480 updates total).  The all-five arm also records 7,680
calls for each prompt form; completed `random1` arms have near-uniform
one-form counts, as required by the deterministic selector.

There is one explicit limitation rather than an unrecorded claim of exact
framework equivalence.  The HTTP custom rollout consumes vLLM completion text
and re-tokenizes it for TRL; it does not carry vLLM's sampled per-token
log-probabilities into a TIS/MIS rollout-correction calculation.  Recent TRL
and verl implementations expose such corrections because inference/training
numerics can differ.  This limitation is common to both declared arms, so it
does not selectively favor `all5` or `random1`, but it limits an *absolute*
claim about the GRPO update.  The v8 matrix is not altered mid-run.  A future,
separately versioned verification arm may capture rollout log-probability
diagnostics or use a supported integrated stack; it must not be mixed with the
declared v8 comparison.
