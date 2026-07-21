# RLAIF/GRPO prompt-ensemble v3 policy-length gate — 2026-07-22

- **Status:** the bounded rollout-length repair passed its preflight and both
  one-update arm gates. Its paired all-five pilot then failed the declared
  decision-scale latency bound and was stopped before any usable adapter,
  validation generation, or score comparison was produced.
- **Unchanged:** the 2,000-row train source, reserved 80-row contrastive
  holdout, eligible 1,920-group population, SFT adapter, four policy samples,
  five-form/all-five and random-one arms, Qwen reward model/forms/seeds,
  reward mapping, optimizer, GPU allocation, and frozen-v6 evaluation plan.
- **Changed policy-output bound only:** each generated rationale field is
  limited to 128 Korean characters rather than 192.  The SFT target-field
  aggregate distribution supports this bound: among 18,000 train fields,
  median/p95/p99/max length was 80/114/126/160 characters; 143 fields (0.79%)
  were at least 128.  No raw target text was inspected or copied.

## Why this repair was necessary

The v2 full run did not fail JSON parsing or Qwen scoring.  Its first full
batch parsed 64/64 and made 320 Qwen calls, but a subsequent 64-continuation
batch drained to a small, very slow decode tail, hit the 600-second client
limit, and deterministically retried.  This is a per-batch rollout latency
problem, not evidence that fewer training essays would improve the reward
model or rationale quality.  Its negative record remains preserved in the v2
experiment record.

The v3 field limit retains nearly all of the observed SFT rationale-length
distribution while giving the JSON grammar a materially smaller maximum
completion space.  It does not change the final frozen-v6 evaluation contract:
validation still accepts the original 192-character rationale-only schema, so
post-RL scores remain comparable to the frozen SFT baseline.

## Full-batch live policy gate (`20260722-011`)

Using GPU0 only, local vLLM 0.25.1, the Midm bundle SFT adapter, the exact
score-blind policy prompt, 16 train-only source prompts, four samples per
prompt, temperature 0.7, top-p 0.95, and the v3 128-character JSON schema,
the full 64-completion rollout batch completed in **190.588 seconds**.  The
strict parser accepted 64/64 (1.000); p50/p95/max HTTP request latency was
183.059/190.247/190.584 seconds.  The promotion limit was 240 seconds.

The aggregate-only gate is
`outputs/rlaif-grpo-prompt-ensemble-v3/20260722-011/aggregate/midm2_bundle_v3_full_batch_gate.json`.
It contains no essay, prompt, completion, score, candidate score, or judge
response.  Qwen was not started and no optimizer update occurred.  The
preflight passed, but it is not an RLAIF outcome and does not justify a score
claim.

## Declared decision-scale pilot

The next stage is a **320-group**, one-pass Midm bundle decision experiment,
not a replacement claim for the 1,920-group study.  It has 20 policy rollout
batches and 80 optimizer updates per arm, retaining four samples per source
and four policy updates per rollout.  The 320 groups are selected from the
same eligible 1,920 train-only population by a deterministic opaque
SHA-256-based rank using the fixed policy seed; no source identifier list is
written.  The all-five arm makes 6,400 Qwen reward requests and the random-one
arm 1,280.  Both adapters are evaluated with the unchanged frozen-v6
validation protocol before any full-scale or remaining-model promotion.

## Negative pilot execution result and stop (`20260722-011`)

The all-five pilot was deliberately stopped before a usable adapter or
validation artifact was produced. This is an **operational negative result**,
not a score result: the first two exact DDP rollout groups reached the Qwen
reward and four optimizer updates each, but took 323.4 s and 646.1 s for their
first updates, respectively. The second group exceeded the 240 s predeclared
full-batch promotion limit by a wide margin. In both groups parser success,
reward standard deviation, and completion lengths were healthy (mean/max
policy completion tokens 125.4/162 then 129.9/186; Qwen mapped reward
mean/std 0.3687/0.3200 then 0.4089/0.2652). Thus it is not evidence of a
reward, parsing, or learning failure.

vLLM's aggregate server log showed the defect directly: after most of each
64-continuation structured JSON-schema batch completed, one or a few requests
remained at about 0.1 generated tokens/s with only 0.6% KV-cache use. The
first tail lasted about 100 s; the second about 350 s. The run was interrupted
cleanly, ignored logs and partial artifacts were retained, and its orphan DDP
worker was terminated after the runner exited. No partial adapter is evaluated
or used for any result.

The bounded v3 schema therefore failed the declared decision-scale latency
gate despite its single preflight batch passing. The isolated v4 repair keeps
strict JSON object/axis/schema-version decoding and the same 128-character
canonical parser, data, reward, optimizer, samples, seeds, and frozen-v6
evaluation. It removes only the redundant `maxLength` state from the *vLLM
decoding grammar*; canonical parsing still rejects an over-limit field before
reward. A fresh full 64-continuation live gate is required before any v4 policy
update.
