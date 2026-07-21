# RLAIF/GRPO prompt-ensemble v5 JSON-object rollout repair — 2026-07-22

- **Status:** follows the rejected v3 and v4 structured-schema latency/parse
  gates. No v5 policy update, Qwen reward call, adapter, validation generation,
  or score comparison has started.
- **Unchanged:** the score-blind 2,000-row train source, reserved 80-row
  holdout, opaque 320-group pilot subset, SFT adapter, four samples, both
  all-five/random-one arms, Qwen reward prompts and seeds, reward mapping,
  optimizer, GPUs 0--3 allocation, and frozen-v6 evaluation.

## Isolated change

v5 uses vLLM OpenAI-compatible `response_format: {"type": "json_object"}`
rather than the JSON-schema constrained decoder. The policy's canonical parser
still requires exactly `schema_version` plus the requested axes, each containing
only a nonempty Korean `rationale`; it returns the existing -1 invalid reward
before Qwen if any condition fails. The policy parser field limit returns to
the original 192 characters, matching frozen-v6 evaluation. This reversal is
necessary because v4's 128-character parser rejected 3/64 naturally completed
JSON objects; it does not expose scores or change the target rationale
semantics.

The required gate remains 16 train-only prompts x four samples, strict
canonical parse 64/64, and wall time at most 240 s. Only a passing fresh v5
gate permits the existing one-update arm gates and the 320-group paired pilot.

## Aggregate-only grammar result (`20260722-014`)

v5 achieved strict canonical parsing for all 64/64 completions, but its
single-GPU policy batch took 301.769 s (p50/p95/max 243.383/301.761/301.767),
exceeding the unchanged 240 s operational gate. No reward or optimizer call
occurred. Thus the JSON-object formatting path is retained as the only format
variant with a passing parser, but the one-GPU rollout topology is rejected
for the decision pilot.
