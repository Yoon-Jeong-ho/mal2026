# RLAIF/GRPO prompt-ensemble v4 decoding-grammar repair — 2026-07-22

- **Status:** authorized after the v3 decision-scale policy-latency gate
  failed; no v4 policy update, Qwen reward call, adapter, validation
  generation, or score comparison has started.
- **Purpose:** isolate the vLLM structured-decoding latency defect observed in
  v3 without changing the RLAIF science.
- **Unchanged:** source/holdout population and opaque 320-group pilot
  selection, SFT adapter, four samples, five-form/all-five and random-one
  arms, Qwen model/forms/seeds, score-to-reward mapping, optimizer, GPUs
  0--3 allocation, and frozen-v6 validation protocol.

## Single implementation change

v3 used a strict JSON schema with a 128-character `maxLength` on every
`rationale` string, then repeated that same 128-character check in canonical
completion parsing. Its actual DDP pilot showed long constrained-decoding
tails after otherwise healthy 64-completion batches: first-update wall time
323.4 s then 646.1 s versus the 240 s promotion bound, while valid output
lengths were far below the cap. v4 retains strict JSON object shape, required
axes, schema version, and nonempty strings, but removes only `maxLength` from
the **vLLM grammar**. The canonical parser retains the 128-character cap, so
an over-limit output is invalid and receives the existing -1 reward before any
Qwen request.

This is a latency/formatting repair, not a change to a judge prompt, reward,
data split, output semantics, or final evaluator. Its explicit config switch
is `policy.rollout_json_schema_enforces_field_limit: false`; the completion
artifact reports that switch alongside parse validity. A fresh aggregate-only,
16-prompt x four-sample live rollout gate is required with the same 240 s
maximum wall limit. Only if it parses all 64 canonical completions and meets
that limit may the fresh 320-group paired pilot begin.

## Preserved launcher negative (`20260722-012`)

The first v4 launcher made no policy generation or optimizer update. Its
aggregate-only batch-gate runtime config was incorrectly bound to the
four-row `gpu0_actual` phase while requesting 16 source prompts, so the gate
rejected it immediately before sending an inference request. The log and
ledger are retained under ignored outputs. The follow-up launcher binds that
gate to the already-declared 320-row pilot phase (still slices exactly 16
train-only prompts and creates no training output) and uses a fresh runtime
lineage; no data, reward, model, or evaluation contract changed.

## Aggregate-only grammar result (`20260722-013`)

The corrected v4 gate completed its 64 policy samples in 225.462 s, within
the 240 s wall bound, but canonical parsing accepted only 61/64 (0.953125),
below the required 1.000 gate rate. No reward call or optimizer update
occurred. Therefore v4 is rejected: removing only the schema `maxLength`
reduced latency but did not retain the required rationale-only completion
contract. The aggregate report is ignored-output evidence at
`outputs/rlaif-grpo-prompt-ensemble-v4/20260722-013/aggregate/midm2_bundle_v4_full_batch_gate.json`;
it contains metrics and provenance only, not completions.
