# OpenAI explanation v5.2 semantic-abstention taxonomy — 2026-07-20-001

**Role and scope:** TAXONOMIZE investigator; aggregate-only review of the
stopped, train-only, three-essay v5.2 GPU0 smoke. No raw essays, candidate
explanations, candidate identifiers, prompts, request/response payloads, keys,
runtime processes, or GPU state were read or changed. No code, configuration,
data, or existing artifact was changed.

## Gate outcome

The failed smoke is classified as a **semantic abstention-policy / response
contract conflict**. The recommended state-machine route is `TAXONOMIZE` to
`MINIMAL_PATCH`; this is a recommendation for the state-machine lead, not
authorization to patch or replay.

| Non-content-bearing category | Count |
| --- | ---: |
| `semantic_abstain_without_failed_gate` | 32 |
| JSON-schema parser categories | 0 |
| transport/retry categories | 0 |
| runtime/watchdog categories | 0 |

The first category is exact: the v5.2 inherited response normalizer assigns it
only when a response has `verdict=abstain` while all three declared hard-gate
booleans are true. It is therefore not a malformed JSON/schema response.
The aggregate metric named `transport_or_schema_failures` reports 32 because it
uses the runner's broad fail-closed bucket; its detailed
`failure_categories` map contains only the semantic category above. That
bucket must not be reinterpreted as 32 transport or JSON-schema faults.

The smoke aggregate contains 105 calls (45 deterministic, 45 dispersion, and
15 controls). It records zero validation requests, zero validation-source rows,
zero watchdog faults, a 350--1,264 prompt-token range under the 4,096-token
slot budget, and GPU0-only topology. Its hard-gate record failed
deterministic-repeat agreement, duplicate identity, evidence validity,
padded-verbosity, and the broad failure bucket; invalid control, context,
validation, sample-size, and watchdog gates passed. Cross-GPU agreement was
not evaluated on the single-GPU smoke.

## Diagnosis

**Most likely:** abstention-policy / semantic-contract failure. The v5.2 GPU0
synthetic preflight passed 20/20 schema-valid calls with zero final
transport/schema categories and all controls. The real smoke's sole detailed
failure category is the normalizer's explicit logically inconsistent abstain
state. Together with zero watchdog faults and in-budget prompts, this weighs
against runtime, context, transport, or JSON-shape/schema breakage.

**Not supported by the permitted evidence:** a data-quality diagnosis. The
aggregate-only evidence establishes neither whether any underlying writing is
defective nor whether any particular rubric, layout, repeat, or model behavior
caused the abstention. It must not be used to relax a hard gate.

## Minimal-patch nomination

Reject a blind prompt rewrite. The aggregate evidence identifies a
machine-checkable verdict-to-hard-gates inconsistency, not a wording-specific
cause; changing prose would confound the already-fixed request contract with
multiple unobserved variables.

Nominate exactly one future variable: **the response-schema conditional that
couples `verdict` to `hard_gates`**. Express the existing logical rule in the
response schema (a `scored` branch requires all gates true; an `abstain` branch
requires at least one gate false), rather than changing prompt text, scoring,
sampling, runtime, data, controls, or thresholds. This is only a candidate for
a separately authorized one-variable MINIMAL_PATCH followed by immutable
REPLAY; it is not implemented here.

## Limits of inference

This review cannot infer the affected writing, candidate, request, response,
layout, repeat, rubric axis, model rationale, or textual trigger; none was
read. It cannot determine whether the nominated schema conditional is accepted
by the serving stack, whether it removes abstentions without another failure,
whether evidence-validity or stability would then pass, or whether multi-GPU
agreement would pass. No selection, SFT, DPO, GRPO, pilot, full run, or replay
is authorized by this record.

## Evidence used

- `data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/judge_runs/openai-repeat-v5_2-gpu0-3-20260720-002/aggregate_pilot_report.json` (aggregate-only; SHA-256 `b3398eca04a5ab0112a9fd44e739cab3fbc5e65a5953617b242be0ab0b9569a3`).
- Its aggregate-only `manifest.json` (SHA-256 `6b90adb3fb11c00f80c6bee7cab6d28b3ea95a8f9594c63e34b5983dc75f9eac`).
- `docs/experiment_records/openai_explanation_repeat_distribution_v5_remediation_20260720_001.md` (SHA-256 `b52ec86700d0dec3c3beb8fd23afc8b7727267afefab9f6b11cc122ecd491090`), including the v5.2 GPU0 synthetic-preflight aggregate digest `0b9bb5a41deb0fb64a758170c3ec7d065d91d4ae98da70a80d78f6a6f856df73` and server-attestation digest `fc74580045ed37f5000bf362bb6205e7f9a2e971529f2573aba184aeaca1124b`.
- `configs/openai_explanation_repeat_distribution.v5_2.gpu0_3.json` (SHA-256 `daff36e1ffee5f7ee022f543cd988c65614fb4a20de8601741715e04ec904a97`).
- `scripts/run_openai_explanation_repeat_distribution_v5_2_gpu0_3.py` (SHA-256 `31459b013c93482735472fc5822a06996cedbe25cf0f9a40d321ea446dfe36a5`) and inherited response-schema normalizer in `scripts/preflight_openai_repeat_v5_synthetic.py`.

The machine-readable companion is
`outputs/aggregate-reports/openai_explanation_v5_2_taxonomy_20260720_001.gate-summary.json`.
