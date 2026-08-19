# OpenAI explanation v5.2 conditional response-schema remediation — 2026-07-20-001

**Run ID:** `openai-explanation-v5_2-conditional-20260720-001`
**Lineage:** `openai_explanation_v5_2_taxonomy_20260720_001`
**State transition:** `MINIMAL_PATCH` to `REPLAY`

## Authorized one-variable change

The response schema in `scripts/preflight_openai_repeat_v5_synthetic.py`
now declares the existing normalizer relationship: `scored` requires all three
hard gates true, and `abstain` requires at least one hard gate false. No prompt
prose, rubric, thresholds, sampling, data access, runtime topology, or
selection behavior changed. The normalizer itself was not changed.

## Bounded verifier and static gate

- Scoped internal verifier: pass; read-only review of the schema location and
  one-variable boundary. Its aggregate summary is retained in the run log.
- Static checks: Python compilation; the two v5/v5.2 contract modules; JSON
  Schema Draft 2020-12 compilation; and the eight immutable
  verdict-by-hard-gate truth combinations.
- Static outcome: all checks passed. Technical failures: 0. Semantic
  abstentions: 0. Immutable regression outcome: not run at this transition.

## Provenance

- Git SHA: `86902f1e3a077b1178d1297a1dcccf10e929453d` (dirty worktree was
  preserved; this remediation changes only the files named above).
- Response-schema script SHA-256: `77d998e364e380e66d37ce3c637b6ffbca6d76d703884d0754242d6a84fbaf2a`.
- Static-contract test SHA-256: `d4eebb981866898a292bd8811b62ce1091dade84ddf107933817cfe24f29ad25`.
- Fixed v5.2 config SHA-256: `daff36e1ffee5f7ee022f543cd988c65614fb4a20de8601741715e04ec904a97`.
- Runtime preflight SHA-256: `799765fa3c47a5ae9fc9ebcae2f307102453181d965cbde83990a0d9b8858527`.
- Runner SHA-256: `31459b013c93482735472fc5822a06996cedbe25cf0f9a40d321ea446dfe36a5`.
- GPU allocation for the next immutable replay: `[0]`; no GPU outside the
  project-owned range will be queried or used.

The machine-readable gate summary is
`outputs/aggregate-reports/openai_explanation_v5_2_conditional_20260720_001.minimal-patch.gate-summary.json`.

## Immutable GPU0 GOLDEN/REPLAY gate

The immutable synthetic replay ran on GPU 0 only, with the fixed v5.2 config
and no project-data access. It passed every gate: required rubric fields,
deterministic repeats, duplicate/invalid/padded controls, retry contract,
server liveness, and zero transport-or-schema failures. The report contains no
failure category, hence technical failures and semantic abstentions are both
0. Raw prompts and responses were not persisted.

- Replay run ID: `openai-repeat-v5_2-gpu0-3-20260720-003`.
- Replay aggregate SHA-256: `1b411f4a66eb65e665c8d52a5ef041803b3e28552a04b101ebcee7459df125a6`.
- Result: pass; transition authorized from `REPLAY` to the unchanged
  train-only `SMOKE`.

The corresponding machine-readable transition is
`outputs/aggregate-reports/openai_explanation_v5_2_conditional_20260720_001.replay.gate-summary.json`.

## Unchanged GPU0 train-only SMOKE gate

The fixed three-essay GPU0 smoke ran durably as
`openai-repeat-v5_2-gpu0-3-20260720-004`. It failed closed and stops the
forward path. No GPU1--3 utilization action or full run was started.

- Technical failures: 0. The broad runner bucket reports 28, but its sole
  detailed category is semantic, not transport or schema.
- Semantic abstentions: `semantic_abstain_without_failed_gate=28`.
- Failed unchanged gates: deterministic-repeat agreement, duplicate identity,
  evidence validity, padded verbosity, and the broad failure bucket.
- Passed unchanged gates: context budget, invalid control, sample size,
  validation rows/requests (both 0), and watchdog (0 faults).
- Immutable regression: pass. Data scope: train only. Validation was not
  opened or requested; no selection artifact was constructed.
- Aggregate report SHA-256:
  `3abec1daa07679452350f57e9dfbb831ce9405771c6727a8eec753c22da0ffbf`.
- Manifest SHA-256: `c0b558d1e6e4238f3e4745e4a77072f44697d92fa26f723df120c1df48c925ab`.
- Watchdog aggregate SHA-256:
  `adac8f21a0be0741773e3171b9fad0dfcdc9b3ac766294150efcb455a9645172`.

The state transitions from `SMOKE` to scoped `TAXONOMIZE`; no blind rewrite is
authorized. The machine-readable failure summary is
`outputs/aggregate-reports/openai_explanation_v5_2_conditional_20260720_001.smoke.gate-summary.json`.

## Scoped TAXONOMIZE outcome

The aggregate-only investigator classified the failed smoke as the same
semantic abstention-policy/response-contract category as the prior taxonomy,
not a transport, parser, schema-shape, runtime, context, or watchdog failure.
The tested schema conditional reduced the category from 32 to 28 but did not
eliminate it. The current state therefore remains `TAXONOMIZE`; no additional
variable was changed or nominated, and no downstream stage is authorized.

The aggregate-only taxonomy summary is
`outputs/aggregate-reports/openai_explanation_v5_2_conditional_20260720_001.taxonomize.gate-summary.json`.
