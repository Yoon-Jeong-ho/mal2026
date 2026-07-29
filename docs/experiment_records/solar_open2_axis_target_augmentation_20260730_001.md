# Solar-Open2 axis-target augmentation — run series 001

Status: **negative feasibility checkpoint; full not ready and not authorized**

## Scope and immutable bindings

This run series tests train-only, source-grounded augmentation against the
three-axis 1--5 rubric parsed from `evaluation.txt`. Validation data is bound
only for the train/validation disjointness check and is not used to generate,
filter, or select variants. No `average` field is generated or consumed.

- Git SHA at the current prompt-iteration checkpoint:
  `de71382557b3356780748fdbf40567a5aae9f750` (worktree dirty and recorded as
  such in run manifests).
- Generator and blind verifier:
  `/dataset/large-models/nota-ai/Solar-Open2-250B-Nota-INT4`.
- Runtime weights: 142,924,057,368 bytes in 27 shards.
- Official image ID and repository digest:
  `sha256:b34b7dd42d986bafb8dcfb285758de2fbfd55e7cf28a1b5ba6b858e6b2b7c8c3`.
- Hardware scope: physical H100 80-GB GPUs 0--3 only, tensor parallel 4 and
  expert parallel enabled. GPUs 4--7 were neither queried nor used.
- Current prompt config SHA-256:
  `87427079e91909c899fe5ab5e60a32981775400dfe685475a3db652f05c91d9b`.
- Current core implementation SHA-256:
  `18743d153d87953f4902b3806f138d8878fc9b0e8eae4702f361e79a3ba0cb1e`.
- Runner SHA-256:
  `dd17291856d9e76d7eaa6a094d361e42287b562be97317be613276a9a24262e2`.

The official Docker server loaded successfully once on all four authorized
GPUs and served the OpenAI-compatible endpoint with a 4,096-token model
length. The server was reused across prompt probes rather than reloading the
142.9-GB checkpoint. Near-full reported GPU memory is expected because vLLM
preallocates its KV-cache pool; observed utilization reached 100% on all four
GPUs while a request batch was executing.

## Fixed safety and evaluation contract

Every candidate starts from the immutable canonical training essay, not from
a previous variant. Four operation families are independent sequential
fallbacks, not a four-way comparative selector; no candidate, score, rationale,
or judge feedback is passed to a later family.
The editor receives the original prompt, essay, and three source scores. The
blind verifier receives only the official scoring prompt, the original task
prompt, and the candidate essay; it does not receive a target score or source
score. A candidate is accepted only when the target axis equals the requested
integer and the other two blind-verifier scores equal their scores on the
canonical source. A separate fidelity check rejects topic, stance, genre, and
external-fact changes.

Structured edit schemas preserve canonical lineage and constrain operations:
sentence-positioned content/expression edits, sentence-inventory-preserving
organization plans, and source-indexed evidence additions for content 5. The
latest expression-5 exact-span experiment was reverted after it generated no
valid scored candidates; the stable sentence-positioned schema is restored.
Focused CPU validation after the restoration passed **29/29 tests**.

## Random-five smoke evidence

The deterministic, order-independent SHA-256 sampling procedure selected five
training sources. The same five sources were used for all diagnostics below.
No row content or identifier is included in this record.

### Strict 75-cell smoke 017

The requested Cartesian smoke contains five sources times three axes times
five target scores, or 75 cells. It accepted **28/75** cells and rejected 423
candidate attempts. There were 47 terminal failures, dominated by 32 target
score failures. The preserved aggregate result is:

- `outputs/solar-axis-target-v2/solar-open2-axis-target-v2-smoke-20260730-017/result.failed.json`
- SHA-256:
  `99772764f975bf42e5c8efde3ca5e4edb97ab6481fc6f48c8e06d0e29e23a608`.

The accepted score-cell matrix was incomplete on every axis and had no
target-5 success. Manual and subagent aggregate audits found that exact
content-1 candidates were judged `(1,1,1)`, not as isolated content damage.
This is evidence of cross-axis coupling, not permission to relabel a near
miss.

### Extreme-cell diagnostics 018b--018c

Twenty attempts per selected extreme cell were generated from four operation
families. Content-1 reached its target in three scored candidates, but all
three simultaneously reduced organization and expression to 1. Organization-5
had no target-5 candidate; one candidate improved organization from 3 to 4
while exactly preserving the two non-target scores. Content-5 and
expression-5 likewise had no target-5 candidate. Mechanical organization
connector failures were removed in the subsequent prompt/schema revision, but
the target ceiling did not change.

Preserved aggregate SHA-256 values:

- 018b: `0d4b0f9cf07548a4ca5008ab8902b51e59bee7888a8cad5dab4d212ff0994202`;
- 018c: `dd9ae1c96dcf8e3872e11309f667716de209ff7159590eb7142f7157102d5bf6`.

### Judge ceiling control

A separate source-grounded holistic rewrite control showed that the verifier
can emit the top score: four of five canonical rewrites were judged `(5,5,5)`
and the remaining one `(4,4,5)`. Therefore the missing isolated target-5
results are not a hard decoder or verifier output ceiling. The aggregate is
preserved at
`outputs/solar-axis-target-v2/canonical-all5-rewrite-probe-20260730-002/aggregate.json`,
SHA-256
`a8be408801285d07c1c8480bb58d153009f205dfa2a8caee981ecb39b921f444`.

### Expression-5 schema restoration probes 019--020

After restoring the stable sentence-positioned expression schema, five
expression-5 tasks tried all four operation families:

| Run | Editor reasoning | Scored candidates | Expression score distribution | Exact accepted tasks |
|---|---|---:|---|---:|
| 019 | none | 15/20 | 3: 7, 4: 8, 5: 0 | 0/5 |
| 020 | medium | 17/20 | 2: 3, 3: 8, 4: 6, 5: 0 | 0/5 |

Medium reasoning reduced mechanical failures from five to three but made the
score distribution worse. It is therefore retained as a negative result and
is not adopted. The exact execution commands were:

```bash
PYTHONPATH=src .venv-standard/bin/python \
  outputs/solar-axis-target-v2/solar-open2-axis-target-v2-expression5-probe-20260730-019/probe_driver.py
PYTHONPATH=src .venv-standard/bin/python \
  outputs/solar-axis-target-v2/solar-open2-axis-target-v2-expression5-medium-reasoning-probe-20260730-020/probe_driver.py
```

The ignored probe drivers and manifests preserve the endpoint, external
container binding, source-score digest, configuration, implementation, and
driver digests. Aggregate hashes are:

- 019 aggregate:
  `e2245d91f7fe49af897fcf03e2b2f973268fe6eb7ab08f3572515f422f858e99`;
- 020 aggregate:
  `7dbddbab3a031474ca37517b2a5ad80ae9f01ffe1100cae67b0935ac06e3a2a9`.

### Final unchanged-gate local-edit probe 021--023

A subagent audit of 019--020 found that the candidates receiving 3 were edited
more aggressively than those receiving 4. The final prompt revision therefore
kept every gate unchanged and added only an ordered silent checklist for
punctuation, notation, agreement and clause links, collocation and register,
and sentence-internal redundancy. It also restated the conditional JSON
invariant `apply=false` implies an empty replacement. Reasoning remained
disabled.

Runs 021 and 022 exposed an integration boundary rather than a scientific
result: with a 1,900- or 1,800-token editor cap, the longest selected prompt
plus requested completion exceeded the server's fixed 4,096-token context by
one token. The server stayed healthy and returned explicit HTTP 400 validation
errors. The cap was reduced to 1,700 without changing the edit or score gates;
run 023 then had no context-length rejection.

Final run 023 produced 14 judgeable candidates from 20 attempts. Their
expression distribution was 2: 1, 3: 7, 4: 6, and **5: 0**. The remaining
mechanical failures were one conditional schema violation and three lexical
scaffold violations. No source produced an accepted expression-5 variant.
This completes the predeclared last conservative prompt probe.

Preserved aggregate SHA-256 values:

- 021: `76b8bc15d1369d790d4329eee60901b808425f26bffa0a961efa8f7e3498f35a`;
- 022: `d1d1ae1c821d20104d60d614f83ee78fa10f4843ed30505ca1f2e8eed1645420`;
- 023: `46e2c1fabd067c22b42dc709f07da1e336a712c2a156f5cf6340fd0404d68fdb`.

## Scientific interpretation and stop boundary

The official rubric asks a judge to assess three perspectives independently;
it does not imply that the underlying properties can always be manipulated as
a full Cartesian product for every essay. Removing all meaningful evidence
can also damage coherence and natural expression. Conversely, creating a
fully sufficient argument can require organization and wording changes.

The evidence therefore contradicts the current requirement that every one of
the same five sources yield all 15 exact axis-target variants while both
non-target integer scores remain unchanged. Launching 30,000 requested-label
rows now would create missing cells or mislabeled pseudo-targets. The full run
and all downstream score-model retraining remain intentionally unlaunched.

Continuing requires a recorded scientific choice between:

1. label every source-grounded candidate with its three actual blind-verifier
   scores and separately tag exact single-axis variants; or
2. preserve the exact single-axis gate while selecting different feasible
   training sources per axis-target cell and recording infeasible cells and
   selection bias.

The strict same-source Cartesian design remains fail-closed; near misses are
not silently accepted or relabeled. The checked-in prompt config explicitly
sets `execution_gate.full_run_authorized=false`, and the runner rejects full
mode before model or dataset work unless a later authorized protocol changes
that bound value. The focused suite covering this fail-closed gate and the
augmentation contracts passes **31/31 tests**.

Before any later authorization flips the full-run gate, the smoke-approval
validator must be hardened to canonical-replay all 75 accepted rows: exact
task IDs and source lineage, structured editor reconstruction, blind
verifier/source-verifier/fidelity fields, score provenance, and
`validate_candidate` must be rechecked. Reviewer coverage must likewise bind
distinct lead and subagent identities to the exact task-ID set rather than
counts alone. These dormant gaps cannot currently reach full execution. The
external runtime binding has already been hardened to require host network and
IPC, auto-remove, the offline environment, the writable vLLM cache mount, the
read-only model mount, the exact inner command, and physical GPUs 0--3.
