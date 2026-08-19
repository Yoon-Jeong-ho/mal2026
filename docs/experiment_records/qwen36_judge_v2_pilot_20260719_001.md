# Qwen3.6 GGUF judge-v2 pilot: 20260719-001

**Status:** the split-isolation prerequisite is complete and bound into the v2
preflight. No judge request, synthetic smoke, GPU workload, selection artifact,
SFT, DPO, or GRPO ran.

## Immutable pilot intent

- Run ID: `qwen36-judge-v2-pilot-20260719-001`
- Candidate batch lineage: `openai-rationale-terra-full-20260719-001`
- Protocol/config: `configs/qwen36_gguf_judge.v2.pilot.json`
- Implementation: `scripts/judge_feedback_candidates_v2.py`
- Planned runner: `scripts/run_qwen36_judge_v2_pilot.sh`
- Git SHA at implementation/preflight: `86902f1e3a077b1178d1297a1dcccf10e929453d`
- Config SHA-256 after restricted-artifact binding:
  `f99717506e22c81601ecb967353d17db7058e6548a61d8ac52a1ca385319d701`
- Judge-script SHA-256 after lineage binding:
  `61bd023c411bfa40ef9c943ca4f611949e726d34a7526e9556061f38fd10f39b`
- Runner SHA-256: `8bccf6053b0f664eed309ee8e8e60ef6694aa79e5b1348993d24559241ee7c68`
- Fixed maximum sample: 128 train essays, deterministic SHA-256 rank under
  seed `2026071902`.
- GPU envelope: physical GPU 0 only, with an owned localhost-server
  attestation.  GPUs 4--7 are excluded from query, visibility, and use.
- Selection/training boundary: disabled in configuration and absent from the
  pilot implementation.

## Preflight evidence

The v2 implementation requires a pre-existing restricted
`candidates.train.jsonl` plus `candidates.train.manifest.json` that attests
only train rows and its checksum.  The existing lineage contains only its
combined candidate artifact.  Rather than deserialize/filter its validation
records, `prepare` failed closed before opening any candidate payload or
creating a pilot directory/request.

Aggregate preflight result: exit status `1`; stdout bytes `0`; expected failure
class `missing_train_only_candidate_artifact`.  No student text, candidate
text, source identifier, or candidate identifier was emitted.

## Artifact-layout remediation audit (aggregate-only)

**Outcome:** fail closed; no path alias or artifact conversion was permitted.

The restricted lineage has one completed combined candidate file:
`candidates.jsonl` (12,915,060 bytes, 7,200 nonblank records, SHA-256
`4ef414dd35b831092fcea24c7770a5f18fbe0df1d4e6aa74d55613b8cad71e2e`).
Its validated producer manifest reports 3 candidates per essay, 2,000 train
rows, 400 validation rows, 7,200 accepted records, and zero rejected or
missing records.  Its combined-file checksum agrees with that manifest.

No `candidates.train.jsonl`, `candidates.train.manifest.json`, or other
split-scoped candidate artifact/manifest pair exists in the restricted layout.
Repointing v2 to the combined file would require deserializing an artifact
whose manifest explicitly covers validation, so it is not a safe config-path
repair.  The configured split-scoped paths remain correct; the authoritative
train-only input is absent.

The v2 guard was minimally strengthened to require the train-only manifest's
numeric `row_count` to equal a streaming, nonblank JSONL byte-line count, in
addition to the existing regular-file, non-symlink, lineage, split, schema,
and SHA-256 checks.  This count check occurs before either candidate or source
deserialization.  Repaired script SHA-256:
`94a2d4bc6d83555e16a5f2766f6f5c46860711965b0742374135e6778187c13c`.

### Exact safety checks

- Artifact inspection used only filenames, byte sizes, SHA-256 values, line
  counts, and numeric split-level manifest metadata.  No candidate, provider,
  judge, essay, explanation, or source-identifier payload was read or
  emitted.
- A synthetic guard replaced both candidate and train-source deserializers.
  `prepare` failed at the missing-artifact guard with zero calls and no pilot
  destination/request directory created.
- The real `prepare` preflight exited 1 with zero stdout bytes and no pilot
  destination/request directory.  Its expected classification is
  `required_train_only_artifact_absent_fail_closed`.
- Static checks passed: Python compilation, runner shell syntax, five
  dependency-free synthetic contract tests (including matching/mismatched
  train-manifest count coverage), and `git diff --check`.
- The bounded internal read-only review found the initial missing count-binding
  blocker; its one permitted re-review passed after the repair.  The review
  also confirmed train-only source routing, no validation request route,
  GPU-0-only wiring, and synthetic-only test inputs.
- No validation rows were opened, no validation requests were constructed, no
  judge server or pilot stage was started, and no GPU was queried or used by
  this audit.  GPUs 4--7 were untouched.  No candidate selection, SFT, DPO,
  or GRPO action ran.

## Static checks

- Python compile check passed for the v2 script.
- Shell syntax check passed for the GPU-0 runner.
- Synthetic-only contract checks passed: non-thinking request field, v2
  train-only/GPU-0/selection-disabled config, malformed-schema fail-closed
  handling, factorial label/position balance, and hybrid consensus mapping.
- The configured environment lacks `pytest`; no package was installed.  The
  same dependency-free test functions were invoked directly.

## Independent bounded review

One server-internal review and its one permitted re-review completed without
opening restricted inputs or using a GPU.  The initial checklist found three
blockers: combined-artifact validation leakage, an unattested server endpoint,
and missing enforcement of the schema's reason-length bound.  The bounded
repair and re-review passed the same checklist: the pilot now requires a
pre-existing train-only artifact before candidate deserialization, requires an
owned localhost GPU-0 server attestation before smoke/execution, and treats an
overlength reason as a schema failure.  No new review scope was added.

## Next permitted action

The bounded v2 `prepare` preflight may be rerun when separately scheduled. It
may proceed only through its synthetic smoke and hard gates. A passing pilot
still does not authorize selection or training.

## Restricted train-only derivation (2026-07-19 KST)

The validated combined parent artifact was transformed once into a **new,
restricted, non-overwriting** derived run directory. The transformer never
opened `eval/validation.jsonl`, constructed no validation request, and did not
start a server or query/use any GPU (including GPUs 4--7).

- Derived run: `train-only-candidates-v1-20260719-001`
- Transformer SHA-256:
  `5a0821e60abd1fb583a077291c1b29648c9e62f3bd5547e853c5d4091359ba8d`
- Completed train-only candidate checksum:
  `d69dd9bc349c117a55b4bf652507135f084c9045d95d5939da3980223893aecf`
- Completed derived-manifest checksum:
  `f26cb6da1d10bb13fd8802c01418f176e60d9366b393df7249419e2abb649861`
- Parent manifest checksum:
  `dbb736134686480a372bf1cada42be670bf84b360e9dc9a4b47cf6c12a7a8d8d`
- Parent combined-candidate checksum:
  `4ef414dd35b831092fcea24c7770a5f18fbe0df1d4e6aa74d55613b8cad71e2e`
- Parent authoritative source-map checksum:
  `3c01bcb2ff58ce0a8caae056a1bcdcc81e43e36d64b6d63166481d580d28d973`

The parent source map was the only split-routing authority. Aggregate routing
and completed-manifest proof: 6,000 train and 1,200 validation candidate input
records; 6,000 train and zero validation candidate output records; 2,000 train
and 400 validation input sources; zero validation output sources. Mapping and
candidate custom-key duplicates, source/candidate duplicates, unmapped records,
train/validation source overlap, and train/validation candidate-key overlap are
all zero. The inherited parent strict schema/grounding validation was clean;
the transformer also rechecked candidate routing/schema fields while copying
accepted train records byte-for-byte.

Before accepting the artifact, v2 now verifies the completed status, exact
train-only counts, streamed row count, output checksum, all three parent
checksums, parent strict-validation aggregate checksum, candidate schema, and
zero-valued isolation/deduplication proof. Its configured paths resolve only
within the parent restricted lineage and reject traversal. A synthetic tamper
test confirms a changed parent candidate artifact is rejected. Python compile,
seven dependency-free synthetic checks, and `git diff --check` passed; `pytest`
is unavailable and no package was installed.
