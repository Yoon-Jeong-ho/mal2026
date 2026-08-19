# Synthetic-feedback explanation SFT: immutable pre-SFT contract v1

**Status:** pre-SFT data/model contract only.  No SFT, DPO, GRPO, checkpoint
selection, or explanation scoring is authorized or implied by this document.
Changing any normative requirement below requires a new, versioned contract and
a recorded deviation; it must not silently alter an existing candidate run.

## Purpose and scope

This contract defines a future explanation generator, not a scorer.  It may
produce Korean writing feedback that is grounded in numbered sentences, while a
separate frozen encoder supplies the three analytic score inputs.  The
generator must neither predict nor repeat a score.

All row-level inputs, OpenAI requests/responses, candidate mappings, selected
targets, and derived SFT JSONL remain under the restricted, Git-ignored
operational artifact root.  This tracked document contains no student text,
identifiers, prompts, feedback, generated rationales, or credentials.

## Split isolation (hard gate)

| Candidate source split | Permitted use | Prohibited use |
| --- | --- | --- |
| `eval/train` (2,000 rows) | Candidate generation, post-batch quality/judge processing, and—only after its candidate passes all gates—future SFT target construction. | Prompt examples or few-shot data unless separately approved and recorded. |
| `eval/validation` (400 rows) | Evaluation-only candidate generation, format/grounding evaluation, and aggregate-only reporting. | SFT, training-data construction, prompt examples, few-shot data, candidate selection, base-model selection, checkpoint selection, or any other training decision. |

The candidate manifest must label every record with this split before requests
are built.  The SFT builder must fail closed unless the input manifest asserts
`source_split == "train"` for every selected target.  A validation candidate
must never be relabelled, copied, or joined into the train candidate set.

## Immutable per-example contract

The restricted, in-memory training example is the following conceptual
structure.  Field names are normative; it is intentionally not a serializable
example in this repository.

```text
example_key: opaque, run-local key
sentence_id_contract: "mal-sentence-id-v1"
sentences: ordered [{sentence_id: positive integer, text: restricted string}]
encoder_score_context:
  contract_version: "mal-encoder-score-input-v1"
  encoder_run_id: immutable scoring run identifier
  encoder_model_state_sha256: immutable frozen encoder state digest
  encoder_config_sha256: immutable scoring configuration digest
  content: finite decimal in [1, 5]
  organization: finite decimal in [1, 5]
  expression: finite decimal in [1, 5]
target:
  schema_version: "rationale-v3-sentence-id"
  content|organization|expression:
    evidence_sentence_ids: one or two sentence IDs
    diagnosis: non-empty Korean feedback
    next_step: non-empty Korean actionable feedback
```

`mal-sentence-id-v1` assigns `1..N` after splitting the canonical essay in
order on `#@문장구분#` or on whitespace immediately following `.`, `!`, or
`?`, then trimming and dropping empty pieces.  The same versioned splitter is
used for candidate generation, candidate validation, SFT construction, and
evaluation.  IDs are local to the example and are never document identifiers.

The `encoder_score_context` is a separate, frozen model input.  It contains
exactly `content`, `organization`, and `expression`, in that order.  Its
provenance fields must bind it to one frozen encoder state and configuration;
scores must not be recomputed, calibrated, averaged, or replaced while an SFT
dataset is being built.  The target is explanation-only: it contains no score
field, no score echo, no total, rank, or label, and no `average` field.  The
SFT loss is computed only on the target feedback completion, never on the
separate encoder-score context.

## Candidate admission gates

For each candidate, the restricted validator must require all of the following
before it can be considered for SFT:

1. the exact `rationale-v3-sentence-id` schema with precisely the three
   analytic axes and no additional properties;
2. one or two integer evidence IDs per axis, each within the example's
   `1..N` sentence range;
3. non-empty `diagnosis` and `next_step` for every axis;
4. the matching train split, request/input hash, response/candidate mapping,
   and candidate number; and
5. an accepted post-batch judge decision recorded in the restricted manifest.

Schema or grounding failures, API failures, missing records, abstentions, and
judge rejections are excluded rather than repaired by inventing text.  Retry
lineage must preserve the original request identity and record only the
missing/failed candidate; it must not duplicate a successfully submitted
request or candidate.

The SFT builder must emit aggregate counts and hashes only.  It must reject a
dataset if it finds validation membership, an unpinned encoder provenance,
duplicate `(example_key, candidate_number)` pairs, score-like target fields,
or an `average` field anywhere in the score context or target.

## Selected explanation-model configuration

The chosen base is a **fresh Qwen2.5-7B-Instruct** snapshot at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, with the tokenizer pinned to the
same revision.  It is initialized from the base snapshot, not from either
existing direct-score or human-feedback adapter: those adapters were trained
to emit score-bearing formats and would violate this explanation-only
contract.

If a later SFT is explicitly authorized, its fixed model-side configuration is
TRL `SFTTrainer` conversational prompt-completion training with
`completion_only_loss=True`, PEFT LoRA (`r=32`, `alpha=64`, `dropout=0.05`),
seed `2026`, BF16, maximum sequence length `4096`, microbatch `1`, and
gradient accumulation `16` on the permitted GPUs 0--3 (global effective batch
64).  Its assistant completion is only the `target` object above.  No score
token is an assistant target.  Any schedule, train count, or checkpoint rule
requires a later approved SFT protocol; this contract does not launch one.

## Model-selection evidence and decision boundary

This is a compatibility- and provenance-based base-model decision, not a
claim that an explanation model has been evaluated.  The local
source-development selection record for this exact Qwen revision selected a
direct-score checkpoint at step 600 using source-dev macro MAE
`0.4848736584349276`; the corresponding human-feedback score format was
worse (`0.5639555255171718`).  The standard stack also documents maintained
TRL prompt-completion training and the stable four-GPU batch/accumulation
contract.  These are aggregate-only historical evidence of the base/runtime,
not rationale quality evidence and not use of current validation candidates.

The future explanation model must be evaluated only after batch results exist
and the judge protocol is ready.  It cannot be promoted, compared, or selected
using any `eval/validation` candidate.  No model is selected for DPO or GRPO;
neither objective is part of this protocol.

Evidence locations (all existing operational artifacts are ignored):

- `outputs/standard-runs/matrix-4gpu-20260717-rerun2-decoder-direct-selection/selected_checkpoint.json`
- `outputs/standard-runs/matrix-4gpu-20260717-cont5-decoder-human-feedback-selection/selected_checkpoint.json`
- `docs/standard_decoder_stack.md`
- `docs/standard_experiment_matrix.md`
