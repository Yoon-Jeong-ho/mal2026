# Judge-v2 decision memo: position-bias remediation

**Status:** preregistered train-only pilot; no SFT/DPO/GRPO, candidate
selection artifact, or full rerun is authorized by this memo.

## Decision

The v1 reconciler was conservative but not reliable enough for selection: the
reported aggregate run contained 1,371 swap agreements, 2,054 swap
disagreements, 2,554 ties, and 2,075 abstentions.  Judge-v2 therefore changes
the unit of evidence from one swapped pair to a small, factorially balanced
panel.  It will use only a deterministic sample of at most 128 **train**
essays.  Validation is neither loaded as a source row nor placed in a request,
control, prompt example, pilot report, or training input.

The pilot is explicitly diagnostic.  A passing pilot authorizes only writing
an aggregate pilot record and proposing a separate production decision; it
does not create selection data or authorize any training.

## Evidence and inference

* Shi et al.'s primary study defines the relevant diagnostics as repetition
  stability, position consistency, and preference fairness, and observes that
  position bias varies by judge and task ([paper](https://arxiv.org/abs/2406.07791)).
* The RLAIF primary paper operationalizes position bias as choosing the same
  *position* after candidate order is swapped, rather than the same underlying
  response ([appendix](https://arxiv.org/pdf/2309.00267)).
* The local runtime supports schema-constrained `response_format` requests
  ([llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)),
  and Qwen documents request-level `chat_template_kwargs.enable_thinking=false`
  for its OpenAI-compatible interface ([model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/main/README.md)).

**Inference for this project:** swapping A/B alone conflates a visible label
with first/second display position.  It also cannot measure stability for an
identical request.  A factorial presentation, repeated calls, and neutral
controls are required before pairwise outputs may be considered reliable.
Multiple calls to the fixed Qwen GGUF are independent *request lanes*, not
independent models; the pilot reports cross-lane agreement and makes no claim
of model-family independence.

## Frozen v2 pilot protocol

1. `prepare` admits only schema-valid, mechanically grounded candidates from
   `eval/train.jsonl` and a pre-existing restricted, split-scoped
   `candidates.train.jsonl` artifact with its train-only manifest.  It refuses
   the combined candidate file rather than deserializing and filtering
   validation candidates.  It takes the lowest SHA-256 ranks of train source keys
   under the fixed selection seed, capped at 128.  Keys, candidate content,
   and essay text are written only to ignored restricted request files; no
   aggregate output contains them.
2. Each unordered candidate pair receives two independently seeded rubric
   lanes.  Across all pairs, the six permutations of `content`, `organization`,
   and `expression` are assigned by a deterministic balanced schedule.
3. Within each lane, the two underlying candidates are evaluated in all four
   combinations of candidate label (`A`/`B`) and physical display position
   (first/second).  Each exact request is repeated once.  Thus label and
   physical position are separately balanced, rather than merely swapped.
4. A pointwise screen independently assesses each candidate twice with
   balanced rubric order and repeat calls.  Pairwise panel aggregation is
   diagnostic only: a lane is decisive only if every factorial cell and its
   repeat maps to the same underlying candidate; the two lanes must then agree.
5. Synthetic non-student controls are included: exact duplicate/identity
   pairs (expected `tie` or `abstain`) and deliberately invalid candidates
   (expected `abstain`).  They are never mixed with train data or used for
   selection.
6. All raw requests, responses, source mappings, and per-call outcomes remain
   below the ignored restricted root.  The tracked report contains aggregate
   counts, rates, checksums, provenance, and gate results only.

The planned load is 128 essays maximum, three pairs per essay, two rubric
lanes, four label/position cells, and two exact repeats (at most 6,144 real
pairwise calls), plus bounded pointwise and synthetic-control calls.  It is a
pilot, not a full rerun.  It runs on physical GPU 0 only with
`CUDA_VISIBLE_DEVICES=0`; GPUs 4--7 are never queried, exposed, or used.

## Preregistered hard gates

The configuration is the authority for exact thresholds.  The pilot fails
closed (and no selection artifact is constructed) if any of these are false:

| Gate | Threshold |
| --- | ---: |
| Valid pilot source isolation | exactly 0 validation rows loaded; 0 validation requests |
| Sample size | `1..128` train essays |
| Transport/schema failures | 0 |
| Pointwise exact-repeat stability | >= 0.98 |
| Pairwise exact-repeat stability | >= 0.98 |
| Factorial order consistency | >= 0.90 |
| Absolute label-win imbalance | <= 0.05 |
| Absolute first-position-win imbalance | <= 0.05 |
| Raw pairwise abstention | <= 0.30 |
| Two-lane panel consensus rate | >= 0.60 |
| Identity non-neutral choices | exactly 0 |
| Invalid-control abstention | >= 0.99 |

Rates with a zero denominator are failures.  Gates are not tuned after pilot
results.  Any protocol change requires a dated v3 amendment and a new run.

## Versioned implementation plan

| File | Role |
| --- | --- |
| `configs/qwen36_gguf_judge.v2.pilot.json` | immutable runtime pin, sampling, factorial schedule, control count, and gates |
| `scripts/judge_feedback_candidates_v2.py` | train-only preparation, request execution, aggregation, and fail-closed gate report; contains no selection command |
| `tests/test_qwen36_judge_v2_pilot.py` | synthetic-only contract and aggregation tests; no restricted inputs |
| `docs/experiment_records/qwen36_judge_v2_pilot_*.md` | aggregate-only observed result after execution |

Before execution, run the static tests and a GPU-0-only synthetic wire smoke.
The owned runner writes a localhost server attestation from its launched
`CUDA_VISIBLE_DEVICES=0` process; smoke and execution refuse an unattested or
remote endpoint.  Run the pilot only after its preflight gate passes.  Stop at
any failed hard gate and preserve the restricted artifacts and aggregate
failure record.
