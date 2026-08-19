# Qwen3.6 GGUF feedback-candidate judge (v1)

This document fixes the local LLM-as-judge contract for the feedback-candidate
stage.  It does not authorize candidate judging.  Candidate execution begins
only after a completed, downloaded, and schema-validated Batch artifact is
available in the ignored restricted-artifact root.

## Provenance and runtime

The checked-in configuration is `configs/qwen36_gguf_judge.v1.json`.  It pins
the GGUF repository revision, exact file name, byte count, SHA-256, license,
llama.cpp revision, CUDA build settings, and the GPU allowlist.  The source
build is necessary because the pinned llama.cpp release has no Linux CUDA
binary.  Downloaded model weights, cloned runtime sources, binaries, server
logs, requests, and per-record decisions are operational artifacts and must be
kept below an ignored `data/processed/` run directory.  Do not put them in Git.

The source build has a hard host precondition: CMake and a CUDA-capable C++
toolchain must already be available.  Do not install a build tool ad hoc.  If
the documented environment lacks CMake, record that as a blocked runtime
preflight and obtain a separately approved environment remedy before building.

Before use, download the one configured GGUF file from the configured immutable
repository revision, then verify both the byte count and `sha256sum` against
the configuration.  Clone llama.cpp at its configured immutable revision and
verify `git rev-parse HEAD` before the CUDA build.  Capture those observed
values, the CUDA/driver version, and `llama-server --version` in the ignored
run manifest.  A changed source revision, hash, license, or build option is a
new protocol version rather than an in-place change.

Only GPU 0 is eligible for the initial smoke test.  Verify via `nvidia-smi`
that it is idle immediately before launch.  Start with all model layers on GPU
and a single request slot; the Q4_K_M file is about 22.3 GB, so one H100 80 GB
is sufficient for the model and a bounded smoke context.  The launcher may add
GPUs 1--3 only after recording that each is idle and only when measured
throughput requires it.  GPUs 4--7 are forbidden, including discovery,
visibility masks, and tensor split settings.

The smoke request uses a synthetic, non-student Korean sentence list and the
judge JSON schema.  It must demonstrate: server starts on `CUDA_VISIBLE_DEVICES=0`,
the configured model loads, JSON is parseable, seed/temperature are accepted,
and no generated text is printed.  Store the raw smoke response only in the
ignored run directory and put only pass/fail, hashes, versions, GPU index, and
timing in an aggregate experiment record.

Every Chat Completions request must include
`"chat_template_kwargs": {"enable_thinking": false}` alongside the existing
schema-constrained `response_format`.  This is Qwen3.6's documented
OpenAI-compatible non-thinking switch, and pinned llama.cpp b10068 forwards
that template kwarg to `enable_thinking`.  It is required so the constrained
JSON answer is emitted in `choices[0].message.content`, rather than consuming
the output budget as a separate reasoning field.  Do not replace this hard
template switch with a prompt-only `/no_think` instruction.

## Candidate eligibility and isolation

1. Load only candidates whose Batch row and candidate schema both validated.
2. For every candidate, re-check that each feedback evidence sentence ID is an
   integer in the essay's numbered-sentence range.  A failure is a hard-gate
   failure, not a judge preference.
3. Supply the frozen three independent encoder inputs (`content`,
   `organization`, and `expression`) separately.  Never derive, expose, or
   train an explanation score or their average.
4. Feed the judge a numbered essay, the three frozen inputs, and two feedback
   candidates.  Candidate provenance IDs and candidate numbers are never shown
   to the judge.
5. Validation candidate rows are retained for evaluation-only storage.  They
   are excluded from judge calls, prompt examples, SFT data, and model
   selection.

All such inputs, raw model completions, candidate text, mappings, and
per-record verdicts are restricted.  They must never be printed, tracked, sent
to W&B, or included in a report.

## Pairwise judge schema and decision rule

Each train essay has the three unordered pairs `(1,2)`, `(1,3)`, and `(2,3)`.
Every pair is evaluated twice: once as blind `A`/`B` and once in the swapped
order.  The presentation order is derived deterministically from the fixed
seed and stable HMAC of the restricted record key; the result stores only the
restricted mapping.  The two calls use fixed `seed=2026`, `temperature=0`,
and `top_p=1`.

The required response object is:

```json
{
  "schema_version": "qwen36-gguf-judge-v1",
  "verdict": "A|B|tie|abstain",
  "hard_gates": {
    "A": {"score_conditioned": true, "sentence_id_grounded": true, "non_speculative": true},
    "B": {"score_conditioned": true, "sentence_id_grounded": true, "non_speculative": true}
  },
  "refusal_or_abstention_reason": "string-or-empty"
}
```

`abstain` is required for refusal, invalid/missing JSON, any hard-gate failure
that cannot be resolved mechanically, order-swap disagreement, or inability to
distinguish candidates.  A non-abstaining result is accepted only when both
orders select the same underlying candidate after unblinding.  A candidate that
fails a mechanical pre-gate loses only to a pre-gate-valid counterpart; two
failed candidates produce `abstain`.  No retry changes a result; retries are
limited to transport failures and retain the same prompt, seed, and order.

Selection is by deterministic pairwise win count over accepted train-only
comparisons.  Ties/abstentions remain in the denominator report and never
silently become wins.  No selection result can initiate SFT, DPO, or GRPO;
those require a separately approved immutable pre-SFT contract.

## Aggregate-only report

The tracked experiment record may contain only the model/runtime provenance,
input and output checksums, candidate/pair counts, hard-gate counts, valid
order-swap agreement rate, abstention/refusal/transport-failure counts, and
aggregate candidate win/tie counts.  It must contain no essay text, identifiers,
candidate feedback, individual verdicts, or explanations.
