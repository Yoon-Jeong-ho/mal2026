# Qwen3.6 native-FP8 vLLM fast-path (v1)

This is a separate runtime-validation lane.  It neither amends nor replays the
stopped v5.2 GGUF lineage, whose state remains `TAXONOMIZE`.  It inherits its
train-only request builder, prompt, rubric, sampling, semantic gates, candidate
artifact, validation isolation, and selection prohibition without modification.

The checkpoint is precisely `Qwen/Qwen3.6-35B-A3B-FP8`.  The official
Hugging Face CLI downloads it resumably to the ignored
`outputs/model-cache/Qwen--Qwen3.6-35B-A3B-FP8-native-fp8-v1` path.  Before a
server may start, the download manifest records the resolved immutable revision,
all repository filenames, per-file byte counts and SHA-256 values, the observed
total, and the published all-file total of 37,493,015,668 bytes.  A mismatch
fails closed.  Credentials are never printed or recorded.

The executable sequence is fixed:

1. GPU0 only: at most 20 fixed synthetic OpenAI-compatible JSON-schema calls,
   `enable_thinking=false`, `max_model_len=4096`, and `max_tokens=192`.
2. On the same fixed controls only, compare native vLLM with an isolated GGUF
   runtime.  The only reported comparison fields are schema/transport counts,
   no-thinking placement, latency, and throughput aggregates.  Verdicts are
   not compared and this stage cannot select data or a runtime.
3. Only if native vLLM has zero synthetic schema/transport failures and the
   synthetic contract does not regress, start four independent TP=1 workers on
   GPUs 0, 1, 2, and 3.  Tensor parallelism is never used.  Each worker must
   independently pass the same synthetic controls before real data is opened.
4. Run no more than three train essays through the inherited v5.2 semantic
   gates.  Any semantic failure stops at `TAXONOMIZE` with an aggregate-only
   record.  A pass permits only the authorized 2,000 train essays, three
   isolated candidates, and five repeats.  Validation remains unopened and
   evaluation-only.

Every transition writes the machine-readable fields specified by
`docs/omx_codex_research_orchestration.md`.  Raw prompts, outputs, candidate
text, writing, identifiers, and server logs stay in ignored runtime roots.

The initial GPU0 server preflight may use the single documented runtime repair
`--gdn-prefill-backend triton` when the default FlashInfer GDN JIT cannot start
because the already-provisioned environment lacks `ninja`.  This is a
one-variable runtime change, recorded through `TAXONOMIZE`, `MINIMAL_PATCH`,
and `REPLAY`; it neither installs a dependency nor changes any request or
semantic gate.
