# Native-FP8 vLLM dependency repair: 20260720-001

**Lineage:** `native-fp8-vllm-20260720-001`
**Initial state:** `TAXONOMIZE` after the bounded GPU0 default and Triton-GDN
server-start attempts both reached a FlashInfer JIT failure caused by missing
`ninja`.  No synthetic request or train/validation content was opened.

## Authorized one-variable repair

The only environment mutation is the project-local Python package
`ninja==1.13.0` in the existing ignored `.venv-standard` runtime.  No system
package, environment, vLLM, torch, FlashInfer, or SGLang change is permitted.

| Field | Recorded value |
| --- | --- |
| Interpreter | `.venv-standard/bin/python` (Python 3.12.3) |
| Package | `ninja==1.13.0` |
| Provenance | PyPI `ninja` distribution (`scikit-build/ninja-python-distributions`); installed by `pip`; package metadata lists the Ninja project homepage `http://ninja-build.org/` |
| Runtime lock | `requirements-standard.txt` pins `ninja==1.13.0` |
| Initial verification | Pass: `ninja` imports, resolves to `.venv-standard/bin/ninja`, reports `1.13.0.git.kitware.jobserver-pipe-1`, and `pip check` passes with `CUDA_VISIBLE_DEVICES=''` |

## Declared continuation gates

1. GPU0 `REPLAY`: unchanged native vLLM server start and exactly 20 synthetic
   schema/non-thinking controls.
2. Unchanged, contract-only GGUF synthetic comparison.
3. Graceful release of utilization-only GPUs 1--3 under their documented
   ownership protocol, then four independent TP=1 vLLM synthetic workers.
4. At most three train essays through the inherited semantic gates.
5. The already-authorized 2,000-essay, three-candidate, five-repeat train-only
   stage only after every preceding gate passes. Validation, selection, SFT,
   DPO, and GRPO remain out of scope.

## Outcome

GPU0 was idle at start and cleanly released afterward. The unchanged replay
loaded the native FP8 checkpoint but failed before `/health` and before any
synthetic request: its inherited server-process `PATH` did not include
`.venv-standard/bin`, so FlashInfer's first-use JIT raised `FileNotFoundError`
for `ninja`. This is a technical replay-gate failure, not a model, schema, or
semantic result. The aggregate-only failure evidence is
`outputs/native-fp8-vllm/native-fp8-vllm-20260720-001/gpu0-repair-runtime-failure.json`.

Accordingly the lane stopped at `TAXONOMIZE`; no GGUF comparison, GPU1--3
action, synthetic worker, train essay, full run, validation access, selection,
SFT, DPO, or GRPO was started. Runtime logs and restricted artifacts remain
under ignored `outputs/` and `data/processed/` roots.

## Append-only PATH repair replay (gate 001f)

**Exact patch:** added `export PATH="$ROOT/.venv-standard/bin:$PATH"` directly
after the computed `ROOT` in `scripts/run_qwen36_native_fp8_vllm_v1.sh`, before
any vLLM process can launch.  No other launcher argument, model, package,
protocol, request, schema, data, or GPU allocation was changed.

**No-GPU verification:** `bash -n scripts/run_qwen36_native_fp8_vllm_v1.sh`
passed.  Under exactly
`PATH=/dataset/aa007878/mal2026/.venv-standard/bin:$PATH`, a clean subprocess
resolved `ninja` to `/dataset/aa007878/mal2026/.venv-standard/bin/ninja` and
reported `1.13.0.git.kitware.jobserver-pipe-1`.

**Replay command:** `scripts/run_qwen36_native_fp8_vllm_v1.sh
native-fp8-vllm-20260720-001 gpu0-synthetic`.  GPU 0 only was used.  The server
reached `/health` (HTTP 200) and all 20 synthetic requests returned HTTP 200;
no real essay or restricted data was opened.  The recovered FlashInfer path is
confirmed by the server log's `Using FlashInfer for top-p & top-k sampling`.

**Outcome:** fail closed at `TAXONOMIZE`.  The immutable aggregate reports 20
calls, 0 schema-valid calls, 15 `schema_shape` failures, 5 `envelope_finish`
failures, no-thinking placement passed, mean latency 2.308132 seconds, maximum
latency 17.123989 seconds, and throughput 0.433251 requests/second.  This is a
synthetic response-envelope/schema taxonomy after the PATH repair, not a Ninja
or server-health failure.  The gate JSON is
`outputs/native-fp8-vllm/native-fp8-vllm-20260720-001/gates/001f-path-repair-replay-taxonomize.json`;
the aggregate and restricted server log are respectively
`outputs/native-fp8-vllm/native-fp8-vllm-20260720-001/gpu0-synthetic/aggregate.json`
and `outputs/native-fp8-vllm/native-fp8-vllm-20260720-001/gpu0-replay-gpu0.log`.

No second prompt or code change was made.  Per the bounded failure rule, all
later gates remain unlaunched.
