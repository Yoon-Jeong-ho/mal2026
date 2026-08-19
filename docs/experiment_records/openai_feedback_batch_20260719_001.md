# OpenAI synthetic-feedback Batch run: 20260719-001

**Status:** Batch output has been downloaded and strictly validated.  The
reasoning-remediation synthetic and wire-smoke gates passed with the pinned
runtime; the approved train-only pairwise judge is running durably on GPU 0.
No selection artifact, SFT, DPO, or GRPO has run.

## Immutable run record

- Run ID: `openai-rationale-terra-full-20260719-001`
- Git SHA at preparation: `86902f1e3a077b1178d1297a1dcccf10e929453d`
- Model: `gpt-5.6-terra`
- Candidate schema: `rationale-v3-sentence-id`
- Command: `.venv-standard/bin/python scripts/generate_openai_rationales.py prepare --run-id openai-rationale-terra-full-20260719-001 --model gpt-5.6-terra`, followed by `submit` and `poll` with the same run ID/model.
- Population: exactly 2,000 train plus 400 validation essays, three candidates
  per essay (7,200 requests).
- Request SHA-256: `dac08deb2ce8ea1752cfdb37ae56e38ac31e994a83ccc6598e71484efc7911a9`
- Source-map SHA-256: `3c01bcb2ff58ce0a8caae056a1bcdcc81e43e36d64b6d63166481d580d28d973`
- Final provider counts: exactly 7,200 completed, 0 failed, and 7,200 total.
  Provider file IDs remain only in the ignored restricted manifest.

The script is fail-closed to this model, the two exact splits, and three
candidates.  It stores raw requests, source mapping, raw responses, candidate
feedback, provider errors, status, and checksums only under
`data/processed/restricted/openai_rationale_batches/<run-id>/`.  Submission
uses a deterministic provider idempotency key and a persisted submit intent;
a repeated `submit` reports the existing batch rather than creating another.
If a completed Batch has invalid, failed, or missing rows, the validator writes
only those rows to the restricted error set and permits one separately recorded
retry Batch.  No retry was submitted in this run because the validated error
and missing counts are zero.

## Download and strict validation (2026-07-19 KST)

The existing resumable `download` command retrieved the completed output into
the ignored restricted root and performed its fail-closed schema and
sentence-ID validation.  No provider error artifact existed because the
provider reported zero failed requests.  An aggregate-only second pass
revalidated each response and stored the result only as
`validation_aggregate.json` under that restricted run directory.

- Raw output records: 7,200; source mappings: 7,200; accepted candidates:
  7,200; rejected: 0; missing: 0.
- Aggregate audit also found 0 duplicate or unknown mappings/records and 0
  schema-or-grounding failures in the raw or accepted candidates.
- Raw-output SHA-256:
  `d74bb084f58006679c7880cf73337f5d88e04a4d0631fa752bf3a30e9596e7c7`
- Accepted-candidate SHA-256:
  `4ef414dd35b831092fcea24c7770a5f18fbe0df1d4e6aa74d55613b8cad71e2e`
- Error-set SHA-256 (empty file):
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

The candidate rows remain generated-and-stored evaluation-only until their
separate permitted use: validation rows are excluded from judge calls,
selection, SFT, prompt examples, and model selection; no candidate content was
printed or tracked.

## Pilot gate

A live one-row `gpt-5.6-terra` pilot completed on 2026-07-19 KST.  Its strict
JSON structure and all sentence-ID range/shape checks passed.  Aggregate
provider usage was 1,337 input tokens, 357 output tokens, and 1,694 total
tokens.  The raw provider response is restricted and is not reproduced here.
A previous restricted pilot is retained as a negative/stale schema result; it
is not evidence for promotion.

## Isolation and next gate

The generated validation rows are evaluation-only.  They are prohibited from
SFT, prompt examples, judge/model selection, and any training decision.  The
future train-only SFT admission contract is
`docs/synthetic_feedback_sft_contract_v1.md`; it retains separate frozen
content, organization, and expression encoder inputs and prohibits explanation
scores and averages.

`configs/qwen36_gguf_judge.v1.json` and
`docs/qwen36_gguf_judge_protocol.md` fix the future local judge protocol:
sentence-ID hard gates, frozen score conditioning, blinded A/B order swaps,
fixed sampling, refusal/abstention, and aggregate-only reporting.  No candidate
has been judged.

## Judge runtime preflight (blocked)

The configured runtime is the Qwen3.6-35B-A3B Q4_K_M GGUF pinned by
`configs/qwen36_gguf_judge.v1.json` (expected SHA-256
`b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`) and
llama.cpp revision `571d0d540df04f25298d0e159e520d9fc62ed121`, CUDA build
flag `GGML_CUDA=ON`, architecture `90`.  The GGUF had already completed its
documented hash verification before this continuation.

An independent bounded runtime-fallback review confirmed that CMake and Ninja
are absent, while CUDA 13.0, GCC/G++, and Make alone cannot satisfy the pinned
CMake build.  It found no nonrestricted compatible local `llama-server`,
`llama-cli`, or existing build.  The protocol forbids an ad-hoc dependency
installation, so no clone, build, download, system modification, server, or
smoke request was attempted.  Consequently there is no `llama-server --version`
evidence and the GPU-0-only smoke gate did not pass.

No runtime, judge, selection, or training GPU workload ran.  GPUs 4--7 were not
used or exposed to a workload.  **Recorded deviation:** an initial broad
status-only `nvidia-smi` inventory included GPUs 4--7 before the explicit
exclusion was enforced; it made no state change and no later command queried or
used those GPUs.  A separately approved environment remedy that already
provides CMake is required before a pinned local build and the documented
GPU-0-only synthetic smoke can proceed.

## Pre-SFT boundary

The approved train-only blinded pairwise judge and restricted SFT
candidate-selection construction are not eligible: their necessary GPU-0 smoke
gate is blocked.  This run stops before those stages and before all training.

## Official release preflight continuation (2026-07-19 KST)

The approved project-local runtime fallback was checked against the official
`ggml-org/llama.cpp` release matching the pinned configuration.  The GitHub
release is [`b10068`](https://github.com/ggml-org/llama.cpp/releases/tag/b10068),
published 2026-07-18T13:36:42Z and targeting pinned commit
`571d0d540df04f25298d0e159e520d9fc62ed121`.  The downloaded official release
metadata has SHA-256
`b07d449d3fe7b2f869ac845f53486f736f7a1631be24e57a8ed1ece6446ba144`.

Its CUDA assets are Windows x64 only (`cudart-llama-bin-win-cuda-12.4-x64.zip`,
`cudart-llama-bin-win-cuda-13.3-x64.zip`,
`llama-b10068-bin-win-cuda-12.4-x64.zip`, and
`llama-b10068-bin-win-cuda-13.3-x64.zip`).  It has **zero** Linux/Ubuntu CUDA
assets.  The other Linux packages are CPU, Vulkan, ROCm, SYCL, or OpenVINO and
are not compatible substitutes for the required CUDA judge.  Therefore no
binary was selected, downloaded, or extracted; an asset checksum is not
applicable.  The official API asset provenance (including its published digest
fields), release response, and aggregate preflight manifest are stored only in
the ignored restricted runtime-preflight directory.  The aggregate manifest
SHA-256 is `affcc98ad2f18e8f1b0d266f4765b6bfa7cc070014b586b06e2bdbdaee34aa82`.

A repository-local, metadata-only search found the configured GGUF in the
ignored project-local model cache.  Its observed size was `22,285,080,192`
bytes and its SHA-256 was
`b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`, matching
the configuration exactly.  The judge script's synthetic-only static
wire-contract check passed: it sends `POST /v1/chat/completions` with
`response_format` `json_object` plus the schema and fails closed on malformed
responses.  The pinned official server README documents that exact endpoint
and schema form.  No GPU was queried or used, no server was started, and no
student or candidate text was printed or judged.

**Gate result:** blocked before the GPU-0 synthetic smoke and train-only
pairwise judging, because the official pinned release has no compatible Linux
CUDA prebuilt runtime.  No SFT, DPO, GRPO, or other training was started.

## Approved project-local CUDA source-build continuation (2026-07-19 KST)

**Status:** stopped at the required synthetic GPU-0 smoke gate; the train-only
pairwise judge, selection construction, SFT, DPO, and GRPO were not started.

The explicitly approved isolated runtime was created only beneath the ignored
`outputs/runtime-cache/qwen36-judge-pre-sft-20260719-001/` root.  Its CMake
3.31.6 wheel SHA-256 was
`1c8b05df0602365da91ee6a3336fe57525b137706c4ab5675498f662ae1dbcec`; its
Ninja 1.13.0 wheel SHA-256 was
`fb46acf6b93b8dd0322adc3a4945452a4e774b75b91293bafcc7b7f8e6517dfa`.
Both came from official PyPI binary wheels and were installed offline into the
project-local environment.  CUDA was 13.0.88 and the host compiler was GCC
13.3.0.

The source clone observed exactly pinned llama.cpp commit
`571d0d540df04f25298d0e159e520d9fc62ed121` (release `b10068`) with a clean
source tree.  The CUDA/Ninja build completed with `GGML_CUDA=ON`, CUDA
architecture `90`, and `LLAMA_CURL=OFF`.  `llama-server --version` reported
version 10068 at that commit.  The configured GGUF again matched both its
expected byte count (22,285,080,192) and SHA-256
`b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`.

Immediately before launch, explicitly targeted physical GPU 0 was idle.  The
server ran only with `CUDA_VISIBLE_DEVICES=0`, all layers offloaded, one request
slot, and a bounded context.  The synthetic non-student JSON request reached
the OpenAI-compatible endpoint successfully, but the returned assistant
`content` was empty (length 0) while a separate reasoning field was present.
Consequently the required JSON object could not be parsed from `content`, so
the synthetic smoke failed.  Only aggregate field names, lengths, status, and
file hashes were recorded; no generated response text was printed or tracked.
The owned server session was stopped.  Per the fixed stop rule, no judge-script
wire smoke, candidate judging, selection, or training was attempted.

## Reasoning-remediation continuation (2026-07-19 KST)

The pinned llama.cpp b10068 source and server documentation were inspected
without changing the runtime.  They support both the server switch
`--reasoning off` and the OpenAI-compatible request field
`chat_template_kwargs.enable_thinking`; request processing forwards that
boolean to the active Jinja template.  The official Qwen3.6 model card
documents the same OpenAI-client `extra_body` form with
`{"chat_template_kwargs": {"enable_thinking": false}}`.  The request-level
switch was selected so the frozen judge request is explicit and independently
reproducible.

One justified synthetic variant was run on physical GPU 0 only, with an owned
localhost server that was stopped after the request.  It added
`chat_template_kwargs.enable_thinking=false` to the existing JSON-schema
request.  The aggregate result was: parseable JSON in `message.content`
(length 362), valid judge schema, no reasoning field, `finish_reason=stop`,
and 184 total tokens.  The ignored raw response SHA-256 was
`50c2138b772c4aa53b5fa82a74380f34b1a9fe953b24add658ba387a22eb82df`.

The judge request builder now obtains that option from
`configs/qwen36_gguf_judge.v1.json`; the resulting config and builder hashes
are respectively `a373c16816a3d2eeb403fa88be5cea91fd4d6d907e60e5b7635a9a36d4601254`
and `94cb5625c45e363baf3e57996e8668bc4d6d3c752813bf227d82105e42423ae5`.
A non-restricted request-contract test and Python compile check passed.  The
available environment has no `pytest` module, so no package installation was
performed.

A second owned GPU-0 localhost server then ran the actual judge script's
`response_json` transport against a synthetic request.  Its aggregate result
was parseable and schema-valid JSON in `content` (length 362), no reasoning
field, `finish_reason=stop`, and 186 total tokens; the ignored raw response
SHA-256 was
`1865f4d57a37c8709872ef08e87e7bdbcc9a6250f5837a7afceaba5d5103662e`.
The server was stopped before the production judge launch.  No synthetic
generated text was printed or included here.

After the successful gates, the train-only preparation accepted 6,000
validated train candidates, excluded all 1,200 validation candidates, excluded
zero invalid train candidates, and constructed 12,000 blinded order-swapped
requests.  The durable session `qwen36-judge-pre-sft-002-retry2` is executing
only on `CUDA_VISIBLE_DEVICES=0` against localhost port 18083.  Its prelaunch
GPU query addressed only GPU 0; it records the Git SHA, model and runtime
hashes, exact command, and aggregate completion manifest in the ignored judge
run directory.  Two short launcher bootstrap attempts failed before request
execution because of an incorrect relative repository root; their separate
ignored logs/status files are preserved.  The repaired retry is the only
production execution attempt.  Selection, SFT, DPO, and GRPO remain prohibited
until this judge finishes with an aggregate reconciled manifest.

**Recorded deviation:** before the GPU-0-only guard was applied in this
continuation, one status-only `nvidia-smi` command inadvertently enumerated all
physical devices, including 4--7.  It made no state change, launched no
workload, and printed no restricted data.  Every later GPU operation was
explicitly constrained to physical GPU 0; no GPU 4--7 was subsequently queried
or used.
