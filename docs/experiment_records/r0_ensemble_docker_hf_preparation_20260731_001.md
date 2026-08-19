# R0 prediction ensemble Docker/Hugging Face preparation: 20260731-001

**Status:** complete. The runtime bundle, public custom-weight repository, CPU
contract tests, GPU0 synthetic inference, model-bearing Docker image, no-argument
container/API smoke, Docker Hub push, and anonymous pull verification all
completed. The initial storage/registry boundary described below was resolved
by explicit user authorization on the same task card.

## Authorization and boundaries

- The user explicitly requested preparation of the strongest recorded model for
  Docker submission and authorized use of the configured Hugging Face account.
- GPU scope was the default MAL2026 scope; only physical GPU 0 was used for the
  two model-loading smoke paths. No process was displaced or terminated. GPU 0
  was an `NVIDIA H100 80GB HBM3` with 81,559 MiB total memory and returned to
  0 MiB used after the smoke.
- No training or retuning was performed. No restricted train/validation row,
  writing text, identifier, row-level score, prediction, or generated row output
  was read or uploaded. Synthetic task text only was used for runtime checks.
- No package was installed and no environment was created. The existing
  `.venv-standard` and the pinned upstream vLLM container were used.
- Repository Git SHA at preparation time:
  `a96f029b1ea640878607d4eb7bd817e6099334c1`. The working tree already contained
  unrelated user changes; they were not modified or reverted.

## Canonical submission contract

The supplied seven-page Docker rule PDF has SHA-256
`40d35b56af956f76adc52acff19ccf8c30425ebdd5208a3fb7e298c2ad3be15e`.
The supplied nine-page task specification has SHA-256
`125896cdeb0862816b41df4e02e3972c85b1e36ee999b3fd3644e2f8f5bf5080`;
it permits sequential models/ensembles but requires the complete inference path
to fit on one L40S 48 GB GPU (14B or smaller recommended).
The implemented boundary follows its required behavior:

- no-argument container startup with a foreground main process;
- HTTP bind on `0.0.0.0:8000`;
- `GET /health`, `GET /v1/models`, and
  `POST /v1/chat/completions`;
- OpenAI-style `model`, `messages`, `max_tokens`, `temperature`, `top_p`,
  `seed`, and `stop` request handling and a nonblank
  `choices[0].message.content` response;
- one visible GPU, an internal fixed model name, and offline startup with all
  model weights embedded in the image.

The public writing-output contract is the strict three-axis integer score plus
nonblank Korean rationale JSON. The local `evaluation.txt` used for contract
tests has SHA-256
`1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145`.

## Frozen model identity and development evidence

The selected score candidate is the historical R0 P1--4 prediction ensemble:

| Component | Frozen identity |
| --- | --- |
| Score base | `Qwen/Qwen3-Embedding-8B` @ `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| Rationale base | `skt/A.X-4.0-Light` @ `ba21c20ea1b31ded1ec3e2fb432335077dc4be98` |
| Epoch 1 raw trainable state | `fd317dd017133f2d6120d857a2d7f7d6caebaf547590cead155ca0ffda1a9c7c` |
| Epoch 2 raw trainable state | `dd47ac7ea93dce46da5f9e9b44cf86039331a0973f5f7f21cee46b2f4b85b57d` |
| Epoch 3 raw trainable state | `96f5897a27f600b52156dd992e41fa7aefb0d7c6948d3fbf71b0c002939d469d` |
| Epoch 4 raw trainable state | `30b8c677973a2b0df8052cf67a2717ba37b6f64180bc4e48be1572c3e4b6c592` |
| Historical score-blind rationale adapter | `39c68bb5c98da25eaa466434ba1c6d4a47bedcec580c991f00335627382a3a73` |
| Final emitted-score-conditioned DPO adapter | `887abf9d1bf07693251a17b7a0fb655fe8203fa6945e9c178a38bdc538ded826` |

The request path is frozen as: score-blind historical rationale generation,
four independent Qwen LoRA/head forward passes, uniform continuous prediction
mean, clipping to `[1, 5]`, half-up integer rounding, then final rationale-only
DPO generation conditioned on those emitted integer scores. Human/reference
scores are never supplied to either rationale pass.

Previously exposed 400-row validation evidence is retained only as development
evidence:

| Metric | Value |
| --- | ---: |
| Continuous macro RMSE | 0.5582937519204271 |
| Continuous macro Spearman | 0.6441959864775355 |
| Integer macro RMSE | 0.6158981882311673 |
| Integer macro Spearman | 0.5681974394968795 |

This is the strongest recorded continuous-score candidate, but it predates the
exact `evaluation.txt` prompt re-audit and its validation split was previously
exposed. It is not an untouched hidden-test estimate. The final DPO rationale
stage was added for participant-output quality; the RMSE above measures the
frozen score ensemble, not a newly measured end-to-end hidden test.

## Bundle and reproducibility

The ignored offline bundle is `deployment/runtime_bundle_r0/`. Its apparent
file size is 31,047,156,883 bytes (35 GiB allocated on the project filesystem).
`bundle_complete.json` has SHA-256
`58c029d014b80ac9530a1e6e2535a235bb8e90f45b1713246b0919c11045e80f`.
All 506 exported adapter/head tensors in each of epochs 1--4 were compared with
their raw checkpoint tensors and matched exactly.

The bundle was prepared with:

```bash
PYTHONPATH=src:. .venv-standard/bin/python \
  scripts/prepare_r0_ensemble_submission_bundle.py --stage
```

The runtime manifest is `deployment/runtime_manifest.r0.template.json`
(SHA-256
`a44760a93d7f84d31aff2a6531cd5895044db64d394a373ffd154273c25b641d`).
The Dockerfile is SHA-256
`0fe2c0134625ccae9a677c5650b61ebff95cb67d47b13baf39b6cc0578b7e9bb`.

The immutable Linux/amd64 base is:

```text
vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268
```

Its local image ID equals the digest and its logical size is 8,818,367,963
bytes. The verified base runtime contains vLLM 0.25.1, Python 3.12, PyTorch
2.11.0+cu130, Transformers 5.13.1, Accelerate 1.14.0, Safetensors 0.8.0,
FastAPI 0.136.3, and Uvicorn 0.51.0. PEFT was absent, so the already-installed
PEFT 0.19.1 Python package source was vendored into the offline model bundle;
no installation occurs at build or startup.

## Verification evidence and negative results

CPU contract tests and syntax checks:

```bash
PYTHONPATH=deployment/src:src .venv-standard/bin/python -m unittest -v \
  tests.test_submission_server
PYTHONPATH=deployment/src:src .venv-standard/bin/python -m py_compile \
  deployment/src/mal2026_submission/*.py \
  scripts/prepare_r0_ensemble_submission_bundle.py \
  scripts/publish_r0_ensemble_hf.py
```

The focused submission suite passed 9/9 tests. A final combined regression run
of `tests.test_submission_server`, `tests.test_evaluation_prompt_matrix`, and
`tests.test_evaluation_prompt_score_encoder` passed 19/19 tests; the same
`py_compile` gate also passed. The first GPU0 model-load attempt reached a
FlashInfer sampler JIT path and failed because the immutable base lacks
`ninja`. This negative result was preserved conceptually and recovered without
installing a package by setting `VLLM_USE_FLASHINFER_SAMPLER=0`. The next
attempt exposed an unrelated missing return from `_rationale_schema`; the
return was restored and covered by the focused tests.

The final audit also found that the initial serving implementation rejected a
score-encoder input above 2,048 tokens, whereas the frozen R0 train/evaluation
path used right truncation at 2,048. Serving was changed to the same
`truncation=True, max_length=2048` contract and a focused unit test was added.
The first edit accidentally moved `_score_input`'s return below the helper and
produced one 18-pass/1-fail CPU run; that negative result was immediately
repaired, and the final 19/19 run above is the post-repair evidence. The GPU
smoke's synthetic input was already below 2,048 tokens, so this correction does
not change the recorded smoke computation.

The repaired direct GPU0 synthetic pipeline smoke completed:

- model load: 68.445 seconds;
- inference: 28.336 seconds;
- strict integer scores: 3 / 3 / 4 for content / organization / expression;
- completion: 216 tokens;
- three nonblank rationale lengths: 182 / 156 / 138 characters.

A second smoke ran the unmodified pinned base container with the prepared code
and model bundle bind-mounted at their final in-image paths. This tests the
base image's real dependencies and the no-argument submission server entry
path without duplicating the 35 GiB bundle in Docker storage. It reached ready
health in 98 seconds; `/v1/models` returned
`mal2026-r0-ensemble-v1`; the synthetic `/v1/chat/completions` request completed
in 30.842 seconds with `finish_reason=stop`, scores 3 / 3 / 4, nonblank
rationales, and usage 1,553 prompt / 234 completion / 1,787 total tokens. Peak
resident allocation observed after inference was 52,079 MiB on the 81,559 MiB
H100. The container was stopped and removed cleanly.

The manifest uses `gpu_memory_utilization=0.43`; vLLM's cache reservation scales
with visible GPU capacity. This supports, but does not replace, the mandatory
final smoke on the target L40S-class evaluation GPU. No L40S was available in
the authorized GPU 0--3 scope during this preparation.

### Conservative L40S memory-budget proxy

After the user requested a 48 GB fit check, the full synthetic pipeline was
replayed on physical GPU 0 with only the vLLM memory fraction changed in memory
from `0.43` to `0.24`; the staged manifest was not modified. On the 81,559 MiB
H100 this gives vLLM a 19,574.16 MiB budget, slightly less than the 19,809.24
MiB obtained by the production `0.43` fraction on a representative 46,068 MiB
L40S. This is therefore a conservative capacity proxy rather than an L40S
performance benchmark.

The complete two-rationale/four-score synthetic request passed. Model load took
95.701 seconds, inference took 28.032 seconds, and 323 samples at 250 ms
intervals observed a peak of 36,677 MiB. That is 9,391 MiB below the 46,068 MiB
reference L40S capacity. GPU 0 was released to 0 MiB afterward. The aggregate,
synthetic-only report is
`outputs/r0-l40s-memory-proxy/r0-l40s-memory-proxy-20260731-001/result.json`;
it contains no restricted row or generated rationale text.

This materially supports single-L40S memory fit with roughly 9.2 GiB measured
headroom under an equal-or-smaller vLLM budget. It cannot verify L40S-specific
kernel compatibility or latency because no L40S is available locally; those
remain final evaluation-environment uncertainties.

## Public Hugging Face artifact

The upload command was:

```bash
PYTHONPATH=src:. .venv-standard/bin/python \
  scripts/publish_r0_ensemble_hf.py --publish
```

The public repository is
[`yoonLM/mal2026-r0-ensemble-v1`](https://huggingface.co/yoonLM/mal2026-r0-ensemble-v1)
at immutable revision
`b5029e8c1e78a5a2533f5e7c62c26fc0dd1041be`. An unauthenticated
`HfApi(token=False).model_info(..., files_metadata=True)` check confirmed
`private=False`, the same revision, and all 22 expected files. The uploaded
custom artifacts have an apparent size of 1,344,861,907 bytes. The artifact
manifest SHA-256 is
`0b87dc138ee3ae4bab4a3f3f52840ab192555eac7cc6b06913ad425c17e51077`.
The repository excludes base weights, competition rows, text, identifiers,
predictions, credentials, and optimizer/run artifacts; it points to the two
pinned public base revisions.

## Initial Docker boundary (subsequently resolved)

The root filesystem had 66 GiB free after pulling the base image, while Docker
reported 88.98 GB of inactive images and the R0 bundle occupies 35 GiB on the
project filesystem. A classic full build can temporarily require both a bundle
content blob and its unpacked snapshot, leaving an unsafe root-filesystem
margin or exhausting it. Therefore the final `docker build` was deliberately
not started. No inactive image was pruned and Docker's data root was not moved.

Once a safe builder/data root and a registry namespace are authorized, the
frozen build command is:

```bash
docker build \
  --build-arg 'BASE_IMAGE=vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268' \
  --build-arg 'RUNTIME_BUNDLE=runtime_bundle_r0' \
  -t '<registry>/<namespace>/mal2026-r0-ensemble:20260731-001' \
  deployment
```

Required continuation gates are: materialize the image on the safe Docker data
root, run it with no arguments on one target L40S-class GPU, repeat the three
endpoint and synthetic long-input checks, record the immutable image digest,
push that exact digest to the authorized registry, and verify an unauthenticated
`docker pull` if the evaluator expects a public repository.

## Authorized Docker continuation and final outcome

The user subsequently authorized deletion of exactly these two inactive local
images and supplied the public Docker Hub repository
`yoonjh1/mal2026-r0-ensemble`:

| Deleted reference | Local image ID | Logical inspect size |
| --- | --- | ---: |
| `upstage/vllm-solar-open2:latest` | `sha256:b34b7dd42d986bafb8dcfb285758de2fbfd55e7cf28a1b5ba6b858e6b2b7c8c3` | 30,264,870,473 bytes |
| `nvidia/cuda:12.8.1-base-ubuntu24.04` | `sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2` | 101,013,058 bytes |

No container or volume existed at deletion time. Docker reported zero images
after the two exact removals and root free space rose to 149 GiB. The Docker
daemon is the current user's rootless user service, but its data root was not
moved: `/dataset` is NFSv3 while the daemon uses `overlayfs`, so moving the
overlay store there would introduce an unsupported/risky backing-filesystem
combination. The immutable vLLM base was then pulled again and resolved to the
expected amd64 image ID
`sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268`.

### Model-bearing image build

The exact build was:

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg 'BASE_IMAGE=vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268' \
  --build-arg 'RUNTIME_BUNDLE=runtime_bundle_r0' \
  -t 'docker.io/yoonjh1/mal2026-r0-ensemble:20260731-001' \
  deployment
```

It completed in 1,106 seconds. BuildKit emitted the expected warning that the
deliberately mandatory `BASE_IMAGE` argument has no valid default; the supplied
immutable digest resolved and every build step completed. The final image has:

- OCI index digest:
  `sha256:bbba8b2bb66701a04bb0165c8f36fad5856cb72e3c318859dd5335a1f1d9af06`;
- Linux/amd64 image manifest:
  `sha256:e85872b8fa740c3ac9a139730c0afb6c6e2f2dabdc4bae2119079f2ec6de4b68`;
- config digest:
  `sha256:97fee171fe31c816ec58a89d467e1fa84944a49daca71ce9efd2ae4d0983b8bd`;
- fixed entrypoint:
  `["python3","-m","mal2026_submission.server"]`;
- Docker Hub compressed full size: 33,364,220,556 bytes.

The ignored build log/result are under
`outputs/r0-docker-build/r0-docker-build-20260731-001/`.

### Built-image no-argument smoke

The image itself—not a bind-mounted bundle—was started without any command
after the image reference, exposing only physical GPU 0 and host port 8000 for
local probing. Docker inspection recorded `Path=python3` and args
`["-m","mal2026_submission.server"]`. Health became ready in 98 seconds,
`/v1/models` returned `mal2026-r0-ensemble-v1`, and both task requests returned
strict JSON with `finish_reason=stop`:

| Synthetic request | Latency | Contract evidence |
| --- | ---: | --- |
| Short | 23.374 s | scores 3/3/4; three nonblank rationales; 1,541 prompt and 243 completion tokens |
| Long | 16.849 s | 2,325 Qwen essay tokens, 1,288 A.X blind-prompt tokens; scores 3/3/4; three nonblank rationales; 11,026 aggregate prompt and 218 completion tokens |

The long request deliberately exceeded the frozen 2,048-token score-encoder
limit while remaining below the 4,096-token rationale context, directly
exercising the audited right-truncation path. Peak H100 memory was 52,577 MiB;
the owned container was removed and GPU 0 returned to 0 MiB at smoke cleanup.

The first aggregate mistakenly recorded the length of the A.X tokenizer's
`BatchEncoding` mapping (two keys) instead of `input_ids`. The original
`result.json` was preserved with SHA-256
`d3d7e9276b1712ce3c6af5aad813e2e474b4c9685c53f1e7d11140d3360d974a`;
`result-corrected.json` records the verified 1,288 input tokens and has SHA-256
`658ffc9d1709d1fa6821ddf0cfe63c314592f02b8799516e89e1621fedeaf7d5`.
No restricted text is present in either aggregate.

### Docker Hub publication and anonymous verification

The prevalidated image was pushed with:

```bash
docker push --quiet \
  docker.io/yoonjh1/mal2026-r0-ensemble:20260731-001
```

The push completed in 2,582 seconds. An anonymous Docker Registry pull token
then resolved the public tag to the same OCI index digest. The Linux/amd64
manifest, config, and all 36 layer descriptors were checked; all 37 config/blob
HEAD requests returned HTTP 200. Docker Hub independently reported the same
index digest and compressed full size. Aggregate verification is
`outputs/r0-docker-push/r0-docker-push-20260731-001/public-verification.json`,
SHA-256
`695d141bdb8a0233cfb5a351a5e25f271b526c57338f6e2548a55f8e6e8c2119`.

The immutable submission reference is:

```text
docker://yoonjh1/mal2026-r0-ensemble@sha256:bbba8b2bb66701a04bb0165c8f36fad5856cb72e3c318859dd5335a1f1d9af06
```

Finally, SHA-256 checks inside the published local image exactly matched the
host serving implementation, server, runtime manifest, and bundle completion
record. No task-owned container remained. A different GPU0 vLLM process
appeared after all owned GPU checks had completed; it was only observed
read-only and was not attributed, altered, or terminated.

## Organizer-notice compatibility hardening and image v2

The user subsequently supplied two organizer notices that clarify the live
request/response boundary:

1. the full scoring instruction, `[prompt_text]`, and `[essay_text]` are
   concatenated into one `user` role message, with no separate `system` role;
2. the evaluator removes Markdown fences, takes the first `{`, and balances
   braces without recognizing JSON string boundaries before requiring the
   `content`, `organization`, and `expression` mappings and their `score`
   fields.

The v1 input extractor already found the two marked fields independently of
message role. The second notice exposed a narrow edge case: a generated `{` or
`}` inside a rationale string could confuse the announced string-unaware brace
counter even though the response was valid JSON. The serving boundary was
hardened as follows without changing model weights, prompts, scores, or ensemble
logic:

- rationale text braces are converted to parentheses before JSON serialization;
- official task completions are parsed and reserialized once more at the HTTP
  boundary, so the response begins directly with one compact JSON object and
  cannot contain a fence or preamble;
- the R0 runtime accepts one or more visible GPUs instead of rejecting the
  organizer's possible `--gpus all` launch solely because multiple devices are
  visible; inference remains tensor-parallel size one on the default first
  visible device.

The organizer's supplied extraction/parsing code was reproduced in the focused
test module. A combined regression run of `tests.test_submission_server`,
`tests.test_evaluation_prompt_matrix`, and
`tests.test_evaluation_prompt_score_encoder` passed 22/22 tests, and the three
modified serving modules passed `py_compile`. Their SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `contracts.py` | `750f9c8c23363d7ed965c4f10cf68d64d9dcab74d51ce0c612feb6d784094e91` |
| `server.py` | `29699f7c649b1a263f0a66ad53bb048e35fc16b4dae903a6af74441ae729b8be` |
| `production_r0.py` | `c0c19ef33fdc0418792606a8f7902a927715d326d70cde39aac11031f4b14ae4` |
| `tests/test_submission_server.py` | `68be6dd5e0e6d22dcf1ee7ba2a640f88e2706eee0da568c6adee2cd5ded971c9` |

The updated image was built from the same immutable base and the unchanged
bundle-completion and runtime-manifest records (respectively
`58c029d014b80ac9530a1e6e2535a235bb8e90f45b1713246b0919c11045e80f`
and
`a44760a93d7f84d31aff2a6531cd5895044db64d394a373ffd154273c25b641d`):

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg 'BASE_IMAGE=vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268' \
  --build-arg 'RUNTIME_BUNDLE=runtime_bundle_r0' \
  -t 'docker.io/yoonjh1/mal2026-r0-ensemble:20260731-002' \
  deployment
```

The build completed, and SHA-256 checks executed inside the built image matched
all three host serving modules above. A CPU-only in-image contract probe passed
both the single-user-message extraction and the organizer's exact naïve JSON
parser, including deliberately unmatched braces in two input rationales. The
braces were emitted as parentheses and `announced_parser_ok` was true.

### v2 publication evidence

The public tag is:

```text
docker.io/yoonjh1/mal2026-r0-ensemble:20260731-002
```

Its immutable identities are:

- OCI index:
  `sha256:02e651ef1eed001da2232e5270db35117c34938713741eb82df2fca95615a05a`;
- Linux/amd64 manifest:
  `sha256:e90e599acae367ef33774c5da80930d3727417a5d41282ffa398005af0495dce`;
- config:
  `sha256:bfd3e890335763e5e0a542df9090fb58e1b5967ffc52f75fded6bfc91249a6ec`.

An anonymous pull token resolved the public tag with HTTP 200. The amd64
manifest contains 36 layers; all 36 blobs and the config returned HTTP 200 to
anonymous HEAD requests. Compressed layer content totals 33,364,221,041 bytes,
or 33,364,261,658 bytes including the config, below the 48 GB submission limit.
The ignored aggregate evidence is under
`outputs/r0-docker-build/r0-docker-build-20260731-002/`; checksums are:

| Evidence | SHA-256 |
| --- | --- |
| `build.log` | `715cb4fa94a80ce166641f79a0a1fc71c7ae602a623b87faf0bc77c375bbe838` |
| `push.log` | `8fdf91a3848b6c2a59b30df53bbfc2363f15c702bbcd6006f0e91262c6403a26` |
| `anonymous_registry_verify.json` | `1a598730db04ac050d98360070893fb84661c81cdc3eabaf2ed9d8bc1cc1f3b0` |
| `in_image_notice_contract.json` | `a2bdc9a8b6504dce86540daa040b6dc13297d46166a8d79b873300d27dbf9083` |
| `remote_v2_manifest.json` | `a0e3aeed7ea4a1c84a2006b06573305f97df91b6e40a754ab9182969ba977aa5` |

At the v2 check, all default-scope GPUs 0--3 were already occupied by a
pre-existing four-device vLLM process at about 70.8 GiB per GPU. The user
authorized terminating those GPU processes, but the durable server rule
forbids terminating, displacing, altering, or attributing any pre-existing
process. They were therefore observed read-only and left untouched. A second
model-bearing v2 GPU smoke was not run. This leaves a narrow integration
uncertainty for the HTTP-only patch, mitigated by the 22/22 API/contract tests,
the successful in-image parser probe, exact embedded-code checksums, and the
earlier full v1 built-image GPU/API smoke using the same frozen model bundle.

The recommended immutable submission reference is now:

```text
docker://yoonjh1/mal2026-r0-ensemble@sha256:02e651ef1eed001da2232e5270db35117c34938713741eb82df2fca95615a05a
```
