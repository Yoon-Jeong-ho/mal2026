# MAL2026 Docker submission runtime

This directory contains the OpenAI-compatible serving boundary for the
MAL2026 writing-scoring submission.  It is intentionally separate from
training and restricted row artifacts.

The evaluator contract is:

- the image starts without command-line arguments;
- the server listens on `0.0.0.0:8000`;
- `GET /health`, `GET /v1/models`, and
  `POST /v1/chat/completions` are available;
- the evaluator may place the scoring instructions, `[prompt_text]`, and
  `[essay_text]` together in one `user` message; no `system` message is
  required;
- the task completion is one compact JSON object containing integer scores and
  nonblank Korean rationales for `content`, `organization`, and `expression`;
- free-text rationale braces are converted to parentheses so the organizer's
  announced string-unaware first-JSON brace counter cannot terminate early.

## Runtime bundle

`runtime_bundle/` is ignored except for its README.  The prepared layout is:

```text
runtime_bundle/
  manifest.json
  evaluation.txt
  score/
    backbone/                 # merged BF16 Hugging Face encoder + tokenizer
    score_head.safetensors    # three-axis bounded-regression head
  rationale/
    base/                     # pinned skt/A.X-4.0-Light snapshot
    adapters/
      score_blind_v2/         # exact evaluation-prompt draft rationale adapter
      final_dpo/              # optional score-conditioned DPO adapter
```

Copy `runtime_manifest.template.json` as the bundle manifest for the fully
matched two-pass exact-prompt candidate.  Use
`runtime_manifest.dpo.template.json` only for the three-pass DPO-rationale
candidate.  The latter requires a separate frozen end-to-end gate.

Do not place evaluation rows, generated rationale rows, identifiers,
predictions, optimizer states, logs, or credentials in the bundle.

### Metric-first R0 ensemble bundle

`runtime_bundle_r0/` is the staged bundle for the strongest recorded score
candidate.  Its request path is:

1. historical score-blind A.X rationale generation;
2. four Qwen3-Embedding epoch adapters/heads evaluated independently;
3. uniform mean of the four continuous predictions, clipping to `[1, 5]`,
   then half-up rounding;
4. final A.X DPO rationale generation conditioned only on the emitted integer
   scores.

The custom adapters and regression heads are also published at
[`yoonLM/mal2026-r0-ensemble-v1`](https://huggingface.co/yoonLM/mal2026-r0-ensemble-v1),
revision `b5029e8c1e78a5a2533f5e7c62c26fc0dd1041be`.  The runtime bundle additionally
embeds the pinned public base snapshots, so startup is offline and does not
depend on Hugging Face availability.

## Local contract test

The API tests use an in-memory fake pipeline and do not load models or touch a
GPU:

```bash
PYTHONPATH=deployment/src:src .venv-standard/bin/python -m unittest -v \
  tests.test_submission_server
```

## Image build

The Dockerfile deliberately has no default base image.  Supply an immutable
CUDA/vLLM image digest that already contains the locked runtime dependencies;
the Docker build must not resolve mutable package versions.

```bash
docker build \
  --build-arg BASE_IMAGE='<repository>@sha256:<digest>' \
  -t '<registry>/<namespace>/mal2026-writing:<immutable-tag>' \
  deployment
```

For the R0 bundle, use the already-audited Linux/amd64 vLLM base digest and
select the R0 build context explicitly:

```bash
docker build \
  --build-arg 'BASE_IMAGE=vllm/vllm-openai@sha256:f0b9a0dc75a9fca3b6811e3279367b2d6a448055a000bfd13859587d74cef268' \
  --build-arg 'RUNTIME_BUNDLE=runtime_bundle_r0' \
  -t '<registry>/<namespace>/mal2026-r0-ensemble:20260731-002' \
  deployment
```

The staged R0 bundle is about 35 GB.  Confirm adequate Docker data-root space
before building because a classic Docker build may temporarily retain both the
content layer and its unpacked snapshot.  Do not substitute a mutable base tag.

The model bundle is copied into the image.  No Hugging Face download or other
network access is performed when the container starts.

## Recommended published image: R0 score ensemble + best rationale SFT

The current recommended image preserves the strongest recorded R0 score
ensemble and replaces only the post-score rationale stage with the independently
selected score-blind `mal-direct-lora-epoch2` SFT model.  Its rationale prompt
is the exact hash-bound `Rationale_evaluation_training.txt` used during training
and evaluation; the externally received single `user` message is parsed into
`prompt_text` and `essay_text` before that internal template is rendered.

```text
docker.io/yoonjh1/mal2026-r0-ensemble:20260812-003
```

Immutable reference:

```text
docker://yoonjh1/mal2026-r0-ensemble@sha256:fb226eb5263021b6c35190ada65920c8101b7dedc45a836eb476f8fc10907f28
```

The Linux/amd64 manifest is
`sha256:70a052a02e79b0006d68a111777c6402f59e00940588f0dcf939ed9b3789b974`.
Its 38 compressed layers total 33,664,030,086 bytes.  A no-argument GPU0
container/API smoke reached health in 124 seconds and passed the organizer's
single-user-message and first-JSON parser contract.  An anonymous registry
manifest request returned HTTP 200 at publication verification time.

## Previous published R0 image

The verified public Linux/amd64 image is:

```text
docker.io/yoonjh1/mal2026-r0-ensemble:20260731-002
```

For an immutable submission reference, use the OCI index digest:

```text
docker://yoonjh1/mal2026-r0-ensemble@sha256:02e651ef1eed001da2232e5270db35117c34938713741eb82df2fca95615a05a
```

The Linux/amd64 image manifest is
`sha256:e90e599acae367ef33774c5da80930d3727417a5d41282ffa398005af0495dce`.
An anonymous registry pull-token check resolved the index and returned HTTP 200
for every config/layer blob.  Docker Hub reports a compressed full size of
33,364,221,041 layer bytes (33,364,261,658 bytes including the config), so
allow substantial pull time before container startup.  Tag `20260731-001`
remains published as the previous parser-hardened baseline.  Use
`20260812-003` for the latest rationale candidate.
