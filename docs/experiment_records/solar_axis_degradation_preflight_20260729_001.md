# Solar-Open2 train-only axis-degradation preflight — runs 001--003

Status: **prompt/data preflight passed; model runtime unavailable in the pinned environment**

## Trigger and protocol

The augmentation gate was triggered by frozen-validation macro RMSE
`0.595936` for Qwen3-Embedding-8B and `0.644058` for KURE-v1. Validation is not
used to generate or select augmented records.

For every one of the 2,000 training essays, the fixed prompt requests exactly
three new essays:

1. degrade `content` while preserving organization and expression;
2. degrade `organization` while preserving content and expression;
3. degrade `expression` while preserving content and organization.

Each generated essay is still scored jointly on all three axes. This is data
augmentation and is not axis-triplet rationale evaluation. No `average` is
generated or consumed. The expected output count is 6,000.

Prompt config:
`configs/solar_axis_degradation_prompt.v1.json`, SHA-256
`5b1d436d4e31e194bf3a617d3489e8e2be3768fc760bbed5c45f26a81b58d422`.
The Solar tokenizer audit over all 6,000 requests found maximum 1,150 prompt
tokens, p99 1,072, p95 1,039, and median 948. With a 2,200-token output budget,
the fixed 4,096-token model length does not truncate any audited request.

The parser fails closed unless the requested target axis is lowered by the
bound amount, non-target axes remain within the configured tolerance, all
scores are quarter-step continuous values in `[1,5]`, the output schema is
exact, and the rewritten essay passes the length-ratio gate.

Git SHA at preflight: `848aa8e608c9d7f9d94034c8715d50c972ea10fb`.
Authorized GPU scope: physical GPUs 0--3 only, TP=4. GPUs 4--7 were neither
queried nor used.

## Model binding

The requested BF16 model is locally present at
`/dataset/large-models/upstage/Solar-Open2-250B`, but occupies
500,617,804,951 bytes. Its local README assumes eight GPUs with at least 141 GB
each, so the BF16 weights cannot be served on four 80-GB H100s.

The same local README identifies NotaAI's official quantized models for smaller
GPU configurations. The preflight therefore bound the already-local W4A16
INT4 derivative at
`/dataset/large-models/nota-ai/Solar-Open2-250B-Nota-INT4`, whose runtime
weights occupy 142,924,057,368 bytes across 27 shards and whose README specifies
vLLM tensor parallel 4.

Bindings:

- base config SHA-256:
  `fb6428ba165af1ace1d98f9170f6bafce061347593a94bd16b4b8aa3d6fe09f9`;
- INT4 config SHA-256:
  `039c9fe98844aa026aba4260692c1869a3bd2eae385d06f865714b816928a7b5`;
- weight-index SHA-256:
  `255b0cb9e82b5f564290bdd1c52734e2f9809d74ee80b056fdf3e3c601df1ae7`;
- remote model-code SHA-256:
  `b6ea8bfbbf66588ec47e6b7fa683a7ca75c328546c331a7a51015f7bb0563ed1`;
- chat-template SHA-256:
  `111eec19d6dd69146a4f29a084ea50a356aa907e83029e0d8d1c9dec883679c0`.

Using the quantized derivative is a documented runtime-feasibility deviation;
the requested Solar-Open2-250B model family and augmentation protocol are
unchanged.

## Runtime attempts and exact failures

The pinned environment contains vLLM 0.25.1, Transformers 5.14.1, and PyTorch
2.11.0. It does not contain `fla-core` or `auto-round`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:. \
  .venv-standard/bin/python scripts/run_solar_axis_augmentation.py
```

The runner writes a manifest before starting the server, launches one TP4
server, requires a single real row to pass strict parsing, and only then opens
the full 64-request asynchronous queue.

| Run | Recovery change | Result |
|---|---|---|
| 001 | initial command | CLI rejected obsolete `--disable-log-requests`; no model load |
| 002 | removed the rejected flag | installed vLLM did not register `SolarOpen2ForCausalLM`; no model load |
| 003 | forced Transformers model implementation | remote model module import failed because `fla` is absent; no model load |

Run 003's terminal cause is:

```text
ModuleNotFoundError: No module named 'fla'
ImportError: Plese run `pip install -U fla-core`
```

The installed vLLM then reports that no usable registered architecture or
remote `AutoModel` can be loaded. All three failed manifests and server logs
are preserved under ignored output paths. No augmented row was emitted and no
GPU memory was allocated. A post-failure read-only check showed GPUs 0--3 idle.

## Boundary and required recovery

The local Solar README recommends either the Upstage Solar vLLM fork based on
vLLM 0.22.0/CUDA 12.9 or the `upstage/vllm-solar-open2` Docker image. Neither is
present locally. Installing `fla-core`/the Upstage fork or pulling that Docker
image changes the pinned environment or transfers external data, so it was not
performed without explicit authorization.

The next valid action is to authorize one of those official runtime paths,
then rerun the same one-row TP4 smoke and continue directly to all 6,000
train-only generations on success. Replacing Solar with a different generator
would be a scientific-protocol change and has not been done.
