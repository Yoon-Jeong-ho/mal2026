# Official bundle rationale handoff and rationale-aware encoders — run 021

Status: **encoder baseline completed; augmentation gate triggered**

## Fixed contract

- The deployed rationale contract is one bundled JSON containing `content`,
  `organization`, and `expression`. Axis-triplet prompts are not used for
  training, model selection, or the encoder handoff.
- The score encoder receives the prompt, essay, and the complete three-axis
  rationale bundle. It predicts three continuous scores in `[1, 5]` with
  ordered heads `content`, `organization`, and `expression`.
- The original essay is authoritative and the rationale is untrusted auxiliary
  evidence. Gold/reference scores and `average` are absent from encoder input.
- Model selection uses a deterministic prompt-stratified 1,600/400 split made
  only from the 2,000 training rows. The canonical 400-row validation split is
  loaded only after selected-epoch refit on all 2,000 training rows.
- The canonical validation metrics below are descriptive frozen-validation
  results and were not used to choose an epoch.
- Authorized hardware was physical GPUs 0--3 only. GPU0 was used for the
  smallest real preflight before four-GPU training.

Git SHA at execution: `848aa8e608c9d7f9d94034c8715d50c972ea10fb`.

The derived rationale-aware encoder prompt is
`configs/official_rationale_aware_score_prompt.v1.json`, SHA-256
`692da6e051ba9864d5699f5e8e11c143ce30784415c5050b89d63a3aedccc60c`.
It is derived from `evaluation.txt`, not represented as a verbatim organizer
prompt. The generative rationale-output instruction was removed because these
models have three regression heads rather than a decoder output channel.

## Selected bundle rationale handoff

The exact-Q4 frozen-validation comparison selected the DPO arm numerically:
`4.9877083/5` versus `4.9839583/5` for the SFT baseline. The paired difference
was small (`+0.045/60`, exact sign-test `p=0.092461`) and the judge was ceiling
saturated, so further judge-optimized RL was stopped. This is a saturation
decision, not evidence that rationale quality is perfect.

The selected model generated a complete deterministic handoff:

| Split | Records | SHA-256 |
|---|---:|---|
| train | 2,000 | `45dc9bfd05d60c75214221e34149ed7bff6dae0d571a90fde287ab193bb6f347` |
| validation | 400 | `0e7d368493e39cc9d88ed0d8cf4f3a1bfc14c8e007dca6e4ef75f4f0a7d9f7bd` |

Handoff manifest SHA-256:
`ea962b5bdad406f3090a22999c5be34c3289d418d2ce383b09186f4888efb41f`.
The manifest records `structure=bundle`,
`axis_triplet_used_for_training_or_selection=false`, and
`human_or_reference_score_read_or_prompted=false`.

```bash
PYTHONPATH=src:. .venv-standard/bin/python \
  scripts/run_selected_dpo_rationale_handoff.py
```

All row-level rationales remain under the ignored restricted-data path. No
essay, rationale, identifier, row score, or prediction is included here.

## KURE AI-Hub full-parameter initialization

KURE-v1 was first trained on the 48,016-row AI-Hub prepared dataset using a
three-score head. Selection used only the deterministic AI-Hub train/dev
partition; the project validation split was never accessed. A BF16 attempt
showed non-finite gradients/loss and was preserved as a negative run. The
repaired run used FP32 without changing the data or target contract.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:. \
  .venv-standard/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_kure_aihub_score_pretrain.py \
  --config configs/official_kure_aihub_score_full_pretrain.v1.json \
  --mode full
```

The selected AI-Hub epoch was 4:

| Metric | Value |
|---|---:|
| continuous macro RMSE | 0.776428 |
| continuous macro Spearman | 0.323324 |
| projected-integer macro RMSE | 0.821568 |

The refit used all 48,016 rows for four epochs (3,004 optimizer steps). The
completion record SHA-256 is
`c91704e5a5c5f54b086552731fe87febdaee8c42273f93a5492f1f8626b47959`;
the full-model artifact binding is
`e60ff3270cd856e663243822056b81d08966ee50dc9b8da75f69a138d611a126`.

## MAL rationale-aware encoder results

Both encoders loaded the full AI-Hub backbone and matched three-score head,
then attached a fresh MAL LoRA adapter. Raw fractional labels were preserved;
no half-up target projection or average target was used.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:. \
  .venv-standard/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/run_rationale_aware_encoder.py \
  --config configs/rationale_aware_qwen3_embedding_8b_aihub_mal.v1.json \
  --mode full

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:. \
  .venv-standard/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/run_rationale_aware_encoder.py \
  --config configs/rationale_aware_kure_v1_aihub_mal.v1.json \
  --mode full
```

| Model | Selected train-internal epoch | Validation content RMSE | organization RMSE | expression RMSE | **macro RMSE** | macro Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Embedding-8B | 3 | 0.533135 | 0.719022 | 0.535651 | **0.595936** | 0.563817 |
| KURE-v1 | 7 | 0.588744 | 0.784341 | 0.559090 | **0.644058** | 0.506220 |

Result bindings:

- Qwen result SHA-256:
  `a5ccc689cd3888a6181b73e8ff4339f3494370ea51410081c769ac6b0febdfbc`;
  selected/refit trainable-state SHA-256:
  `80cc4d092231d9c400fc79b3f5c2394148fb750c9d3e9ac4bb7fd36beb2d3e9e`.
- KURE result SHA-256:
  `7761f1942cbef2245575ddbc17e23855bce40103d2ac7d7193abbf529f28a7cc`;
  selected/refit trainable-state SHA-256:
  `de742ab2af7826adb1e9058926fbb80e89395a135fdd68ab79547d385525307c`.

The Qwen shuffled-rationale diagnostic degraded macro RMSE from `0.595936` to
`0.764778`, showing that this model uses aligned rationale information. KURE
changed only from `0.644058` to `0.646435`, so its prediction is much less
sensitive to the rationale. The shuffled arm is diagnostic only and was not
used for training or selection.

### Preserved negative runs

- The first Qwen full run was intentionally stopped after 15/400 selection
  steps when monitoring showed that a larger microbatch would fit. The repaired
  run changed `batch/accumulation` from `2/4` to `4/2`, preserving global batch
  64 and the scientific protocol.
- KURE MAL run 001 completed with non-finite predictions under BF16. Its
  `result.json` is preserved (SHA-256
  `30fdd18e6d21e1043c39b9ba6056ba0db6af4db4d34c4f41a09a46d8c5f9fe94`).
  Run 002 changed only the numerical dtype to FP32 and added fail-closed
  non-finite checks.

## Decision

Qwen3-Embedding-8B is the current best rationale-aware score encoder, but both
models exceed the predeclared `0.5` validation macro-RMSE gate. Solar train-only
axis-degradation augmentation is therefore required before final
train+validation refit. Organization is the largest error axis for both models.
