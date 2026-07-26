# RLAIF Qwen3-Embedding two-initialization comparison v1 — 2026-07-26-001

## Authorized task card

The user clarified that the rationale-conditioned score regressor must use the
embedding-tuned `Qwen/Qwen3-Embedding-8B` checkpoint rather than the earlier
Qwen2.5-Instruct decoder backbone, and authorized two otherwise identical
training runs:

1. initialize directly from the immutable public Qwen3-Embedding-8B snapshot;
2. initialize from the completed Qwen3-Embedding-8B model previously trained
   on 48,016 eligible AI-Hub human-feedback records, then continue on the same
   rationale-conditioned data.

Both arms use the already completed, independently selected
`rank2_ax4_random1` rationale source: 2,000 train rationales and 400 validation
rationales.  The input contains the writing prompt, student essay, and three
rationale strings.  Supervised targets are exactly `content`, `organization`,
and `expression`; the prior AI-Hub model's fourth `average` head is discarded
before continuation and no average target or prediction is used.

The public snapshot is pinned to
`Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
The warm-start is bound to the prior completion metadata and model-state
checksum.  Its first three regression-head rows and all trained LoRA tensors
are restored over the same immutable base; the large saved copy of frozen base
weights is not redundantly loaded into memory.

## Fixed execution contract

- Last-nonpad pooling and L2-normalized Qwen3 embedding, followed by an
  unbounded three-output linear regression head and raw MSE.
- LoRA rank 16 / alpha 32 / dropout 0.05 on Q/K/V/O and gate/up/down
  projections; BF16, TF32, learning rate `1e-4`, weight decay `0.01`, warmup
  ratio `0.05`, seed `2026072601`.
- Twelve epochs, global batch 64 (`4` examples/GPU × four-process DDP × four
  accumulation steps), expected 384 optimizer updates.  Validation uses batch
  8/GPU.
- One short GPU0 gate for each initialization (four examples, one update),
  followed automatically by the fixed full DDP runs and validation on GPUs
  0--3.  GPUs 4--7 are neither queried nor used.
- Aggregate-only persistence under ignored `outputs/`; essays, rationale text,
  IDs, row predictions, and generated data remain in restricted/ignored roots.

The runner is `scripts/run_rlaif_qwen3_embedding_comparison_v1.py`.  It records
the Git SHA, exact configs/commands, environment, GPU authorization, source and
model bindings, finite-loss/state gates, 400-example validation population,
and per-axis RMSE/Spearman.  The three-axis macro is a diagnostic only, not an
average-score target.  Because the same canonical validation split informed
earlier model/rationale selection, this comparison is descriptive and will not
be used for further hyperparameter tuning.

## Result status

Implementation and GPU-free contract checks passed.  The existing environment
does not include pytest, so the seven directly relevant test functions were
imported and executed with `.venv-standard`; all seven passed, as did Python
byte-compilation and both-arm config validation.  Actual GPU0 gate, full
training, and validation evidence will be appended after the durable runner
completes.
