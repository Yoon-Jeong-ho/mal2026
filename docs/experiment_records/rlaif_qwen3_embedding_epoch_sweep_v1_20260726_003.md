# Qwen3-Embedding warm-start epoch sweep v1 — 2026-07-26-003

## Authorized task card

After the Qwen3-Embedding two-initialization comparison selected the AI-Hub
48,016-row warm-start, the user explicitly authorized repeating that arm while
saving the model after every epoch and evaluating all saved epochs.

- **Arm:** only `qwen3_aihub_warmstart`; the failed public-base scorer is not
  extended.
- **Data:** the same 2,000 train writings and A.X `random1` rationales; the
  same 400 canonical validation writings are evaluated once per checkpoint.
- **Targets:** exactly `content`, `organization`, and `expression`.  The old
  AI-Hub `average` head remains discarded and no average target is read or
  emitted.
- **Training:** the prior fixed seed and optimizer schedule are unchanged:
  12 epochs, 32 optimizer updates per epoch, 384 updates total, global batch
  64, BF16/TF32, LoRA rank 16, and maximum length 2,048.
- **Checkpoints:** the reconstructable LoRA tensors and three-output head are
  saved after steps 32, 64, ..., 384.  Each checkpoint is approximately 201
  MiB and is bound to the immutable base snapshot; redundant 17-GiB frozen
  base copies and optimizer states are not stored because evaluation is the
  only declared checkpoint consumer.
- **Evaluation:** all 12 checkpoints are loaded in one DDP evaluation process
  and evaluated independently on 400 unique validation essays, one prediction
  per essay per checkpoint.  Only aggregate C/O/E RMSE and Spearman are saved.
- **Resources:** the new save/load path receives one four-example GPU0 train
  and evaluation gate, then the full train and evaluation use DDP on GPUs
  0--3.  GPUs 4--7 are neither queried nor used.

The canonical validation set was already exposed in earlier selection and is
now explicitly reused at the user's request.  Therefore the lowest-RMSE epoch
is a descriptive validation-selected checkpoint, not an untouched
generalization estimate.  No hyperparameter is changed after viewing its
curve.  Generated text, inputs, identifiers, row predictions, checkpoints, and
logs remain in ignored/restricted roots.

## Result status

Implementation, GPU-free checks, actual GPU gate, full training, and the
twelve-checkpoint evaluation are pending.  Aggregate evidence and limitations
will be appended after the durable runner completes.
