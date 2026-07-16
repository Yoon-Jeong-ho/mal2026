# Standard encoder training and evaluation stack

Qwen3-Embedding-8B and NV-Embed-v2 use the maintained Hugging Face
`Trainer` API for their complete optimization, DDP, checkpoint, and
early-stopping lifecycle. The project does not provide a bespoke optimizer or
manual distributed training loop for encoder scoring.

## Common data and scoring contract

Both backbones receive exactly the same prepared AI-Hub **Training-only** rows:
`selection_train` and prompt-group-disjoint `selection_dev` for model selection,
then `refit_train` for the selected fixed update count. The essay corpus and
AI-Hub upstream validation are excluded. Inputs contain only the prompt and
student writing; identifiers, human feedback, and scores are never model input.

The regression head emits `content`, `organization`, `expression`, and
`average`. Optimization is mean squared error. Selection reports clipped
1--5-score macro MAE (the mean of the four field MAEs) and uses it for the
standard `Trainer` best-checkpoint and early-stopping callbacks. Final frozen
`eval/validation.jsonl` is inaccessible to training and may be evaluated only
from a completed `refit` artifact.

## NV-Embed-v2 remote-code boundary

NV-Embed-v2 requires `trust_remote_code=True`. It is allowed only when the
configuration supplies a named, approved research/non-commercial review for the
exact immutable revision and a SHA-256 inventory of **every** Python file in a
local, non-symlinked snapshot. Before remote code or its tokenizer/config is
imported, the runner verifies the inventory and sets Hugging Face and
Transformers offline flags. The reviewed local snapshot is also forced as the
NV text-config tokenizer source. There is no generic pooling fallback: the
remote model must return `sentence_embeddings` of shape `[batch, 4096]`.

## Lifecycle

1. Copy the relevant `configs/standard-encoder-*.template.json` into an
   ignored configuration path and make a one-GPU smoke configuration.
2. Run a selection job with standard Torch DDP:

   ```bash
   torchrun --nproc_per_node=8 scripts/train_standard_encoder.py --config /ignored/qwen3-selection.json
   ```

   Verify the ignored run directory contains `final_model/model.safetensors`
   and `standard_encoder_training_complete.json`. The latter records the
   selected Trainer global step and aggregate-only metrics.
3. Create a `refit` config with `eval_steps`/`save_steps` set to zero and
   `selection_metadata_path` set to that selection completion file. The runner
   takes the selected update count from the immutable selection metadata rather
   than accepting a hand-entered refit stop point.
4. Evaluate selection or a completed refit model through standard
   `Trainer.predict`:

   ```bash
   .venv-standard/bin/python scripts/evaluate_standard_encoder.py --config /ignored/encoder-final-eval.json
   ```

Only aggregate metrics, state hashes, and non-sensitive configuration/provenance
are written. Prompts, essays, feedback, IDs, predictions, and model text remain
in memory and are never written to W&B, tracked files, or result JSON.
