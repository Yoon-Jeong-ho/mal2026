# Standard decoder training and evaluation stack

The current Qwen decoder implementation uses no bespoke optimizer, DDP, or
`generate()` evaluation loop.

| Stage | Maintained component | Purpose |
|---|---|---|
| direct score SFT | TRL `SFTTrainer` + PEFT LoRA | assistant-only conversational SFT, output score JSON |
| human feedback → score SFT | TRL `SFTTrainer` + PEFT LoRA | assistant-only SFT, output the nine AI-Hub human-feedback fields then score JSON |
| decoding evaluation | vLLM `LLM.chat` offline batch API | deterministic batched LoRA generation and strict parsing |

## Privacy and data contract

Only the canonical aggregate manifest
`data/manifests/aihub_human_feedback_v1.json` is accepted. It points to
ignored `data/processed/aihub_human_feedback_v1/` files, whose hashes and row
counts are verified before use. The corpus includes descriptive and
argumentative AI-Hub **Training** rows only; the essay corpus is excluded.

Selection uses `selection_train` / `selection_dev`. Refitting uses only
`refit_train`. Final evaluation can read only the fixed `eval/validation.jsonl`
path after the run configuration supplies its SHA-256. Generated rows,
feedback, student texts, prompts, identifiers, and completions are never
persisted. vLLM outputs are parsed in memory and invalid output receives the
predeclared `selection_train` mean-score fallback. W&B receives aggregate
metric scalars only, never tables, examples, prompts, output text, or artifacts.

## vLLM compatibility contract

Every frozen decoder-evaluation config must set both `"enforce_eager": true`
and `"disable_flashinfer_sampler": true`; the evaluator rejects missing or
false values. These are two distinct, explicit compatibility choices for the
shared vLLM 0.25.1 environment, not automatic fallbacks.

`enforce_eager=True` disables vLLM's `torch.compile` integration and CUDA
Graphs, as documented for the offline `LLM` interface. This was the initial
compatibility setting after a TorchInductor worker required unavailable
`ninja`, but it did **not** address the observed evaluation startup failure:
vLLM warmup still initialized FlashInfer's sampler JIT and attempted to run
`ninja`.

Before importing vLLM, the evaluator sets the process-local documented switch
`VLLM_USE_FLASHINFER_SAMPLER=0`, selecting vLLM's native sampler instead. An
already-set caller value other than exactly `0` is a configuration conflict and
is rejected; the evaluator never silently overwrites it. This avoids the
FlashInfer JIT dependency, but native sampling and eager execution can both
reduce decoding throughput. Both frozen fields are persisted in aggregate
evaluator provenance and W&B run config. See the [official vLLM compile
debugging guide](https://docs.vllm.ai/en/stable/design/debug_vllm_compile/) and
the [vLLM environment-variable reference](https://docs.vllm.ai/en/stable/configuration/env_vars/).

## Commands

Create an ignored runtime and install the pinned standard stack there; do not
alter a shared environment:

```bash
python3.12 -m venv .venv-standard
.venv-standard/bin/pip install -r requirements-standard.txt
.venv-standard/bin/pip install -e .
```

Copy a template into an ignored run-config path and replace placeholders. For
selection, use the standard Torch launcher:

```bash
torchrun --nproc_per_node=8 scripts/train_standard_decoder_sft.py --config /ignored/run-config.json
```

Then run the vLLM selector or final evaluator in a separate tmux session:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv-standard/bin/python scripts/evaluate_standard_decoder_vllm.py --config /ignored/eval-config.json
```

Before any multi-GPU launch, perform a one-GPU smoke run and verify that the
adapter plus `standard_training_complete.json` exist; after evaluation verify
only `aggregate_metrics.json` and aggregate W&B scalars. The selection
criterion is `primary_macro_mae`; Trainer's `eval_loss` drives its standard
checkpoint/early-stopping lifecycle. Do not use frozen final validation for
model selection or early stopping.

## Two-stage selection/refit lifecycle

`SFTTrainer` uses `eval_loss` only for its maintained checkpointing and
loss-convergence early-stopping callback. That value is **not** the model
selection metric. Every retained `checkpoint-N` is subsequently evaluated by
the vLLM source-dev evaluator. `select_standard_decoder_checkpoint.py` requires
one aggregate result for every retained checkpoint and deterministically selects
the lowest `primary_macro_mae` (lower update count breaks ties). It writes the
ignored, aggregate-only `selected_checkpoint.json`.

The refit config must set both `selected_global_step` and
`selection_summary_path` from that summary. The TRL refit run uses
`max_steps=selected_global_step`, `refit_train`, and no dev split; final vLLM
evaluation accepts only its `adapter/` export. Thus final validation is never
used for selection, training loss early stopping, or checkpoint choice.

Qwen2.5's chat template lacks TRL's Jinja `generation` mask. The dataset is
therefore represented in TRL's maintained **conversational prompt-completion**
form (`prompt=[system,user]`, `completion=[assistant]`) with
`completion_only_loss=True`; it does not use `assistant_only_loss` or a custom
loss loop.
