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
