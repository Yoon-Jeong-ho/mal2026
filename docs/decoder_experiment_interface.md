# Decoder SFT interface (Qwen2.5-7B)

`src/mal2026/decoder_train.py` and `decoder_eval.py` implement only the two
Qwen decoder regimes. They are invoked through `scripts/train_decoder_sft.py`
and `scripts/evaluate_decoder.py` with `PYTHONPATH=src` (or an installed
package). Do not pass raw writing data, generations, or labels to W&B.

## Inputs

Training/dev/evaluation JSONL rows are restricted local files with exactly the
required fields `id`, `prompt`, `essay`, and
`score.{content,organization,expression,average}`. The runner never uses IDs,
split names, or scores in a user message. Synthetic-rationale JSONL is
train-only and contains exactly `{"id":...,"rationale":[...]}` per training
row; it deliberately cannot contain score fields or split metadata.

Each rationale item has exactly `criterion`, `quote`, `start`, `end`, and
`observation`. Criteria are `CONTENT`, `ORGANIZATION`, `EXPRESSION`; nonempty
candidates must cover all three exactly once and their quotes must equal the
unmodified essay slice at `[start:end]`. The version-controlled shared
validator rejects numeric, rating, or score-proxy text in `observation` (but
not in quoted source text). These generated artifacts remain under ignored
`outputs/` paths.

## Configuration and execution

Both runners require a non-secret JSON config. Model and tokenizer revisions
must be immutable 40-character lowercase Git commit SHAs. A selection config
derives its internal development partition deterministically from the
hash-validated `eval/train.jsonl` prompt groups. A refit config requires the
selected optimizer-update count and uses all training records. Final evaluation
requires both the selection and refit run IDs and uses a saved
optimization-train fallback mean; it cannot be used for selection.

Launch after dependency and config smoke gates with Accelerate, for example:

```bash
PYTHONPATH=src accelerate launch --num_processes 8 scripts/train_decoder_sft.py --config /secure/configs/decoder-direct-selection.json
```

The frozen decoder policy is BF16 DDP (no `device_map`), one example/GPU,
accumulation 8, LoRA rank/alpha/dropout `32/64/0.05`, target projections
`q/k/v/o/gate/up/down`, and 3,072 input tokens. Prefix truncation retains 75%
head and 25% tail; assistant tokens alone contribute to SFT loss.

Targets are exact ordered JSON with numeric two-decimal values. The parser
accepts no prose, markdown, reordered keys, strings, scientific notation, or
out-of-range values. An invalid output receives the frozen optimization-train
mean and remains in metrics. Evaluation writes only restricted ID/prediction
artifacts under ignored outputs; W&B receives aggregate metrics/config only.
