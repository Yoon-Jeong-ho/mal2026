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

All decoder runtime configs require `canonical_config_path`, pointing to a
filled, validated copy of `configs/decoder-direct.template.json` or
`configs/decoder-rationale-score.template.json`. The runtime model, adapter,
sequence, seed, and optimization fields must match that canonical contract.
Model and tokenizer revisions must be immutable 40-character lowercase Git
commit SHAs.

The runners accept only the canonical local `eval/train.jsonl` or
`eval/validation.jsonl` paths with their frozen SHA-256 values. A selection
config derives its internal development partition deterministically from
prompt groups in canonical training data. A refit config requires the selected
optimizer-update count and uses all canonical training records. Final
evaluation requires both selection/refit run IDs, a completed refit adapter
located inside `outputs/runs/<refit-run-id>/`, and the exact saved refit train
mean; it cannot select checkpoints or fit calibration.

Every decoder artifact must use exactly
`outputs/runs/<run-id>`; symlinks anywhere in this path are rejected. Each
completed run writes an aggregate-only local run manifest (code/config/data
hashes, command, environment/hardware, metrics, and deviations). W&B is
rank-zero only, disables code/model upload, and uses the immutable run ID with
`resume="never"`.

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

## Score-blind synthetic rationale generation

Do **not** supply a hand-authored or arbitrary rationale JSONL to SFT. Run the
frozen teacher first, on the SFT training partition only:

```bash
PYTHONPATH=src python scripts/generate_decoder_rationales.py --config /secure/configs/decoder-rationale-teacher-selection.json
```

`decoder_rationale_generate.py` uses only prompt and essay text in its teacher
request: it never reads scores, IDs, document IDs, prompt numbers, or split
names. Its teacher revision, tokenizer revision, custom template hash,
deterministic generation (`do_sample=false`, 512 tokens), seed, and two retry
limit are pinned by the canonical rationale config. Each response must pass the
shared exact-schema/offset/no-score-cue validator. Failed records are retained
as empty local artifacts only; generation stops the protocol if fewer than 85%
are nonempty valid. The resulting ignored run directory contains
`synthetic-rationales.jsonl` and aggregate `rationale_provenance.json`.

Rationale SFT refers to that **run ID**, not a caller-chosen file path. It
checks the source-train hash, deterministic partition ID hash/count, validation
gate, and artifact checksum before use. Synthetic evidence is model-generated
training scaffolding, not a human label or proof that a generated explanation
is faithful.
