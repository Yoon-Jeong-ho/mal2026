# Sequential standard experiment matrix

`scripts/run_standard_experiment_matrix.sh` is the approved foreground launcher
for the complete, **sequential** matrix. Run it from a dedicated tmux session
only after compatible smoke evidence for the permitted GPU 0--3 allocation has been recorded. It does
not download data, read credentials, write raw rows, or upload artifacts.

The launcher uses maintained project entry points only:

1. Direct-score Qwen SFT selection (`TRL SFTTrainer` / Torch DDP), vLLM
   source-development evaluation of every retained checkpoint, metric selection,
   fixed-step refit, and frozen validation vLLM evaluation.
2. The same lifecycle for Qwen human-feedback-then-score SFT.
3. Qwen3-Embedding-8B `transformers.Trainer` selection, fixed-step refit, and
   frozen validation `Trainer.predict` evaluation.
4. Reviewed NV-Embed-v2 `transformers.Trainer` selection, fixed-step refit, and
   frozen validation evaluation.

It runs one stage at a time and refuses to launch when `nvidia-smi` reports
an active compute process on a **selected** GPU. Occupancy on unselected GPUs
does not block the run, and it never terminates another process. All selection
uses only the prepared training/source-development partitions; the fixed
validation hash is required only for final evaluation.

## Preflight and launch

Use absolute local paths and a fresh ignored runtime directory beneath
`outputs/`. The NV review is an aggregate JSON review for the exact immutable
NV snapshot; it is passed explicitly so that remote-code approval remains
visible in the runtime configuration.

```bash
# Check naming, orchestration order, and resource settings without writing files.
scripts/run_standard_experiment_matrix.sh --dry-run \
  --runtime-root "$PWD/outputs/experiment-matrix/approved-matrix-001" \
  --run-prefix approved-matrix-001 \
  --prepared-manifest /absolute/path/to/aihub_human_feedback_v1.json \
  --validation-sha256 "$(printf '%064d' 0)" \
  --qwen-model /absolute/local/qwen2.5-snapshot \
  --qwen3-model /absolute/local/qwen3-embedding-snapshot \
  --nv-model /absolute/local/nv-embed-snapshot \
  --nv-review-json /absolute/path/to/approved-nv-review.json

# Run from tmux after replacing the dry-run placeholder hash with the pinned
# evaluation file SHA-256 and after confirming all selected GPUs are idle.
tmux new-session -s mal2026-matrix \
  'cd /dataset/aa007878/mal2026 && scripts/run_standard_experiment_matrix.sh \
    --runtime-root "$PWD/outputs/experiment-matrix/approved-matrix-001" \
    --run-prefix approved-matrix-001 \
    --prepared-manifest /absolute/path/to/aihub_human_feedback_v1.json \
    --validation-sha256 PINNED_VALIDATION_SHA256 \
    --qwen-model /absolute/local/qwen2.5-snapshot \
    --qwen3-model /absolute/local/qwen3-embedding-snapshot \
    --nv-model /absolute/local/nv-embed-snapshot \
    --nv-review-json /absolute/path/to/approved-nv-review.json'
```

The required path/hash arguments also accept `MAL2026_*` environment variables
listed by `--help`. `--num-gpus` defaults to **4** and therefore the default CUDA allocation is
GPUs **0--3**; it never targets the occupied GPUs 4--7. The launcher derives
accumulation to preserve the frozen global effective batch of 64: stable
`batch=1` uses `gradient_accumulation=16` on the permitted four GPUs. The
launcher rejects any GPU count above four and any physical GPU index outside
0--3. Explicit batch/accumulation overrides are accepted only when they
exactly retain global batch 64. This keeps the higher-batch smoke, which
produced non-finite metrics, out of the default protocol.

## Outputs and failure behavior

For a new matrix, the supplied runtime root must not exist and must be Git-ignored. It receives
only ignored JSON configs, per-stage logs, `matrix_manifest.json`, and an
aggregate-only `matrix_ledger.jsonl`. Training/evaluation output directories
are fresh direct children of the canonical ignored `outputs/standard-*` roots.
The launcher uses `set -e`, records the failed stage/log path, and stops; it
never overwrites or deletes an output.

### Resuming a failed matrix

Use the explicit `--resume-run-prefix` only for the exact same failed run. It
must equal `--run-prefix`, use the same ignored runtime root, and supply the
same prepared manifest, frozen-validation hash, model paths, GPU contract,
batch/accumulation settings, W&B routing, and checked-out Git SHA. The launcher
validates the immutable runtime manifest and every hashed selection config
before it appends a `resume_lineage.jsonl` record. A changed input, config hash,
data-manifest checksum (for schema-v2 manifests), model-review checksum, or Git
SHA is rejected rather than silently mixing protocols.

On resume it re-validates every stage whose latest ledger status is completed.
Only stages that are both ledger-recorded successful and pass the strict
artifact validator are skipped; that decision is appended as
`skipped_verified`. A failed or not-started stage is launched again only when
its output path does not already exist. If an incomplete output exists, the
launcher refuses rather than overwriting it; preserve it and launch a distinct
approved retry instead. Existing per-stage logs are also preserved and a retry
writes a numbered attempt log.

For example, after a transient downstream failure with no output directory for
the failed stage:

```bash
scripts/run_standard_experiment_matrix.sh \
  --resume-run-prefix approved-matrix-001 \
  --runtime-root "$PWD/outputs/experiment-matrix/approved-matrix-001" \
  --run-prefix approved-matrix-001 \
  --prepared-manifest /absolute/path/to/aihub_human_feedback_v1.json \
  --validation-sha256 PINNED_VALIDATION_SHA256 \
  --qwen-model /absolute/local/qwen2.5-snapshot \
  --qwen3-model /absolute/local/qwen3-embedding-snapshot \
  --nv-model /absolute/local/nv-embed-snapshot \
  --nv-review-json /absolute/path/to/approved-nv-review.json
```

### Provenance-linked direct-selection continuation

`--continue-from-decoder-selection-run ABS_DIR` is narrower than resume. It is
only for the recorded case in which an approved pre-fix direct decoder
selection completed, but its first vLLM source-development checkpoint
evaluation failed before producing an evaluation output. It must be combined
with a **new** ignored runtime root and prefix; it cannot be combined with
`--resume-run-prefix`. The parent must be a direct child of
`outputs/standard-runs` named `PREFIX-decoder-direct-selection`.

Before any GPU preflight, the launcher validates the parent runtime/config
hash, exact direct-selection architecture/data/model/batch/seed identity,
finite completion metrics, exact completion-to-config/identity binding, every
retained checkpoint adapter, and the recorded **first-candidate** failed vLLM
stage/log. The historical schema-v1 parent has no stored data-manifest hash, so
the launcher proves the supplied canonical data manifest byte-for-byte against
the immutable `dfa3c34` Git blob; a same-path mutation is rejected. It also
refuses a pre-existing parent `selected_checkpoint.json` or any old
parent-prefix direct source-development evaluator output. The new schema-v3
manifest records the parent paths and Git SHA, current continuation Git SHA,
the exact guarded-entrypoint fix diff and rationale, failure identity, and
SHA-256 evidence for the parent completion, selection config, data manifest,
and retained adapter configs. The new ledger records
`reused_verified_parent` for direct selection; the parent ledger/config/logs
remain unchanged.

The new-prefix vLLM evaluations run for every retained parent checkpoint. Only
after all candidate metrics pass their artifact gates may the selector append a
previously absent canonical `selected_checkpoint.json` to the parent selection
directory. It refuses an existing file rather than overwriting it. The direct
refit, final evaluation, human-feedback decoder, and encoders all use the new
prefix and retain the normal per-stage GPU-exclusive preflight and artifact
gates.

```bash
scripts/run_standard_experiment_matrix.sh \
  --continue-from-decoder-selection-run \
    "$PWD/outputs/standard-runs/old-prefix-decoder-direct-selection" \
  --runtime-root "$PWD/outputs/experiment-matrix/new-continuation-prefix" \
  --run-prefix new-continuation-prefix \
  --prepared-manifest /absolute/path/to/aihub_human_feedback_v1.json \
  --validation-sha256 PINNED_VALIDATION_SHA256 \
  --qwen-model /absolute/local/qwen2.5-snapshot \
  --qwen3-model /absolute/local/qwen3-embedding-snapshot \
  --nv-model /absolute/local/nv-embed-snapshot \
  --nv-review-json /absolute/path/to/approved-nv-review.json
```

After every completed training or evaluation stage it rejects non-finite JSON
metrics/provenance before proceeding. Inspect the aggregate metrics and ledger,
not process existence, to determine completion.
