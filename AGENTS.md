# MAL2026 Korean Writing Evaluation

This repository prepares Korean writing-evaluation data and contains the
reproducible code and documentation for model training and evaluation.

## Data, privacy, and Git boundaries

- Never commit `.env`, restricted evaluation inputs under `eval/`, AI-Hub raw
  or derived data under `data/raw/` or `data/processed/`, downloader logs
  under `data/logs/`, `.omx/`, `.venv/`, or `tools/aihubshell`.
- Never commit checkpoints, generated rationales, W&B local files, or run logs
  under `outputs/` or `wandb/`.
- Keep reproducibility scripts, aggregate data documentation, and
  non-sensitive manifests in version control. Do not put credentials,
  individual writing content, or identifiers in tracked documentation.
- Before downloading or generating a new local dataset, credential, or run
  artifact, add a narrowly scoped `.gitignore` rule when it is not meant for
  version control.

## Reproducibility

- Treat `docs/aihub_writing_evaluation_data.md`, the preparation scripts in
  `scripts/`, and non-sensitive files in `data/manifests/` as the canonical
  record of the AI-Hub data acquisition and preparation workflow.
- Store generated experiment artifacts only in ignored output locations. For
  each experiment, record the Git SHA, exact command/configuration, seed,
  dataset checksum/version, environment and hardware, output path, metrics,
  and deviations in a tracked experiment record.
- Run and verify a small smoke test before a multi-GPU or long-running job;
  preserve null and negative results and do not retune a protocol after seeing
  results without recording the change.
- For routine training, prefer maintained standard Hugging Face training
  integrations (``Trainer``/TRL ``SFTTrainer``) over bespoke training loops.
  Use a custom loop only for a documented framework gap. For decoder batch
  generation/evaluation, use a maintained high-throughput engine such as
  vLLM or SGLang when compatible with the model and evaluation contract.
- When a clarification would otherwise pause an approved experiment, first
  obtain a bounded proceed/stop recommendation from a subagent acting as a
  user proxy. A proceed recommendation authorizes completing the approved
  experiment stage without another routine question; it never substitutes for
  explicit consent for destructive actions, external publication, credentials,
  or use outside the approved resource and data boundaries.

## Guidance maintenance

- Keep this file limited to durable, repository-specific rules. Put current
  run state and results in experiment records, and use
  `$maintain-research-guidance` when a durable repository convention changes.
