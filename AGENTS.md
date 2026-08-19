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

- The default MAL2026 GPU scope is GPUs 0--3, with GPU0 used first for
  preflight and smoke gates. The current user may explicitly expand a named
  task card to specified GPUs among 4--7. After that authorization, the lead
  may perform the minimum read-only availability check needed and use only
  those named GPUs. Never terminate, displace, or alter a pre-existing process,
  and never infer its ownership. Record the exact GPU scope and the user's
  authorization in the run ledger.
- Follow the two-tier execution authority in
  [`docs/codex_execution_policy_v2.md`](docs/codex_execution_policy_v2.md)
  and its MAL2026 implementation in
  [`docs/omx_codex_research_orchestration.md`](docs/omx_codex_research_orchestration.md).
  Integration recovery is autonomous within an approved stage; scientific
  decisions require recorded authorization unless a named experiment plan
  explicitly preapproves them. Current run state belongs only in append-only
  ledgers or aggregate experiment records, never in durable policy documents.
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
- Repository-owned long-running Python processes for the vLLM soak and its
  scheduler must use `setproctitle` with the exact title `(D)_vllm` so server
  operators can distinguish this workload. Do not install a package for this;
  use the existing project environment and fail preflight if it is unavailable.
- Within an approved experiment stage, the lead autonomously completes allowed
  integration recovery and continues after a passing smoke. Stop only for a
  boundary or cost issue, destructive action, unapproved external API or new
  data, or a scientific decision that is not explicitly preapproved.

## Guidance maintenance

- Keep this file limited to durable, repository-specific rules. Put current run state
  and results in experiment records; update this file only after a verified durable
  repository convention changes.
