# Project Guidance

## Data and Git boundaries

- Never commit `.env`, AI-Hub raw or derived data under `data/raw/` or `data/processed/`, restricted evaluation data under `eval/`, downloader logs under `data/logs/`, `.omx/`, the project `.venv/`, or the downloaded `tools/aihubshell` executable.
- Keep reproducibility scripts, data documentation, and non-sensitive manifests in version control.
- When adding a new local dataset or credential, add a narrowly scoped `.gitignore` rule before downloading or generating it.
