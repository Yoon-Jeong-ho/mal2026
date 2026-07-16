# Project Guidance

## Data and Git boundaries

- Never commit `.env`, AI-Hub raw data under `data/raw/`, downloader logs under `data/logs/`, `.omx/`, or the downloaded `tools/aihubshell` executable.
- Keep reproducibility scripts, data documentation, and non-sensitive manifests in version control.
- When adding a new local dataset or credential, add a narrowly scoped `.gitignore` rule before downloading or generating it.
