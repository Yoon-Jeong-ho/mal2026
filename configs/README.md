# Experiment configuration templates

These JSON files are intentionally **not runnable**. Before a smoke run, copy
the relevant file to an ignored run directory and replace every `REQUIRED_*`
value with an immutable commit SHA or approved architecture value. The shared
config validator fails closed for branch/tag revisions, unset pooling, or empty
adapter target modules. Do not put credentials, prompts, writing text, IDs, or
W&B tokens in these files.
