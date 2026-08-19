---
license: apache-2.0
library_name: transformers
tags:
  - korean
  - text-regression
  - ensemble
  - lora
---

# MAL2026 R0 P1--4 prediction ensemble

This repository contains the custom LoRA adapters and three-axis regression
heads used by the MAL2026 metric-first submission candidate.  It does not
contain competition train/validation rows, writing text, identifiers,
predictions, generated row outputs, optimizer state, or credentials.

## Architecture

1. `skt/A.X-4.0-Light` generates score-blind rationales with the pinned
   `rank2_ax4_random1` adapter.
2. `Qwen/Qwen3-Embedding-8B` reads the prompt, essay, and three rationales.
3. Epoch 1--4 LoRA/head predictions are averaged uniformly as continuous
   values, clipped to `[1, 5]`, and rounded half-up for official integer output.
4. The optional final DPO adapter explains the emitted scores.

Pinned upstream revisions and local artifact checksums are recorded in
`bundle_complete.json`.  The Docker submission embeds the upstream base model
snapshots and performs no network download at container startup.

## Development evidence

- continuous macro RMSE: `0.5582937519`
- continuous macro Spearman: `0.6441959865`
- integer macro RMSE: `0.6158981882`
- integer macro Spearman: `0.5681974395`

These are previously exposed 400-row validation results and must not be
interpreted as an untouched hidden-test estimate.
