# OpenAI explanation-data distribution audit: 20260720-001

**Status:** completed aggregate-only audit. No GPU was used; no candidate
selection, SFT, DPO, GRPO, judge invocation, or API call was performed.

## Reproducibility

- Command: `PYTHONPATH=src python scripts/audit_openai_distribution_metadata.py`
- Script: `scripts/audit_openai_distribution_metadata.py`
- Configuration: `configs/openai_distribution_utilization.v1.json`
- Report: `data/reports/openai_distribution_audit_v1.json`
- Inputs: canonical prepared-data manifest plus restricted `eval/train` and
  `eval/validation` files, hash-bound in the resulting aggregate report.
- Hardware: CPU-only metadata pass; no CUDA environment was set.
- Git SHA at execution: `86902f1e3a077b1178d1297a1dcccf10e929453d`.
  The repository was already dirty before this audit; no unrelated changes
  were included.

## Observed aggregate results

The API population has 2,000 train rows and 400 evaluation-only validation
rows; task/prompt metadata is available for every row. The validated API batch
has 7,200 accepted candidates (three per source row) with zero recorded
schema/grounding, mapping, duplicate, missing, or rejected candidates in the
strict aggregate quality-control report.

The report contains fixed component bands, all 64 joint component-score cells,
fixed length bins, opaque task/prompt metadata coverage, source/train/
validation divergence and overlap diagnostics, sparse/zero-cell counts, and
the split-specific prepared-data audit. It intentionally makes no
cross-population distribution claim between the prepared and API populations.

## Gate state and next stage

Judge-v3 has not passed, and the v3 pilot configuration itself prohibits a
selection artifact. `scripts/validate_openai_distribution_utilization.py`
therefore reports `blocked_pending_judge_v3`. The next executable, non-GPU
stage is the same validator after a future aggregate judge-v3 manifest exists;
after a pass, a separately versioned production-selection protocol must be
reviewed and run before any selected target or SFT artifact can exist.
