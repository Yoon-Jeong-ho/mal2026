# MAL2026 Korean Writing Evaluation

Research code for reproducible Korean writing-score experiments. Restricted
writing data, human feedback, checkpoints, and run artifacts are local only
and are intentionally excluded from version control.

See `docs/aihub_writing_evaluation_data.md` for the non-sensitive data record.
The supported training and evaluation entry points are documented with their
standard Trainer/TRL/vLLM implementation; legacy bespoke runners are not part
of this repository.

## Local static/unit checks

Run the repository tests without exposing restricted inputs:

```bash
PYTHONPATH=src python -m unittest discover -v tests
```
