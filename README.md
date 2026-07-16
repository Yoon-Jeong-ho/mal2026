# MAL2026 Korean Writing Evaluation

Research code for reproducible Korean writing-score experiments. Restricted
writing data, generated rationales, checkpoints, and run artifacts are local
only and are intentionally excluded from version control.

See `docs/aihub_writing_evaluation_data.md` for the non-sensitive data record
and the experiment contract under `.omx/plans/` for the approved protocol.

## Local static/unit checks

Run the repository tests without exposing restricted inputs:

```bash
PYTHONPATH=src python -m unittest discover -v tests
```
