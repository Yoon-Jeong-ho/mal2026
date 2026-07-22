# MAL2026 Korean Writing Evaluation

Research code for reproducible Korean writing-score experiments. Restricted
writing data, human feedback, checkpoints, and run artifacts are local only
and are intentionally excluded from version control.

See `docs/aihub_writing_evaluation_data.md` for the non-sensitive data record.
The maintained standard Trainer/TRL/vLLM matrix is documented in
`docs/standard_experiment_matrix.md`.  A separately declared, score-blind
RLAIF/GRPO continuation uses maintained TRL `GRPOTrainer`, a local vLLM policy
rollout server, and a local Qwen judge; its reproducibility record is
`docs/experiment_records/rlaif_grpo_prompt_ensemble_v7_20260722_017.md`.

## RLAIF/GRPO prompt-ensemble status (aggregate-only)

The study compares two fixed reward estimators for Korean rationale decoders:
`all5` averages five independently posed Qwen feedback-quality prompts, while
`random1` uses one deterministic prompt-form assignment per completion.  Both
are score-blind during training; the post-RL comparison is a frozen five-form,
ten-replication v6 LLM-as-judge evaluation on 400 held-out essays.  No raw
writing, rationale, identifier, checkpoint, or per-observation judge output is
tracked.

Completed full bundle comparisons show positive paired frozen-judge macro
deltas for both estimators:

| SFT decoder | `all5` delta (95% bootstrap CI) | `random1` delta (95% bootstrap CI) |
| --- | ---: | ---: |
| Midm-2.0-Base | +0.281633 [0.216633, 0.347800] | +0.232700 [0.166100, 0.301117] |
| A.X-4.0-Light | +0.420767 [0.348783, 0.496533] | +0.426100 [0.355700, 0.499650] |

All three requested axes have positive paired intervals in these completed
comparisons.  They are evidence that the fixed full 1,920-group RL population
should be retained, not evidence to select one estimator or reduce data before
the predeclared full model/task matrix completes.  An A.X content/`all5` run
that stopped at step 444 because of a scoreless vLLM response envelope is
preserved as a failed runtime artifact; a fresh lineage resumes only incomplete
tasks with a bounded identical-request recovery and never overwrites it.  See
the linked experiment record for exact commands, gates, aggregate metrics, and
current resume state.

## Local static/unit checks

Run the repository tests without exposing restricted inputs:

```bash
PYTHONPATH=src python -m unittest discover -v tests
```
