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

## RLAIF/GRPO prompt-ensemble snapshot (aggregate-only, 2026-07-22)

The study compares two fixed reward estimators for Korean rationale decoders:
`all5` averages five independently posed Qwen feedback-quality prompts, while
`random1` uses one deterministic prompt-form assignment per completion.  Both
are score-blind during training; the post-RL comparison is a frozen five-form,
ten-replication v6 LLM-as-judge evaluation on 400 essays held out from RL
updates.  This split was previously used for user-authorized decoder selection,
so it is a descriptive fixed validation comparison rather than a newly
untouched test.  No raw writing, rationale, identifier, checkpoint, or
per-observation judge output is tracked.

Completed full bundle comparisons show positive paired frozen-judge macro
deltas for both estimators:

| SFT decoder | SFT macro | `all5` delta (95% bootstrap CI) | `random1` delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: |
| Midm-2.0-Base | 3.867967 | +0.281633 [0.216633, 0.347800] | +0.232700 [0.166100, 0.301117] |
| A.X-4.0-Light | 3.766183 | +0.420767 [0.348783, 0.496533] | +0.426100 [0.355700, 0.499650] |

All three requested axes have positive paired intervals in these completed
comparisons **under the fixed Qwen-v6 judge proxy**.  They are not independent
human-quality, RMSE/Spearman, or estimator-selection results: reward and
evaluation use the same judge/rubric family, prompt-form calibration variation
is material, and there is one training seed.  The evidence supports retaining
the fixed full 1,920-group eligible RL population, not selecting an estimator
or reducing data before the predeclared matrix completes.  The A.X
content/`all5` `-019` arm stopped at step 444 on a scoreless response envelope;
it is preserved without adapter/evaluation output.  A `-020` pre-update
recovery was also preserved, and a fresh `-021` continuation handles only
explicit vLLM internal-error retries while treating length finishes as terminal.
See the linked experiment record for exact commands, gates, aggregate metrics,
and current continuation state.

## Local static/unit checks

Run the repository tests without exposing restricted inputs:

```bash
PYTHONPATH=src python -m unittest discover -v tests
```
