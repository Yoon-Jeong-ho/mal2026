# MAL2026 Korean Writing Evaluation

Research code for reproducible Korean writing-score experiments. Restricted
writing data, human feedback, checkpoints, and run artifacts are local only
and are intentionally excluded from version control.

See `docs/aihub_writing_evaluation_data.md` for the non-sensitive data record.
The maintained standard Trainer/TRL/vLLM matrix is documented in
`docs/standard_experiment_matrix.md`.  A separately declared, score-blind
RLAIF/GRPO continuation uses maintained TRL `GRPOTrainer`, a local vLLM policy
rollout server, and a local Qwen judge; its reproducibility record is
`docs/experiment_records/rlaif_grpo_prompt_ensemble_v8_20260722_022.md`.

## RLAIF/GRPO prompt-ensemble snapshot (aggregate-only, completed)

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
| Midm-2.0-Base | 3.867967 | +0.306317 [+0.237733, +0.376617] | +0.321133 [+0.252633, +0.392667] |
| A.X-4.0-Light | 3.766183 | +0.417883 [+0.349500, +0.489667] | +0.420850 [+0.349567, +0.492683] |
| Phi-4-mini | 3.049750 | +0.768150 [+0.685917, +0.851350] | +0.668567 [+0.583950, +0.756067] |

All three requested axes have positive paired intervals in these completed
comparisons **under the fixed Qwen-v6 judge proxy**.  They are not independent
human-quality, RMSE/Spearman, or estimator-selection results: reward and
evaluation use the same judge/rubric family, prompt-form calibration variation
is material, and there is one training seed.  The evidence supports retaining
the fixed full 1,920-group eligible RL population, not selecting an estimator
or reducing data before the predeclared matrix completes.  Phi-4-mini
content-only training also improves the requested content target
(+0.529600 / +0.477800), but reduces organization transfer; its
organization-only target intervals are inconclusive (+0.005850 / +0.040300).
Phi expression-only training improves all three axes, including its requested
expression target (+0.960350 / +0.846250).  In completed A.X
single-axis continuations, both arms improve the requested content target
(+0.327450 / +0.355600), but organization transfer is negative; organization
training improves its requested target (+0.250650 / +0.110250) and all three
diagnostic axes; and expression training improves its requested target
(+0.295400 / +0.279800).  Midm content-only training also improves its
requested target (+0.317400 / +0.281550), but reduces organization transfer.
Midm organization-only training improves its requested target
(+0.143450 / +0.107150).  These are task-targeted results, not global
estimator-selection evidence.

The active v8 matrix preserves the initial v7 and A.X rollout failures under
ignored runtime roots, then uses a fresh lineage with aggregate-recorded repair
rules.  It completed all 24 arms and every completed frozen evaluation has
20,000/20,000 valid score observations with zero abstentions and zero evaluator
transport/schema failures.  For the follow-on three-rationale encoder study,
only the three highest complete bundle adapters are selected—Midm `random1`,
A.X `random1`, and A.X `all5`; their outputs were not averaged or ensembled.
The completed separate top-three encoder protocol uses Qwen2.5-7B once per
independent rationale source and trains/evaluates only `content`,
`organization`, and `expression`.  Its best source is A.X `random1` (validation
RMSE: 0.590273 / 0.769708 / 0.629532; the three-axis diagnostic RMSE is
0.663171).  The diagnostic is not an `average` score target or prediction; see
`docs/experiment_records/rlaif_top3_encoder_v1_20260725_001.md` for the full
per-axis result and protocol caveat.

The requested follow-up now uses the actual embedding-tuned
`Qwen/Qwen3-Embedding-8B` snapshot with the same A.X `random1` rationales.  A
public-base initialization did not learn a usable scorer under the fixed
schedule (three-axis diagnostic RMSE `2.062436`), whereas the model previously
trained on 48,016 eligible AI-Hub feedback rows reached content / organization
/ expression RMSE of `0.531378 / 0.710936 / 0.531244` (`0.591186` three-axis
diagnostic RMSE). That is 10.85% lower than the directly comparable Qwen2.5
diagnostic RMSE, but it remains above the requested `0.421300` level.  The old
fourth `average` head was discarded before continuation; neither arm trains or
evaluates an average target.  See
`docs/experiment_records/rlaif_qwen3_embedding_comparison_v1_20260726_001.md`.

The custom vLLM rollout is required because the installed TRL/vLLM versions do
not support their built-in integration; it
is common to both arms, but lacks an external sampled-logprob TIS/MIS
correction.  See the linked experiment record for exact commands, gates,
aggregate metrics, framework caveat, and current continuation state.

## Local static/unit checks

Run the repository tests without exposing restricted inputs:

```bash
PYTHONPATH=src python -m unittest discover -v tests
```
