# Solar-augmented bundle-rationale encoder protocol — predeclared run 001

Status: **protocol and runners verified; awaiting the official Solar runtime image**

Protocol authored against parent Git SHA
`f3884ef420c5597904f3c5da1e6f0442c5821a7e`; every executed run records its own
exact Git SHA and upstream artifact hashes.

## Fixed augmentation and rationale contract

1. Solar consumes only the 2,000 canonical training essays and emits exactly
   three variants per source: one each for content, organization, and
   expression degradation.
2. Every variant is still jointly rescored on all three axes using continuous
   quarter-step values in `[1,5]`. `average` is never generated or read.
3. The selected DPO rationale policy receives each generated essay and its
   Solar pseudo-score vector, then emits one bundled JSON containing all three
   rationales. It does not emit scores or improvement suggestions.
4. The rationale prompt explicitly classifies the score as synthetic Solar
   supervision. Human/reference scores and canonical validation rows are
   forbidden from this stage.
5. Axis-triplet rationale generation, evaluation, training, and selection are
   forbidden. The three Solar variants are separate edited essays, not three
   rationale-evaluation prompts for one essay.

The rationale prompt is
`configs/solar_augmented_bundle_rationale_prompt.v1.json`, SHA-256
`56269ad840cd35b5932577c3c5944d069301e8f07bdd8d2dd31f66f3d816932b`.
It limits each rationale field to 384 characters and the complete bundle to a
900-token generation budget. Before model launch, the runner audits all 6,000
rendered prompts against the 4,096-token context and fails closed on any
possible truncation.

## Encoder comparison

Both Qwen3-Embedding-8B and KURE-v1 restart from their respective completed
AI-Hub full-parameter backbone-plus-three-head artifacts and attach a fresh
LoRA. This avoids selecting an epoch on a train-internal development set that a
previous MAL refit has already seen.

The same deterministic source split as the non-augmented baseline is retained:

| Role | Original essays | Solar variants |
|---|---:|---:|
| epoch-selection train | 1,600 | 4,800 |
| epoch-selection dev | 400 | **0** |
| excluded variants linked to dev sources | — | 1,200 |
| selected-epoch refit | 2,000 | 6,000 |

Thus no augmentation of an internal-dev source can enter selection training.
The dev distribution remains original essays rather than synthetic examples.
Epochs 1--4 are compared by lower continuous three-axis macro RMSE, then higher
macro Spearman, then lower projected-integer RMSE, then earlier epoch. After
selection, refit restarts from the same AI-Hub initialization on all 8,000
records. Only then is the unchanged 400-row canonical validation set loaded
for one descriptive evaluation.

The fixed configurations are:

- `configs/augmented_rationale_aware_qwen3_embedding_8b.v1.json`;
- `configs/augmented_rationale_aware_kure_v1.v1.json`.

## Final winner fit

The final selector compares four already-completed frozen-validation results:

1. original-bundle Qwen3-Embedding-8B;
2. original-bundle KURE-v1;
3. original+Solar-augmented Qwen3-Embedding-8B;
4. original+Solar-augmented KURE-v1.

The winner is lower macro continuous RMSE, then higher macro continuous
Spearman, then the fixed candidate order. Its previously selected epoch count
is reused without retuning. If an original-only arm wins, final fitting uses
2,400 original train+validation rows. If an augmented arm wins, it uses those
2,400 rows plus the 6,000 train-only Solar variants, for 8,400 records. No
post-fit validation evaluation is reported because validation has become
training data.

The fixed selector is
`configs/final_rationale_aware_score_encoder.v1.json`. The final artifact is a
three-score-head plus LoRA trainable state bound to the selected model's
AI-Hub full-tuned initialization.

## Execution topology

`scripts/run_remaining_solar_encoder_pipeline.py` provides a durable sequence:

1. Solar TP4 generation, including its same-server one-row smoke;
2. selected-DPO one-GPU rationale smoke, then TP4 generation of all 6,000;
3. Qwen GPU0 one-update smoke, then DDP4 selection/refit/evaluation;
4. KURE GPU0 one-update smoke, then DDP4 selection/refit/evaluation;
5. final winner GPU0 one-update smoke, then DDP4 train+validation fit.

All phases use only physical GPUs 0--3. The runner does not install packages,
create an environment, pull a Docker image, access GPUs 4--7, or include row
data in tracked reports. A dry plan is available with:

```bash
PYTHONPATH=src:. .venv-standard/bin/python \
  scripts/run_remaining_solar_encoder_pipeline.py --dry-run
```
