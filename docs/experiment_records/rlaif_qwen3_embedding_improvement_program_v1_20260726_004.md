# Qwen3-Embedding score-regression improvement program v1

Status: **protocol frozen before execution** (2026-07-26)

## Question and fixed evidence boundary

This program tests whether the current Qwen3-Embedding-8B validation result can
be improved by separating the value of the essay, Qwen embedding instructions,
trait conditioning, multiple independently generated rationale views, and the
AI-Hub initialization method.  The final targets are exactly `content`,
`organization`, and `expression`; `average` is never a target in the rationale
stage.

The 400-row project validation split has already been inspected repeatedly.
Every result below is therefore descriptive model development evidence, not an
unbiased estimate of held-out generalization.  Arms and selection rules are
fixed in this record before any new validation result is read.

Canonical inputs are:

- pinned `Qwen/Qwen3-Embedding-8B` revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`;
- the AI-Hub 48,016-row prepared manifest
  `data/manifests/aihub_human_feedback_v1.json`;
- train/validation writing sources already bound by
  `src/mal2026/api_rationale_data.py`;
- the three complete, independently generated rationale sources already bound
  by `src/mal2026/rlaif_top3_encoder.py`.

Restricted prompts, essays, rationales, identifiers, row predictions, and
full-parameter model artifacts remain in ignored paths.  Public reports contain
only aggregate metrics, checksums, configurations, and deviations.

## Baselines

- **R0:** AI-Hub LoRA warm start, no embedding instruction, prompt + essay +
  `rank2_ax4_random1` three-axis rationale, epoch 1--12 sweep already completed.
  Epoch 3 is the existing descriptive winner by lower macro RMSE, then higher
  macro Spearman, then earlier epoch.
- **R-public:** public Qwen3-Embedding initialization result already completed;
  it is retained as a negative control and is not rerun.

## Predeclared rationale-stage arms

All newly trained arms use the same AI-Hub 48,016 **LoRA** warm start as R0,
seed `2026072601`, maximum length 2048, BF16, TF32, LoRA rank/alpha/dropout
16/32/0.05 over Qwen projection leaves, global batch 64 on GPUs 0--3, learning
rate `1e-4`, weight decay 0.01, warmup ratio 0.05, and exactly four epochs over
2,000 training essays unless an arm explicitly changes the example count.
Trainable LoRA and three-output head states are saved after every epoch.

1. **E0 (essay only):** prompt + essay, no rationale and no instruction.
2. **EI (essay + instruction):** E0 input prefixed with the exact Qwen-style
   instruction below.
3. **RI (rationale + instruction):** R0 input prefixed with the exact
   rationale-aware instruction below.
4. **T (trait-specific):** three examples per essay.  Each example contains
   the prompt, essay, one axis name, the fixed axis rubric, and only that axis's
   rationale.  One shared backbone and three independent scalar heads are used.
   The loss is computed only for the requested axis.  Four epochs over the
   expanded 6,000-example training set are used.
5. **M (multi-rationale):** three views per essay, one for each frozen top-three
   rationale source.  Training uses the expanded 6,000-example set.  Validation
   obtains one three-axis prediction per view and uniformly averages the three
   prediction vectors per essay before metrics.  This is a prediction ensemble,
   not an `average` label.

Exact instruction strings:

```text
Instruct: Predict the content, organization, and expression scores of a Korean student essay from 1 to 5 using the writing prompt and essay.
Query:
```

```text
Instruct: Predict the content, organization, and expression scores of a Korean student essay from 1 to 5 using the writing prompt, essay, and qualitative rationales.
Query:
```

Fixed trait rubrics:

- `content`: 과제 적합성, 중심 주장과 아이디어의 명료성, 근거의 구체성과 충실성을 판단한다.
- `organization`: 도입·전개·마무리의 구조, 문단과 논리의 흐름, 연결과 응집성을 판단한다.
- `expression`: 문장의 명료성과 자연스러움, 어휘와 문법, 맞춤법과 문체의 적절성을 판단한다.

## Existing-checkpoint combinations

Without retraining, R0 epochs 1--4 are combined in two fixed ways:

- **P1--4:** uniform arithmetic mean of the four prediction vectors per essay;
- **S1--4:** uniform arithmetic mean of corresponding floating-point
  trainable LoRA/head tensors, followed by one validation prediction.

No weights are fitted to validation labels.  Incompatible/non-floating tensors
make S1--4 fail closed and remain a negative result.

## Additional initialization arm: full AI-Hub tuning then rationale LoRA

**F-AIHUB** replaces only the AI-Hub initialization stage:

1. full-parameter Qwen3-Embedding-8B + four-output regression-head training on
   AI-Hub, using native PyTorch FSDP `FULL_SHARD` across GPUs 0--3, BF16, TF32,
   gradient checkpointing, learning rate `2e-5`, weight decay 0.01, warmup
   ratio 0.05, fused PyTorch AdamW, per-device batch 4, accumulation 4, and
   global batch 64;
2. selection on the canonical AI-Hub selection split at every 100 updates,
   maximum 2,200 updates, patience three, lower four-target macro MAE;
3. refit all 48,016 rows from the immutable public snapshot for the selected
   number of updates with the same optimization contract;
4. discard the refit's `average` head row, attach fresh LoRA rank 16 adapters,
   and continue on the R0 rationale input for four epochs with the rationale
   contract above.

The F-AIHUB arm is compared both to the completed AI-Hub-LoRA warm start and to
R0.  Full model checkpoints are ignored research artifacts and are not
committed.  If native FSDP cannot pass a real one-update distributed gate in the
installed environment, the failure is preserved and the arm stops; packages
or environments are not changed.

Selection retains aggregate evaluation history and the selected update count,
not repeated 16 GiB model snapshots; refit always restarts from the immutable
public snapshot, so selection weights are not a required input to refit.  A
single full state is materialized collectively after refit for LoRA continuation.

## Execution and selection rules

1. Run CPU/config/unit checks.
2. Run the smallest real GPU0 one-update gate for LoRA arms.
3. Run each fixed LoRA arm on GPUs 0--3 and evaluate all four epochs.
4. Run P1--4 and S1--4, then T and M.
5. Run a GPU0 construction/forward gate and a four-GPU one-update FSDP gate,
   then continue through F-AIHUB selection, refit, rationale LoRA, and
   evaluation if both pass.

Within an arm, the reported checkpoint is selected by lower three-axis macro
RMSE, then higher macro Spearman, then earlier epoch.  Across arms, the same
rule is used.  Per-axis RMSE/Spearman and organization failure modes are always
reported.  No arm is retuned after its validation result is observed.  The
target `0.421300` macro RMSE is a comparison reference, not a stopping-rule
license to search validation.

GPU authorization is the repository-default GPUs 0, 1, 2, and 3.  GPU0 is used
first for the LoRA gate.  Existing processes are never terminated, moved, or
assumed to belong to this experiment.

## Integration ledger note

The first runtime `20260726-004` stopped at the trait-specific GPU0 gate after
one successful optimizer update.  The expanded 12-view gate reports Trainer
epoch `1/3`, while the checkpoint callback incorrectly required integer epoch
1; training itself did not fail.  The failed output and log are preserved.
Recovery runtime `20260726-006` changes only the one-update gate's checkpoint
label from the rounded fractional Trainer epoch to checkpoint 1.  Full-run
data, optimization, epoch boundaries, metrics, and selection rules above are
unchanged.

Runtime `20260726-006` then completed the fixed epoch ensembles, essay-only,
and essay-instruction arms.  The rationale-instruction arm hit a real CUDA OOM
on its fifth update: the longer input left only 112 MiB free on GPU0 while
per-device batch 4 was active.  The negative run is preserved.  Recovery
runtime `20260726-007` reuses the completed arm reports without retraining and
runs only rationale-instruction, trait-specific, and multi-rationale.  For
these longer-input arms it changes per-device batch/accumulation from 4/4 to
2/8, preserving global batch 64, optimizer, update count, examples, epochs,
seed, and selection rule.  The allocator uses expandable segments to reduce
fragmentation; this is a runtime-only integration recovery.

Before the full rationale stage of runtime `20260726-009` started, its fixed
batch schedule inherited the same longer R0 input and the already observed
batch-4 OOM boundary.  That stage therefore uses per-device batch 2 and
accumulation 8 rather than 4/4.  Global batch 64, examples, update count,
optimizer, seed, epochs, and selection rule are unchanged.  This recovery was
recorded before any full-rationale optimizer update.

Full-arm runtime `20260726-008` passed the GPU0 full-regressor construction
gate, then stopped on the first FSDP2 update before producing any result.  The
FSDP BF16 policy cast the regression head to BF16 while the common pooling path
provided FP32 head input.  Runtime `20260726-009` preserves the failure and
changes only the head boundary: normalized input is cast to the actual
head-parameter dtype for GEMM and logits are immediately restored to FP32 for
MSE/metrics.  Data, parameters trained, optimizer, batch, update schedule, and
selection protocol are unchanged.

Runtime `20260726-009` completed FSDP selection, the selected 100-update refit,
and the real GPU0 rationale-LoRA update.  Its preflight evaluation then stopped
before model construction because the evaluation CLI loaded the training
configuration with the training-only fresh-output assertion.  Runtime
`20260726-010` preserves that negative log, loads evaluation configurations in
read-only existing-output mode (the evaluation function already enforced this
mode), reuses the completed refit and preflight update, and continues with the
unchanged full rationale protocol.

Runtime `20260726-010` reached the actual four-row prediction smoke, where the
one-update checkpoint produced constant ranks and the shared metric helper
correctly rejected undefined Spearman.  Runtime `20260726-011` preserves this
second negative log and treats that smoke as a finite-output/RMSE gate with
undefined Spearman recorded as null.  The 400-row full evaluation still
requires finite Spearman on every axis and is otherwise unchanged.  The empty
evaluation directory created by the failed Trainer initialization is preserved;
runtime 011 writes its evaluations to fresh, uniquely suffixed directories.

Runtime `20260726-011` completed finite predictions and RMSE calculation, then
stopped because `TrainingArguments` had already created the initially fresh
evaluation directory before the aggregate writer attempted a non-idempotent
`mkdir`.  Runtime `20260726-012` preserves that empty directory and failure log,
uses a new evaluation suffix, and permits only that expected empty-directory
transition before the atomic aggregate write.  Metric and training contracts
remain unchanged.

Runtime `20260726-012` completed the entire four-epoch DDP4 rationale training.
During the 400-row sweep, a checkpoint produced constant prediction ranks, so
Spearman was mathematically undefined and the strict shared metric helper
stopped the evaluation.  Runtime `20260726-013` reuses all completed training,
preserves the failed evaluation directory/log, and reruns only evaluation.
Undefined per-axis Spearman values are recorded as null; their macro is null and
cannot win a selection tie.  RMSE remains defined and primary, and no model,
data, checkpoint, or prediction is changed.

## Final aggregate result

Runtime `20260726-013` completed at `2026-07-27 01:48:38 KST`.  Its aggregate
report is
`outputs/aggregate-reports/qwen3-full-aihub-then-rationale-lora-v1-recovery-20260726-013.final-summary.json`
(SHA-256
`f841f1c64c444573ffdab7c736b76d45249c9b5e193179c42e6adc9a9839176a`).
This output is ignored and contains aggregate metrics only.

The best completed result remains the uniform R0 epoch-1--4 prediction
ensemble:

| result | content RMSE | organization RMSE | expression RMSE | macro RMSE | macro Spearman |
|---|---:|---:|---:|---:|---:|
| R0 prediction ensemble | 0.502760 | 0.683891 | 0.488231 | **0.558294** | **0.644196** |
| prior AI-Hub-LoRA R0, epoch 3 | 0.512864 | 0.695580 | 0.499289 | 0.569245 | 0.625288 |
| full AI-Hub then rationale LoRA, epoch 4 | 2.578675 | 2.685845 | 2.973646 | 2.746055 | undefined |

The ensemble improves macro RMSE over the prior R0 checkpoint by `0.010951`,
but remains `0.136994` above the `0.421300` reference.  Organization remains
the largest ensemble error.  Among the fixed input ablations, none displaced
the ensemble; multi-rationale expansion was worse (`0.591079` macro RMSE).

The added full-parameter arm is a clear negative result.  Its four validation
macro RMSE values decreased monotonically (`2.979795`, `2.849175`, `2.769825`,
`2.746055`) but remained unusable.  Epochs 3 and 4 each had one constant-rank
axis, so three-axis macro Spearman is correctly undefined.  The rationale stage
used only `content`, `organization`, and `expression`; `average` was not a
target.

The most likely cause is an optimization/selection-contract failure, not a
benefit or defect of rationale conditioning.  This is an inference from the
following aggregate evidence: the newly initialized head began outside the
score range; unbounded raw selection loss improved from `40.36` to `31.70`,
while clipped four-axis macro MAE stayed exactly `2.437` at steps 100, 200, 300,
and 400.  Early stopping therefore selected step 100 from a saturated clipped
metric, and the 100-update refit handed a collapsed state to LoRA.  The LoRA
stage reduced its own loss trend but could not repair that initialization in
four epochs.

Decision: reject the full-parameter arm and retain the R0 prediction ensemble
as the current development winner.  Any follow-up that changes target scaling,
head initialization/learning rate, bounded output parameterization, or the
unclipped selection metric is a new scientific protocol and was not launched
after observing this result.  Validation has already been used repeatedly, so
all rankings remain descriptive development evidence rather than held-out
generalization estimates.
