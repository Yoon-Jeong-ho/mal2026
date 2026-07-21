# Train-only RLAIF/GRPO prompt-ensemble study — 2026-07-22

- **Status:** the static/data-contract gates and the reward-server health gate
  passed.  The first actual native-Transformers GRPO update failed its declared
  policy-JSON hard gate (0/4 parse-valid, hence 0 judge calls), so no full arm
  or validation comparison has started.  This is a preserved negative result;
  it does not alter the frozen v6 evaluator or replace any prior
  SFT/score-regression result.
- **Authorization:** the current user explicitly authorized additional training
  of the learned rationale decoders, use of the local Qwen judge as a reward or
  discriminator, the two reward conditions below, frozen evaluation, and GPUs
  0--3.  The user also explicitly requested literature/code investigation
  before implementation; that investigation preceded this task card.
- **Question:** starting at a frozen SFT decoder, does train-only
  LLM-as-a-judge reinforcement learning improve the fixed score-blind
  rationale-quality evaluation, and does averaging the five judge prompt forms
  outperform drawing one prompt form per generated rationale?

## Scientific contract

### Models and order

The first completed evaluation-eligible decoder is the all-axis
`midm2_base/bundle` adapter, selected previously among the three all-axis
decoders by the fixed v6 macro judge mean (3.867967).  It is the required first
preflight and full result.  The full declared matrix then covers every completed
SFT decoder as independent continuations: three bases (`midm2_base`,
`ax4_light`, `phi4_mini`) times the bundled task plus the three single-axis
tasks.  Thus there are 12 source adapters and two fresh sibling RL adapters per
source adapter.

Bundled systems receive a three-axis macro reward and macro evaluation.  A
single-axis system receives only its declared axis score/reward/evaluation; it
is not silently converted into a bundled system.  This lets the requested
"other models" stage cover all learned decoders while keeping their original
output contracts intact.

Every arm starts from the same immutable SFT adapter for its base/task, not
from another RL arm.  The reference adapter is an immutable in-memory copy of
that same SFT adapter.  No validation output, checkpoint, or prompt is used to
choose an arm, checkpoint, model, or hyperparameter.  The final declared
checkpoint is the last completed update.

### Inputs and isolation

- Policy/reward updates read only canonical `eval/train.jsonl` through the
  score-blind rationale loader.  They use exactly 1,920 train essay groups:
  the deterministic 80-group contrastive-calibration holdout is recomputed in
  memory, bound to its recorded opaque-set digest, and excluded before any
  rollout.
- The train source and API candidate checksums remain
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737` and
  `d69dd9bc349c117a55b4bf652507135f084c9045d95d5939da3980223893aecf`.
  Source writing scores and API-candidate scores are neither read nor
  prompted.  The policy receives only the task, essay, and rationale-only
  output contract.
- The 400 canonical validation essays remain frozen.  Existing exact v6
  baseline reports are verified and reused for pre-RL values; new RL outputs
  are generated once at the declared deterministic decode setting and scored
  only after training.  These are descriptive validation comparisons because
  this validation split was previously exposed for decoder selection, not a
  newly untouched test claim.
- Restricted rollouts, judge observations, generated rationales, opaque IDs,
  adapters, logs, and checkpoints stay in ignored `data/processed/` or
  `outputs/` roots.  Tracked records contain aggregate counts, hashes, and
  metrics only--never essays, rationales, prompts, identifiers, labels, or
  score observations.

### Policy optimizer

The maintained local `trl==0.29.1` `GRPOTrainer` is used with LoRA policy
continuation.  `GRPOTrainer` is chosen because this experiment explicitly
samples a group of alternatives from the same essay.  To avoid amplification
from a small discrete judge-score standard deviation, the GRPO group baseline
is retained but `scale_rewards="none"` and `loss_type="dr_grpo"` are fixed.
The initial SFT adapter is copied as the PEFT `ref` adapter and a nonzero
`beta=0.02` applies a small KL constraint.  Other fixed settings are one epoch,
four generations per essay, temperature 0.7, top-p 0.95, a 512-token hard
completion cap, learning rate `1e-6`, float32 base/LoRA compute with TF32,
gradient checkpointing, and fixed seed `2026072209`.

The online policy generator uses Transformers inside TRL (`use_vllm=false`).
The installed TRL emits an explicit compatibility warning for its integrated
vLLM path with local `vllm==0.25.1`; that unsupported path is therefore not
used.  vLLM remains the high-throughput judge server and is used for the final
generation/evaluation stages.

For the full run, policy DDP uses physical GPUs 1,2 and the Qwen reward server
uses data-parallel replicas on physical GPUs 0,3.  This uses only the
user-authorized 0--3 set and avoids colocating the trainer with the judge.
With two policy ranks, per-device batch 8, four completions, and
`generation_batch_size=64` (`steps_per_generation=4`), all 1,920 eligible
train essays form complete groups exactly: 120 rollout batches and 480 updates
per arm.  A rollout batch has 16 essays x four completions; it queues 320
all-five or 64 random-one judge requests.  This is deliberately batched while
keeping only four optimizer updates between policy rollouts.  The GPU0 smoke
uses one update and the corresponding complete group, followed by an
actual-input Midm one-update gate before a full arm.

### Reward firewall and the two arms

The new train-only reward protocol copies the already validated Qwen
model/revision, rubric axes, five prompt form texts, score-blind essay/rationale
presentation, and strict score-only JSON response schema from v6.  It has a
new configuration, training-only seed schedule, and attestation; the v6 config
is not modified or used as a training configuration because it explicitly
prohibits SFT/DPO/GRPO use.

Generated text is parsed as exactly `rationale-only-v1` with exactly the
requested axes, no extra fields, nonblank Korean rationale strings, and no more
than 192 characters per field.  It is canonicalized before judge submission.
Invalid or capped policy text receives deterministic reward -1 and is never
sent to the judge.  A judge transport/envelope/schema failure is an execution
failure after bounded retries, not a low-quality label.  There is no hidden
length, writing-score, or candidate-score reward.

For a valid completion, each judge response is reduced to the arithmetic mean
of the requested axis scores and mapped from 1--5 to `[-1,1]` by `(mean-3)/2`.
The two predeclared arms differ only in the estimator:

1. **`all5`:** query every one of the five existing prompt forms once, using a
   deterministic distinct training seed; reward is their equal-weight mean.
2. **`random1`:** for every individual sampled completion, select one of the
   five forms uniformly with a deterministic hash of run seed, arm, opaque
   source key, rollout/update, and generation index; query it once and use its
   mapped score.  Only aggregate form counts are retained.

Consequently `all5` intentionally spends five times as many judge calls per
completion.  It is a variance/cost comparison requested by the user, not an
equal-compute claim.  Both arms use the same policy samples, decoding bounds,
reference, data ordering, and number of policy updates.  Aggregate reward
means/standard deviations, form counts, parse rate, judge failures, clipped
rate, KL, entropy, and zero-reward-standard-deviation group fraction are
recorded.  A zero-variance fraction of 80% or more, parse validity below 98%,
any judge failure, nonfinite optimizer metric, or reference-integrity failure
is a hard scientific gate: preserve the negative artifact and do not proceed
to that arm's validation comparison.

### Evaluation and anti-hacking checks

The primary post-RL comparison uses the unchanged v6 evaluation protocol for
both the original SFT adapter and every final RL adapter: deterministic policy
generation with the strict 192-character schema bound, then the Qwen DP=4
score-only judge with all five forms x ten fixed seeds (50 observations per
rationale).  Report requested-axis and macro raw 1--5 means, paired
essay-level delta, paired bootstrap 95% interval, win/tie/loss, prompt-form
range, parse validity, completion length, and zero transport/schema failures.
The pre-RL baseline must be bound to the identical source adapter and fixed v6
config hash before reuse.

Because the training reward and primary evaluator are related proxies, each arm
also runs a train-only post-training control check on the reserved calibration
groups: a canonical grounded completion must not be outscored by its generic,
wrong-axis, or cross-essay control under the reward endpoint.  This audit is
not used to select or retune a policy; a failure is preserved as a reward-hack
warning even if the v6 mean rises.

## Test ladder, recovery, and completion predicate

1. Static parser/projection/config/hash tests, including random-form
   reproducibility and no-score-access tests.
2. GPU0 Qwen reward-server health plus synthetic score-schema request; then a
   GPU0 policy one-update synthetic/actual-input Midm preflight with a judge on
   a disjoint authorized GPU.  No validation text is opened.
3. Midm bundled `all5` and `random1` actual-input one-update gates, then their
   two full sibling arms and frozen v6 post-evaluations.
4. The same immutable recipe runs every remaining source adapter/arm.  A
   performance decrease is a recorded null result, not a reason to retune or
   skip the remaining models.
5. Aggregate-only final verifier checks the 24 arm completion records, exact
   input/model/config hashes, final adapters, all required evaluation reports,
   and the declared metrics.  A scoped commit contains only reproducibility
   code/config/tests and this record; ignored artifacts are never staged.

Transport, server-launch, parsing, serialization, template, or resource setup
failures may receive up to three same-stage repairs.  A memory-only reduction
from the declared batch to 4 is permitted solely if the full declared batch
cannot enter a first forward/backward update; it retains complete groups and
the same one-epoch prompt coverage, is recorded as a deviation, and is not
chosen from validation.  A failed scientific gate is not silently retuned.

The completion predicate is: every declared source adapter has both fresh arms
completed or a preserved declared hard-gate negative result; all successful
arms have a fixed-v6 post-RL evaluation; and an aggregate comparison reports
whether each arm's judge score rose, fell, or was indistinguishable from its
bound SFT baseline.

## Pre-execution evidence

- Three independent read-only investigations preceded implementation: literature
  favored a critic-free RLOO alternative for discrete rewards; local-stack and
  risk reviews favored maintained GRPO for the requested same-essay candidate
  groups.  The declared no-standard-deviation-scaling GRPO choice preserves the
  group baseline while addressing the discrete-score concern.  All agreed that
  local TRL/vLLM integrated generation is unsupported at the installed version,
  so the policy uses Transformers and vLLM is restricted to the Qwen judge and
  final batch stages.
- Static implementation gates passed on 2026-07-22: JSON validation, Python
  compilation of the new reward/training/evaluation/runner scripts, and four
  GPU-free unit tests.  Those tests bind the five score-blind forms, exact
  rationale-only JSON/no-score schema, 192-character field bound, and stable
  five-way random assignment.  They do not open train or validation text.
- The smallest actual-input data-contract preflight then loaded 2,000 canonical
  train records with `include_scores=false`, reconstructed the recorded
  80-group contrastive digest exactly, and exposed 1,920 policy rows with only
  `prompt`, opaque source key, and in-memory judge entry fields.  It read no
  writing/candidate score and opened no validation source text.
- The local runtime inventory is `trl==0.29.1`, `transformers==5.14.1`,
  `vllm==0.25.1`, `peft==0.19.1`, and `torch==2.11.0`; GPUs 0--3 were idle at
  the time of the minimal availability check.  No GPU 4--7 query or action was
  performed.

### Preserved setup recovery

- The first fresh runner lineage `20260722-001` completed the GPU0 Qwen model
  load and its synthetic strict score-schema request, then its parent terminal
  session ended before the first actual policy update could start.  The server
  was shut down cleanly; no train rationale, score observation, adapter, or
  validation text was produced.  Its ignored log/ledger is preserved.
- This is a launcher-session integration failure, not a scientific gate.  The
  runner now permits a fresh ignored runtime lineage (`20260722-002`), while
  refusing to overwrite the preserved `-001` logs or any arm/evaluation output.
  The model, prompt forms, data contract, reward, GPU scope, and hyperparameters
  are unchanged.
- Fresh lineage `20260722-002` repeated the successful GPU0 strict-schema
  health gate, then exposed a second launcher-only defect: the next reward
  server context tried to recreate an already existing ignored log directory
  without `exist_ok`. It failed before a second server launch or a policy
  update. The directory-creation repair is bounded to runner setup;
  `20260722-003` is the next fresh preserved lineage with the scientific
  protocol unchanged.
- In `20260722-003`, the all-five Midm bundled actual update reached the
  policy/reward boundary but every one of its four unconstrained native
  Transformers completions failed the already-declared exact
  `rationale-only-v1` parser.  The aggregate failure record reports
  `parse_valid=0`, `parse_invalid=4`, `judge_calls=0`, and deterministic
  reward `-1`; therefore it is not evidence about Qwen's rationale judgement
  or about the amount of training data.  The hard gate stopped the arm before
  an adapter export or validation generation.
- A separate GPU0 vLLM 0.25.1 structured-output feasibility gate then used the
  unchanged Midm SFT adapter, the same four score-blind train prompts, four
  samples per prompt, the same temperature/top-p/512-token bound, and a JSON
  schema expressing the existing exact rationale contract.  It passed
  **16/16** strict parser checks (`parse_valid_rate=1.0`) with no writing or
  candidate score access and no raw prompt/completion artifact.  This verifies
  the bounded parsing repair only; it does not yet establish online LoRA
  synchronization, reward variance, or a completed GRPO update.  Any full
  continuation must retain the declared reward/data/optimizer contract and
  record its rollout-backend implementation separately.

## Literature and implementation evidence

- GRPO's grouped relative-reward construction: [DeepSeekMath GRPO paper](https://arxiv.org/abs/2402.03300).
- Direct AI feedback rather than a separately trained reward model: [RLAIF](https://arxiv.org/abs/2309.00267).
- The alternative RLOO analysis found its leave-one-out baseline less sensitive
  to discrete-score normalization; this study instead fixes GRPO with no
  reward-standard-deviation scaling and records zero-variance groups.
- Maintained trainer interfaces: [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/grpo_trainer).
  Local source inspection additionally established that this installed TRL
  version warns against its vLLM integration with vLLM 0.25.1, hence the
  Transformers-policy/vLLM-judge split above.

## Transition ledger

```json
{"run_id":"rlaif-grpo-prompt-ensemble-v1","stage":"task_card","event":"start","failure_family":"none","repair_iteration":0,"evidence_ref":"this aggregate-only task card; three independent read-only literature/stack/critic reviews","command_ref":"new rlaif v1 runner","resource_scope":"GPUs 0-3","gpu_scope_authorization":"current-user explicit GPUs 0-3 authorization","decision":"implement static contracts"}
```
