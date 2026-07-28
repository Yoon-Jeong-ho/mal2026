# Public-spec-aligned score and rationale program v1

Status: **API/SFT/AI-Hub rationale comparisons complete; RL fail-closed at
the frozen safety gate; public-spec score-only integer pretraining and matrix
evaluation running** (updated 2026-07-28)

## Prompt-provenance terminology

The supplied PDFs publish the participant input/output contract, scoring
criteria, judge model/quantization, and judge evaluation dimensions.  They do
not publish an organizer-authored verbatim participant system prompt or the
hidden judge prompt.  This record therefore uses these precise terms:

- **public-spec-aligned participant prompt**: the fixed repository prompt that
  implements the published axes, integer score range, and final JSON shape;
- **rationale-only adaptation**: the same published criteria with the actual
  predicted integer score supplied as immutable context and all score output,
  re-scoring, and improvement advice prohibited;
- **fixed single proxy-judge prompt**: the only judge prompt used in this
  program, aligned to the published 12 dimensions but not claimed to be the
  unavailable organizer wording.

The exact Q4_K_M model and pinned llama.cpp runtime are official-runtime
matches; that fact does not turn reconstructed prompt wording into a verbatim
official prompt.

Accordingly, the new API corpus, all new rationale SFT arms, the AI-Hub to API
continuation, and the declared downstream RL/score runners use this same
public-spec-aligned contract.  Historical runs remain immutable comparison
artifacts and were not rewritten in place.  No artifact in this record is
described as using an unavailable organizer-authored prompt.

## Authorization and question

The current user authorized continued training and evaluation aligned to the
two supplied task PDFs, including separately trained score and rationale
models, joint-output comparison, and new OpenAI rationale generation if it is
needed.  Docker packaging is explicitly deferred.  This program asks which
deployable pipeline best balances official integer score accuracy/ranking and
the exact Q4_K_M LLM-judge rationale criteria.

The 400-row validation split has already been exposed repeatedly.  Every
validation result in this program is descriptive development evidence, not an
unbiased held-out generalization estimate.  No validation label may be used to
fit a threshold, choose an API target, train/reward a model, or revise this
protocol after a result is observed.

## Canonical contract and inputs

- Task PDF SHA256:
  `125896cdeb0862816b41df4e02e3972c85b1e36ee999b3fd3644e2f8f5bf5080`.
- Docker-rule PDF SHA256:
  `40d35b56af956f76adc52acff19ccf8c30425ebdd5208a3fb7e298c2ad3be15e`.
- Canonical writing source checksums remain those in
  `src/mal2026/api_rationale_data.py`: train
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
  (2,000 rows) and validation
  `0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4`
  (400 rows).
- Participant output is one JSON object containing `content`,
  `organization`, and `expression`; every axis contains an integer score in
  `[1,5]` and a nonblank Korean rationale.  Official inference is
  temperature `0`, seed `42`.
- The exact published judge model is `Qwen3.6-35B-A3B` Q4_K_M GGUF.  Our
  frozen single proxy prompt gives it the
  candidate's **actual emitted integer score**, rationale, prompt, and essay;
  it never receives a human/reference score and does not re-score the essay.
  It emits 12 integer cells: three axes by `domain_match`,
  `score_rationale_consistency`, `specificity`, and `groundedness`, each with
  evidence.
- Exact local GGUF SHA256:
  `b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`;
  pinned llama.cpp revision `571d0d540df04f25298d0e159e520d9fc62ed121`.

The earlier FP8 frozen-v6 score-blind judge and GRPO-v8 reward remain preserved
as proxy experiments.  Their means, abstention rates, and "top-three" order
are not official metrics and cannot select a final system.

## Fixed score projection and metrics

All continuous regression candidates use one immutable deployment projection:

1. clip to `[1,5]`;
2. round exact halves upward (`ROUND_HALF_UP`);
3. emit a JSON integer.

No validation-fitted thresholds are allowed.  For each candidate report
continuous diagnostics and official integer per-axis RMSE and tie-aware
Spearman, then the unweighted three-axis means.  The official ranking combines
submission ranks with weights RMSE 45%, Spearman 45%, and LLM judge 10%; a
standalone local run cannot reconstruct ranks against other submissions.
Therefore local selection reports the RMSE/Spearman/judge Pareto frontier
rather than inventing a scalar leaderboard score.

## Fixed pipeline matrix

### A0: no-training audit

Recompute predictions rather than derive them from aggregate reports:

- `S-draft`: current R0 epoch-1--4 uniform prediction ensemble and state soup;
- `S-essay`: essay-only and fixed-instruction Qwen3-Embedding epochs 1--4;
- `S-rationale`: rationale-instruction epochs 1--4.

Persist row predictions only beneath the ignored restricted root.  Public
artifacts contain aggregate metrics and checksums only.

### A1: score-first separated pipeline

`prompt + essay -> S-essay -> three integer scores -> R-score -> final JSON`.
The score encoder is authoritative.  `R-score` cannot change or re-predict a
score; the final composer copies the encoder integers.  This is the primary
non-cyclic deployment candidate.

### A2: draft-score-rewrite separated pipeline

`prompt + essay -> score-blind draft -> S-draft -> integer scores -> R-rewrite
-> final JSON`.  This preserves the strongest continuous-score screening arm
while adding a final score-consistency rewrite.  It is compared with A1 rather
than silently treated as equivalent because it is slower and cyclic designs
are prohibited.

### A3: joint comparison

One decoder maps `prompt + essay` directly to the exact official three-axis
score+rationale JSON at temperature 0 and seed 42.  It is a comparison arm,
not assumed superior to the separated systems.

## Teacher and training rules

The user explicitly authorized a new public-spec-aligned API corpus with three
candidates for every one of the 2,000 training essays.  The resulting 6,000
restricted rows were validated with complete `{1,2,3}` candidate coverage,
zero API failures, strict three-axis integer score/rationale parsing, and
checksum
`a1791c418c79c0b76399ddb993e862f34209c2da95b0c13f7cda87f403a24e4c`.
No validation essay was sent to the API.

Each stored API `score` is the API candidate's own direct prediction, not a
human/reference score.  It is used only to condition the rationale that came
from the same candidate, preserving score-rationale consistency without label
leakage.  All three candidates are used for every rationale SFT arm.  The
bundled arm emits all three rationale fields at once; the content,
organization, and expression arms each emit only their assigned rationale.
Every target is rationale-only strict JSON: score output, re-scoring, and
improvement advice are excluded.

At validation/inference, the separated pipeline supplies only the deployed
score encoder's actual emitted integer score.  The final composer copies those
integers unchanged.  The API candidate scores are not substitutes for human
score labels in the later score-model matrix.

The post-download aggregate audit found 2,000 candidates for each candidate
index, no duplicated full three-axis rationale vector within an essay, and the
following diversity of three emitted score vectors per essay: 906 essays had
one unique vector, 917 had two, and 177 had three.  Thus all candidates are
retained as distinct rationale supervision even when their integer score
vectors agree.  Mean rationale character lengths for candidates 1/2/3 were,
respectively: content 202.1/277.9/240.7, organization 176.2/236.1/207.6, and
expression 182.5/251.4/214.1.

The corrected A.X tokenizer audit found no truncation at the frozen SFT length
of 3,072 tokens: maximum complete rendered lengths were 1,422 for bundled,
1,127 for content, 1,102 for organization, and 1,127 for expression.  The
first audit artifact incorrectly measured the two keys of a `BatchEncoding`
instead of `input_ids`; it remains preserved and is explicitly marked invalid.
The corrected aggregate is
`outputs/official-prompt-alignment-v1/sft-data-audit/official-rationale-sft-token-audit-002.json`.

For the subsequent full-parameter pretraining arm, the closest already
downloaded corpus is AI-Hub 71819 argumentative writing: 16,010 upstream
Training rows with per-axis analytic human feedback.  Its deterministic
projection excludes holistic and task/improvement feedback, maps only the
axis-matched analytic fields, and half-up projects the three component scores
to integers.  Aggregate lineage is frozen in
`data/manifests/aihub_argumentative_official_rationale_v1.json`; no new
download is required.  The A.X tokenizer audit covers all 16,010 bundled and
48,030 axis-triplet rendered examples.  Neither structure exceeds 2,048
tokens (maxima 1,618 and 1,437 respectively), so the full-parameter stage uses
the smaller verified 2,048-token cap rather than an unnecessarily large
context allocation.

The first structure comparison uses the predeclared final checkpoint after two
epochs; epoch checkpoints are preserved and the 400 validation labels do not
choose an epoch.  Rationale SFT uses completion-only loss.  Any later joint
SFT must emit the exact participant JSON.  Average score is never a target.

## Initial rationale-structure result

All four A.X rationale SFT arms completed on all 6,000 API candidates: one
bundled three-axis model and the content, organization, and expression
single-axis models.  Every validation generation arm produced 400/400 strict
outputs.  The same authoritative integer score file was copied unchanged into
both participant files; both composers reported zero score mismatch and
400/400 strict final participant JSON.

Under the single frozen proxy prompt and exact Q4 runtime, the bundled model
scored `4.986250` macro with a `4.9525` worst cell, while the three independent
axis models scored `4.9891667` macro with a `4.9550` worst cell.  The frozen
macro-then-worst-cell rule therefore selects `axis_triplet` for the AI-Hub
arm.  This is only a mechanical selection, not strong evidence of a real
quality difference: 4,746/4,800 bundled cells and 4,750/4,800 axis-triplet
cells received score 5; 341/400 paired essay macros tied, with 30 favoring
axis-triplet and 29 favoring bundled.  The judge is substantially saturated on
these candidates.  Aggregate evidence is stored in
`outputs/official-prompt-alignment-v1/structure-comparison/official-rationale-structure-comparison-v1-20260727-001/`.

The selected AI-Hub path trained A.X full parameters for one epoch on the
48,030 axis-projected argumentative feedback examples, then trains three
separate LoRA continuations on the same 6,000 API candidates used by the
no-AI-Hub arms.  GPU0 and four-GPU FSDP2 one-update gates passed.  Two
pre-training integration negatives are preserved: torchrun world-size was
checked before Accelerate initialized the process group, and Transformers 5
rejected simultaneous generic and FSDP activation checkpointing.  Repairs
only changed launcher-state detection and selected the maintained FSDP2
activation-checkpointing path; data, model, prompts, targets, optimization,
and selection were unchanged.

The full-parameter run completed at step 1,501 with loss `1.2437230634` on
48,030 records; its final-model SHA256 is
`2c1295cc97f35768b4dab06e7a85c52b4aae8ca0be5d96aa939af9948091ebb4`.
The content, organization, and expression API LoRA continuations each used all
6,000 candidates for 750 optimizer steps and completed with losses
`1.3120444883`, `1.3212566541`, and `1.0212873802`, respectively.  All three
validation generations were strict 400/400.

The frozen Q4 comparison gave AI-Hub-full-then-API-LoRA `4.9889583333`
(worst cell `4.950`) and the no-AI-Hub axis triplet `4.9891666667` (worst cell
`4.955`).  The predeclared rule therefore selected the no-AI-Hub triplet, but
the difference was only `-0.0002083333` for AI-Hub minus no-AI-Hub.  Paired
essay macros favored AI-Hub for 33, no-AI-Hub for 29, and tied for 338 of 400.
Furthermore, 4,754/4,800 AI-Hub cells and 4,750/4,800 no-AI-Hub cells were
score 5.  This is a saturated null comparison, not evidence that AI-Hub
pretraining materially harms rationale quality.

## Official judge and reward gates

Primary judge decoding is exactly one call per candidate at temperature `0`
and seed `42`.  The aggregate reports all 12 cell means, axis/dimension means,
macro mean, worst cell, 1--2 score rate, and strict parse rate.  Every request
contains the actual final predicted integer score and never a human score.

Before any fixed-proxy-judge RL stage, a train-only contrastive audit must show
the expected directional response to: swapped-axis rationales
(`domain_match`), score perturbation by two points
(`score_rationale_consistency`), and unsupported rationale replacement
(`groundedness`).  RL is skipped and preserved as a negative gate if this
audit fails.  The old v6 reward is never reused as official reward.

The directional train-32 gate passed all three predeclared checks: swapped-axis
domain-match mean decrease `1.6145833` (75% paired decrease), two-point score
perturbation consistency decrease `1.5104167` (78.125%), and unsupported
groundedness decrease `3.9791667` (100%).  A separate frozen prompt-injection
gate did not pass.  The injected strings never increased the macro score, but
the judge mostly tied rather than applying the predeclared degradation:
rationale injection decreased by only `0.0182292` with 9.375% paired decreases,
and essay injection by `0.0026042` with 3.125% paired decreases.  Therefore
`aggregate_rl_safety_gate.json` has `status=failed_gates` and
`rl_allowed=false`; DPO/GRPO jobs were not launched.  This should not be
misread as evidence that the model followed the attack.  It is evidence that
the immutable gate required a stronger adverse response than the saturated
proxy judge exhibited.  Changing that gate or prompt after seeing these
results is a scientific-protocol change and requires recorded authorization.

## Independent integer-score pretraining status

Score-model work continues independently of the failed rationale-RL gate.  It
uses only the three integer axes (`content`, `organization`, `expression`),
with deterministic half-up projection for AI-Hub decimals; the source
`average` member is neither read as a target nor evaluated as an output.

The first Qwen3-Embedding-8B full-pretraining run was preserved after a JSON
tuple/list selection-to-refit identity mismatch.  The second was preserved
after Transformers 5.14 selected FSDP2 by default: the mixed policy first
exposed a FP32/BF16 score-head GEMM mismatch and then Adafactor failed on mixed
Tensor/DTensor optimizer state.  The repaired lineage keeps the declared
Adafactor optimization and pins the supported FSDP1 full-shard backend; its
root score head is initialized in the backbone dtype and the head input follows
the live parameter dtype.  A four-GPU, one-update FSDP preflight passed at
global step 1 with finite loss `2.3809757233`.  Full selection is now running
on GPUs 0--3 under run
`official-aihub-integer-score-full-pretrain-v1-20260727-003`, Git commit
`8da22b9`.  Failed outputs and logs remain preserved rather than overwritten.

### 2026-07-28 public-score prompt correction and active lineage

The `20260727-003` result above is retained as a legacy compact-prompt
baseline.  It is not reused as the primary warm start for the public-spec
matrix.  The active score prompt is the separately versioned
`public_spec_score_only_v1` reconstruction, SHA256
`82814db3a6a20e72e73d70126eceb99c7ff653037b70c71723dc514089978d51`.
It contains the published definitions and 1--5 anchors for content,
organization, and expression, but prohibits rationale and average output.
This is a public-spec-aligned reconstruction, not unavailable verbatim
organizer wording.  Embedding and decoder AI-Hub pretraining and every new
MAL score-matrix arm are bound to this prompt kind and digest; mismatched
pretraining completions fail validation rather than being silently loaded.

The public-prompt AI-Hub input audit covered all 9,597 deterministic
selection-dev records.  The maximum rendered length was 1,419 tokens for the
embedding tokenizer and 1,418 for the decoder tokenizer, with zero records
over the fixed 2,048-token cap.  The first public-prompt run,
`official-aihub-integer-score-full-pretrain-v1-20260728-001`, remains
preserved with status `superseded_integration_optimization`: its evaluation
batch of one required about 4.7 minutes for each selection event.  The only
change in the successor was evaluation batch one to four; training batch,
gradient accumulation, model, targets, optimizer, seed, selection rule, and
prompt were unchanged.  This reduced the 9,597-row evaluation to about 74
seconds after the first event.

The active successor is
`official-aihub-integer-score-full-pretrain-v1-20260728-002`, launched from
Git SHA `2ee545552560c5aa8ea7eaf157721b0c2e829d05` with:

```text
.venv-standard/bin/python scripts/orchestrate_official_aihub_score_pretrain.py \
  --config configs/official_aihub_integer_score_pretrain.public_spec_score_prompt_eval4.v1.json
```

It uses seed 2026, the frozen AI-Hub manifest
`data/manifests/aihub_human_feedback_v1.json`, Qwen3-Embedding-8B revision
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, full-parameter Adafactor
training, BF16 FSDP1 full sharding, GPU0 for one-update smokes, and authorized
GPUs 0--3 for full stages.  Neither the canonical MAL validation split nor
its labels are accessed during this pretraining.

The bounded-regression selection stage completed normally after early
stopping at step 2,100.  The predeclared lexicographic selection rule chose
step 1,800 on AI-Hub selection-dev: macro integer RMSE `0.7658285489`, macro
integer Spearman `0.3423144375`, and macro continuous RMSE `0.7212457300`.
There were no OOM, NaN, or traceback events.  The full-data refit is running
to exactly the selected 1,800 optimizer steps; its displayed 24,020-step
progress denominator is only the fixed scheduler horizon, and an exact-step
callback plus completion assertion enforce termination at 1,800.  Current
state and commands remain authoritative in the ignored append-only
`ledger.jsonl` under the run root; this paragraph records only the verified
selection result and declared continuation.

After this run completes both bounded and ordinal refits, the already-launched
fail-closed follow-up queue resolves artifact checksums and runs (1) the four
public-prompt embedding MAL essay-only arms, (2) three public-prompt decoder
AI-Hub full-pretraining architectures, and (3) six public-prompt decoder MAL
essay-only arms.  Rationale-input score arms remain pending until the final
rationale handoff exists.  DPO and GRPO remain prohibited while the frozen
combined judge safety artifact has `rl_allowed=false`; changing the judge or
gate is a scientific-protocol change, not integration recovery.

### 2026-07-28 completed public-prompt essay-only score matrix

The follow-up queue completed all declared AI-Hub pretraining and MAL
essay-only arms.  Qwen3-Embedding-8B AI-Hub full pretraining selected step
1,800 for bounded regression (integer RMSE `0.7658285489`, Spearman
`0.3423144375`) and step 2,700 for cumulative ordinal (integer RMSE
`0.8499938415`, Spearman `0.1393128823`).  Both selected states were freshly
refit on the full eligible AI-Hub population and checksum-bound before MAL
LoRA adaptation.

The four embedding MAL arms produced the following single final descriptive
metrics on the 400-row canonical validation split.  Validation was not used
to choose an arm or epoch.

| embedding arm | integer RMSE | integer Spearman | continuous RMSE | continuous Spearman |
|---|---:|---:|---:|---:|
| bounded, public initialization | `0.6979703099` | `0.4937684638` | `0.6713512221` | `0.5162320981` |
| bounded, AI-Hub matched full initialization | `0.7066794047` | `0.4699002131` | `0.6603604325` | `0.5306839154` |
| ordinal, public initialization | `0.9470503003` | `0.0000000000` | `0.8201086698` | `0.5548210885` |
| ordinal, AI-Hub matched full initialization | `0.7530631854` | `0.2799085048` | `0.7120321257` | `0.5463415041` |

The train-internal selection rule chose
`bounded_regression__aihub_matched_full__essay` for bootstrap score emission;
the descriptively best canonical-validation arm was instead public-initialized
bounded regression.  This difference is preserved rather than selecting on
validation after observing it.  The authoritative aggregate is
`outputs/official-score-matrix-public-spec-score-prompt-v1/bootstrap_selection.json`.

Qwen2.5-7B-Instruct decoder AI-Hub full pretraining completed for constrained
generative, bounded-regression-head, and cumulative-ordinal-head architectures.
Their selected AI-Hub dev integer RMSE/Spearman pairs were, respectively,
`0.7624400558/0.3662155082`, `0.7523974843/0.3693720676`, and
`0.7519465706/0.3713056423`.  Each architecture was freshly full-parameter
refit before a new MAL LoRA was attached.

All six decoder MAL essay-only arms then completed with strict three-axis
integer output and no average or rationale target:

| decoder arm | integer RMSE | integer Spearman | continuous RMSE | continuous Spearman |
|---|---:|---:|---:|---:|
| constrained generative, public | `0.7221767088` | `0.4433148256` | `0.7221767088` | `0.4433148256` |
| constrained generative, AI-Hub matched | `0.7021636532` | `0.4834621804` | `0.7021636532` | `0.4834621804` |
| bounded head, public | `0.7141186567` | `0.4436299557` | `0.6682010498` | `0.5000198030` |
| bounded head, AI-Hub matched | `0.7087269751` | `0.4644076402` | `0.6561260123` | `0.5339045173` |
| cumulative ordinal head, public | `0.7044890932` | `0.4649240482` | `0.6529047105` | `0.5387973884` |
| cumulative ordinal head, AI-Hub matched | `0.7104686786` | `0.4653404056` | `0.6589121547` | `0.5261740089` |

The constrained generative AI-Hub arm was selected from train-internal metrics
and was also the descriptively best decoder arm on canonical validation.  It
had strict parse rate `1.0`; all 400 outputs belonged to the fixed 125-element
three-integer JSON space.  Across embedding and decoder essay-only arms, the
lowest observed canonical-validation integer RMSE remains the public-initialized
Qwen3-Embedding bounded head at `0.6979703099`.

After the sixth decoder arm completed, aggregate creation failed without any
loss of model or metric artifacts because the orchestrator expected a stale
`selection.selected_event` key while the runner correctly emitted
`selection.selected_epoch` plus the full `selection.events` history.  Commit
`0581433` repairs only that integration schema lookup, adds strict completed-arm
identity checks and an aggregate-only no-GPU recovery mode, and preserves the
original six completions.  The recovery wrote
`outputs/official-decoder-score-matrix-public-spec-score-prompt-v1/essay_bootstrap_aggregate.json`
with SHA256
`e65a6ee51b093ba21880737c54c7bb3faff17bc4ada5097d2ca5e863c1e6b2d3`;
the orchestration manifest is now `status=completed`.  No arm was retrained.

Rationale-input score arms still require the final SHA-bound rationale handoff.
That handoff remains downstream of the DPO/GRPO comparison, which remains
fail-closed until a scientifically authorized replacement safety protocol is
predeclared and passes.  Existing failed gate artifacts are not reclassified.

## User-supplied `llm_as_judge.txt` exact-prompt rerun

At pre-run Git SHA `042b657`, the user supplied repository-local
`llm_as_judge.txt` with SHA256
`91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`.
It is not byte-identical to the previously frozen proxy prompt, whose SHA256 is
`1a93a3a4c18d34318d6926871fa0a527bbaf422fe78dac8c4efb66345b222e34`.
The exact UTF-8 file contents were sent as the system message; the user message
contained only the canonical prompt text, essay text, and the unchanged
`candidate_predicted_score_and_rationale` JSON.  Human/reference scores were
not read or prompted.  The model/runtime remained the same exact
Qwen3.6-35B-A3B Q4_K_M and pinned llama.cpp build, with temperature `0`, seed
`42`, reasoning disabled, and the same strict 12-cell JSON schema.

After a one-row GPU0 smoke, the two pre-existing SFT validation participant
files were evaluated on GPUs 0--3 with:

```text
scripts/run_official_q4_judge.sh full <run-id> validation <participant-file> 400 llm_as_judge.txt
```

The axis-triplet run completed 400/400 with macro `4.9845833333`, worst cell
`4.9375`, 5-score cell rate `98.6667%`, and 359/400 essays perfect in all 12
cells.  The bundle run preserved one `schema_or_finish` negative and therefore
has 399/400 valid, macro `4.9803675856`, worst valid-cell mean `4.9473684211`,
5-score cell rate `98.3083%`, and 348/399 valid essays perfect in all 12 cells.
On the 399 paired valid essays, axis-triplet was higher for 49, bundle higher
for 35, and 315 tied; its mean advantage was `0.0041771094`.

Relative to the old abbreviated proxy prompt, the exact-file prompt lowered
the axis-triplet macro by `0.0045833333` and the bundle macro by `0.0058479532`
on paired valid essays.  Thus the more explicit prompt is slightly stricter
but remains strongly ceiling-saturated.  The aggregate-only comparison is
`outputs/official-prompt-alignment-v1/llm-as-judge-txt-comparison/llm-as-judge-txt-comparison-20260728-001/aggregate_prompt_comparison.json`
with SHA256
`61f0d83047b835fed6e794ff5f60c03195c507d03109921124954951a92e07da`.
The one bundle failure was not overwritten or silently imputed.

## Execution task card

- Run ID: `official-prompt-alignment-v1-20260727-001`.
- Deliverable: aggregate comparison of A0/A1/A2/A3 under official integer
  metrics and exact-Q4 judge; Docker excluded.
- Completion predicate: strict final JSON success 100%, finite official score
  metrics, exact 12-cell Q4 report, and aggregate lineage/checksums for every
  completed arm; negative arms remain recorded.
- Privacy: prompts, essays, identifiers, rationales, row predictions, API
  responses, and model states remain in ignored roots.
- Resource scope: repository-default GPUs 0, 1, 2, and 3, explicitly requested
  by the user in this thread; GPU0 first for the smallest real smoke.
- Test ladder: CPU contract checks -> one real GPU0 score batch and one
  train-only Q4 schema call -> declared GPU0--3 full stage.
- Integration recovery: launcher, schema transport, serialization, memory,
  and batch-size repairs that preserve data, model, targets, optimization, and
  selection are autonomous under the repository policy.
- Stop/escalate: existing GPU process conflict, destructive overwrite,
  unapproved data transfer/cost beyond the declared API cap, split/checksum
  mismatch, three exhausted integration repairs, or a new scientific variable.
- Ledger: append-only
  `outputs/official-prompt-alignment-v1/20260727-001/ledger.jsonl`.

## Hard failures

- any non-integer/out-of-range final score;
- strict JSON parse rate below 100% after bounded transport retry;
- a final composer score differing from the frozen encoder score;
- nonfinite loss/parameters/metrics or constant predictions on all axes;
- validation rows used for training, reward, prompt revision, or threshold
  fitting;
- human/reference scores included in an official judge request;
- exact Q4/model/runtime provenance mismatch.
