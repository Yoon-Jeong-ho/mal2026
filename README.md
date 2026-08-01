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

## Train-OOF tail refinement (20 rounds, fail-closed)

A fixed 20-round study tested residual, ordinal, tail-weighted, 3-vs-4,
score-blind rationale-evidence, ensemble, and bounded-calibration candidates
on the exact 2,000-row train OOF population. No candidate passed every frozen
promotion gate, so the exact R0 OOF baseline remains selected (macro continuous
RMSE `0.568780`). The exploratory best, a regularized evidence projection,
reached `0.564401` but missed the `0.005` improvement gate and worsened
low-tail `{1,2}` RMSE from `0.923335` to `0.984277`. Validation was not loaded,
and no validation or deployment improvement is claimed. See
`docs/experiment_records/iterative_tail_refinement_20260801_001.md` for all
20 aggregate results, tail/3-vs-4 metrics, deviations, and the fail-closed
decision.

### Leakage-safe tail remediation (v2)

A stricter 5x4 nested follow-up regenerated inner teachers without selection
holdout leakage and evaluated conditional routing, exact monotone calibration,
tail-boundary correction, direct evidence ridge, and a registered ensemble.
All five outer folds failed closed to the exact R0 baseline. The main stable
signal was a trade-off: rebuilt R17 improved score 5 and 3-vs-4 separation but
materially worsened the combined `{1,2}` tail. See
`docs/experiment_records/iterative_tail_remediation_20260801_001.md` for the
protocol repairs, aggregate metrics, and negative result.

### Twenty-cycle routed-tail discovery (v3)

A second fixed 20-cycle train-only program evaluated soft routing, Pareto
stacks, five-band Group-DRO, selective hurdle heads, and final-output ordinal
stacks. All 100 fold-cycle predictions completed before metrics were opened.
No candidate passed the strict gate; R0 remains selected. The best exploratory
cycle improved RMSE only from `0.568780` to `0.568611`, although both `{1,2}`
and score-5 continuous RMSE moved slightly in the correct direction. V3 is
adaptive after observing v2 and is not confirmatory. See
`docs/experiment_records/iterative_tail_cycle_20260801_001.md`.

### Final nested twenty-router study (v4)

The final adaptive train-only attempt rebuilt R16/R17/direct/hurdle/soft
components inside a sealed 5-outer x 4-inner protocol and fit 20 fixed
score-only routers per outer fold. None of the 100 route fits passed the strict
joint gate, so every outer fold fell back to exact R0 and final RMSE remains
`0.568780`. Four-zone routing showed tiny descriptive gains at scores 1, 2,
and 5, but did not improve score-1/5 recall and slightly traded off scores 3/4.
The preregistered stop rule froze the V4 candidate inventory and learned
state. A later user clarification authorized a separately named V5 study; V4
itself remains immutable. See
`docs/experiment_records/iterative_tail_router_20260801_001.md` for nesting,
all 20 aggregate route results, band diagnostics, checksums, and claim limits.

### Fresh nested twenty-learner study (v5)

A separately authorized adaptive train-only study evaluated five new bounded
GPU learner families, four fixed variants each, over frozen
Qwen3-Embedding-8B train-OOF embeddings, exact R0 scores, and score-blind
rationale hash features. The sealed 5-outer x 4-inner protocol completed 100
candidate OOF evaluations (400 fresh inner fits) on GPUs 0--3. None passed the
strict joint gate, so all outer folds fell back to exact R0 and final RMSE
remains `0.568780`.

The best macro candidate still worsened mean RMSE by `0.000404`; it improved
score-5 continuous RMSE and 3/4 balanced accuracy but worsened the `{1,2}`
tail. A different candidate improved both tails while degrading global RMSE
and 3/4 separation. Score-5 integer recall did not change in any candidate
cell. Thus V5 reproduces the tail-versus-center trade-off rather than finding
a promotable learner. It is descriptive same-train evidence only. See
`docs/experiment_records/iterative_tail_learner_20260802_001.md` for all 20
candidate summaries, gate failures, execution evidence, and checksums.

### Cross-fitted directional tail falsification (v6)

V6 replaced the globally capped V5 corrections with identity-default low,
high, and center experts whose high correction could cross the score-5
boundary. Three fixed candidates were evaluated with sealed 5-outer x 4-inner
cross-fitting: 15 candidate evaluations and 60 fresh inner fits, with folds
0--3 initially running concurrently on GPUs 0--3. All five outer selections
failed closed to exact R0, leaving final macro RMSE `0.568780`.

The nonlinear variants improved score-5 continuous RMSE and 3/4 balanced
accuracy in every inner-OOF population, but worsened the `{1,2}` tail in every
one and never changed score-5 integer recall. The linear control produced only
a `0.000183` mean macro gain and worsened 3/4 separation. V6 therefore freezes
further tuning on the same train population and feature sources; it does not
preclude a separately registered study using materially new evidence. See
`docs/experiment_records/iterative_tail_directional_20260802_001.md` for the
full gate matrix, diagnostics, checksums, and claim limits.

### Official Terra agent-score stack (v7)

V7 added a genuinely new score-blind feature source: three completed
GPT-5.6-Terra official participant outputs per train essay. A disclosed
adaptive prestudy found a promising fixed residual stack at RMSE `0.554928`,
with both tails and 3/4 separation improved. The subsequent sealed 5-outer x
4-inner run selected that stack in three folds and fell back to R0 in two.

The nested development prediction improved R0 from `0.568780` to `0.563253`;
its paired improvement CI excluded zero and all axes plus both tails improved.
It was still not promoted because the frozen final requirements were a macro
gain of at least `0.010` and 3/4 balanced-accuracy gain of at least `0.010`,
whereas observed gains were `0.005528` and `0.003271`. Folds 2 and 3 failed
their inner gates solely on 3/4 separation. Exact R0 therefore remains the
protocol-valid model, while an independently registered boundary-head
follow-up is scientifically motivated. See
`docs/experiment_records/iterative_official_agent_stack_20260802_001.md`.

### Official Terra 3/4 boundary heads (v8)

V8 implemented fresh adjacent-3/4, cumulative-`>=4`, and dual logistic heads
on top of the V7 official-agent residual. The adjacent head raised inner 3/4
balanced-accuracy gains, but folds 2 and 3 still missed the frozen BA gate and
fell back to R0. The resulting nested development RMSE was `0.565515`, with
both tails and all axes improved but only `0.003265` macro and `0.003018` BA
gain. Its bootstrap CI crossed zero, so exact R0 remains selected. Smooth
nudges were less efficient than V7; any further boundary attempt must be a
separately registered high-confidence selective flip, not a V8 retune. See
`docs/experiment_records/iterative_official_boundary_20260802_001.md`.

### High-confidence official Terra boundary flips (v9)

V9 preregistered a discontinuous alternative to V8's smooth nudges: a cell
could cross 3.5 only when a fresh adjacent/threshold classifier disagreed
confidently with the fresh official-agent residual and the residual was inside
a fixed near-boundary window. GPU0 smoke passed, then outer folds 0--3 ran
concurrently on GPUs 0--3 and outer fold 4 followed on GPU0. Three folds
selected a flip candidate and two fell back to exact R0 because their inner
3/4 balanced-accuracy gain was below `0.010`.

The nested development RMSE was `0.563067` versus R0 `0.568780`; every axis,
both tails, equal-band RMSE, Spearman, and the paired bootstrap moved in the
favorable direction. The final gate nevertheless failed because macro gain
was only `0.005714` and 3/4 balanced-accuracy gain only `0.004620`, both below
the required `0.010`. Exact R0 therefore remains the protocol-valid model.
See
`docs/experiment_records/iterative_official_selective_flip_20260802_001.md`.

### Score-blind Terra + Luna dual-agent stack (v10)

V10 froze its candidate inventory before downloading a new 6,000-output
score-blind GPT-5.6-Luna batch, then bound only the validated manifest and row
checksums. The fixed 96-dimensional view combined three Terra and three Luna
participant score vectors with within-model and cross-model disagreement
features. A fresh residual ridge or one of two optional 3/4 flip variants was
selected inside the same sealed 5-outer x 4-inner protocol.

Only outer folds 0 and 1 passed the full inner gate; folds 2--4 failed solely
because 3/4 balanced-accuracy gain stayed below `0.010`. The nested development
RMSE was `0.563717` versus R0 `0.568780`, and all axes, both tails, equal-band,
Spearman, and the paired interval improved. Final macro and 3/4-BA gains were
only `0.005063` and `0.004180`, so exact R0 remains selected. See
`docs/experiment_records/iterative_official_dual_agent_20260802_001.md`.

### Class-balanced Terra + Luna 3/4 boundary head (v11)

V11 replaced V10's unweighted boundary correction with a dedicated gold-3
versus gold-4 classifier whose two classes had equal total training weight.
All three specifications were frozen before the sealed 5-outer x 4-inner run.
The narrow-window head passed in two outer populations, while all candidates
in the other three again missed only the 3/4 balanced-accuracy gate.

Nested development RMSE was `0.563918` versus R0 `0.568780`; both tails, all
axes, equal-band RMSE, Spearman, and the paired interval improved. Final macro
and 3/4-BA gains were only `0.004862` and `0.002545`, below the required
`0.010`, so exact R0 remains selected. Class balancing made the three failing
folds' BA gains smaller than V10's plain ridge, freezing further threshold/L2
retuning of this family. See
`docs/experiment_records/iterative_official_balanced_boundary_20260802_001.md`.

### Terra + Luna rationale-semantic falsification (v12, terminal)

V12 embedded 36,000 score-blind Terra/Luna axis rationales with the frozen
public Qwen3-Embedding-8B revision and reduced them to a fixed 201-dimensional
target-blind semantic view. It compared semantic-only residual ridge, fusion
with V10's structured 96 features, and the same fusion with a balanced 3/4
head under the unchanged sealed 5-outer x 4-inner protocol.

All three candidates failed the inner conjunction in all five populations.
Mean macro gains were `-0.040285`, `-0.024304`, and `-0.025000`; mean 3/4-BA
gains were also negative. Although fusion improved score-5 RMSE, it degraded
center accuracy and rank correlation. Every outer fold therefore fell back to
exact R0 and final RMSE remained `0.568780`. This triggers the preregistered
terminal same-train adaptive stop: further tuning now requires independent new
labels or an untouched evaluation population. See
`docs/experiment_records/iterative_official_rationale_semantic_20260802_001.md`.
The full chronology and final selection audit are in
`docs/experiment_records/iterative_program_final_audit_20260802_001.md`.

## `evaluation.txt` end-to-end re-audit (completed)

The current prompt-alignment rerun separates four contracts: score-blind
rationale generation, emitted-score-conditioned rationale generation, direct
three-axis score prediction, and rationale-aware three-axis score prediction.
All retain the exact `evaluation.txt` rubric prefix; no path trains or predicts
an `average` target.  Rationale generation used A.X-4.0-Light with a
four-GPU tensor-parallel vLLM engine, while score prediction compared actual
embedding models `Qwen/Qwen3-Embedding-8B` and `nlpai-lab/KURE-v1`.

On the frozen 400-row validation split, the best arm is Qwen3-Embedding-8B
with matched score-blind rationales: content / organization / expression RMSE
is `0.516236 / 0.706146 / 0.528646`, and the three-axis macro RMSE is
`0.583676` (macro Spearman `0.619365`).  Score-conditioned rationales worsen
both Qwen and KURE relative to their direct baselines.  The byte-exact
`llm_as_judge.txt` Qwen3.6-35B-A3B Q4_K_M evaluation completed seven 400-row
arms with no failures, but is saturated: arm means are `4.9646--4.9796` and
`97.02%--98.38%` of judge cells are 5.  It evaluates rationale plausibility
conditional on the predicted score rather than re-grading score correctness,
so it must not be interpreted as RMSE or used alone to select these arms.
See
`docs/experiment_records/evaluation_prompt_end_to_end_reaudit_20260729_001.md`
for prompt hashes, negative results, commands, complete metrics, and the
paired judge analysis.

## Solar synthetic-score consensus pilot (completed, fail-closed)

The train-only Solar pilot generated 2,400 score-directed essay variants and
then discarded the requested score as a label. Exact `evaluation.txt` blind
scoring produced 1,229 stable 3→5-draw modal candidates after hard filtering.
However, content and organization produced no stable score-5 candidates and
all axes remained concentrated at scores 2--3. Five-fold real-train OOF
scorers reached macro RMSE `0.598753` (Qwen3-Embedding-8B) and `0.632588`
(KURE-v1); their OOF-calibrated agreement retained only 328 candidates.

A score-independent 10% control was rescored by the pinned
Qwen3.6-35B-A3B Q4_K_M judge with the same exact prompt. The supposed
consensus core did not generalize to this independent judge: macro RMSE versus
the Solar pseudo-label was `0.714144` in the core versus `0.671976` in the
disagreement stratum, and exact triplet agreement was `9.09%` versus `23.08%`.
The core expression RMSE was `0.937437`. The predeclared fail-closed gate
therefore blocks synthetic-score mixture training; no validation result was
produced from this augmentation. See
`docs/experiment_records/solar_consensus_filtering_pilot_20260730_001.md`.

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

The follow-up 12-checkpoint epoch sweep selects epoch 3 / step 96 by the fixed
RMSE-first rule: content / organization / expression RMSE is
`0.512864 / 0.695580 / 0.499289`, with three-axis diagnostic RMSE `0.569245`.
Epochs 1--4 form the useful early region; epoch 2 has the highest diagnostic
Spearman and epoch 4 has the lowest organization RMSE.  This is a descriptive
validation-selected checkpoint, not an untouched test estimate.  See
`docs/experiment_records/rlaif_qwen3_embedding_epoch_sweep_v1_20260726_003.md`.

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
