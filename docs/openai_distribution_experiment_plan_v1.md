# OpenAI explanation-data distribution and evaluation plan v1

**Status:** preregistered; audit completed; candidate selection, SFT, DPO, and
GRPO remain blocked. This plan is aggregate-only and contains no student text,
explanations, prompts, identifiers, provider IDs, or credentials.

## Immutable boundaries

`eval/validation` is evaluation-only. It must never choose a judge prompt,
candidate, SFT target, reweighting/coarsening threshold, or any training
decision. The score-model average is not modeled: it is always computed after
the three component predictions or labels as `(content + organization +
expression) / 3`.

The completed audit distinguishes the API population (`eval/train` plus
`eval/validation`) from the separately prepared AI-Hub source population. No
cross-population divergence is claimed without a deterministic provenance
join. Fixed component bands are `[1,2)`, `[2,3)`, `[3,4)`, `[4,5]`; fixed
character-length bins are `[0,200)`, `[200,400)`, `[400,800)`, `[800,1200)`,
and `[1200,inf)`.

## Audit and promotion gates

Run `scripts/audit_openai_distribution_metadata.py` to regenerate the tracked
aggregate report. It reports counts, joint score strata, lengths, opaque
task/prompt-group coverage, total-variation, Jensen-Shannon, Hellinger,
min-mass overlap, maximum share difference, and sparse/zero cells for the
declared split comparisons. A cell below 20 is sparse.

API availability and strict schema/grounding QC are reported only from the
aggregate batch manifests. Availability is not candidate selection.

Judge-v3 must pass every declared global hard gate. Each candidate later used
must also have valid schema/grounding, fixed-repeat pointwise eligibility, and
an invariant pairwise result after predeclared label/position unblinding across
the fixed lanes. Only strict winners are eligible; ties, abstentions, failures,
and missing evidence are excluded. The current v3 pilot has
`selection_artifact_permitted: false`; even a passing pilot requires a new,
versioned production-selection protocol before an artifact can be made.

`scripts/validate_openai_distribution_utilization.py` is deliberately a
non-selecting fail-closed gate. It may be run now and must report
`blocked_pending_judge_v3` until a passing aggregate v3 manifest exists.

## Train-only utilization after the gates

The future production protocol uses only `eval/train`. It will calculate, by
fixed score-joint × length strata, available, judge-eligible, high-confidence,
and selected counts; selection propensity with Wilson 95% intervals; selected
to-available ratios; weight summaries; clipping; and effective sample size.

No calibration is allowed with base count below 30 or accepted count below 10.
Before selection only, a failing joint stratum can use the fixed coarsening
order in [the v1 configuration](../configs/openai_distribution_utilization.v1.json):
joint × length, axis × length, axis, length, then source population. Every
coarsening decision is recorded. Zero-support strata cannot be selected, and a
fallback may not silently remove low-score, short-text, or rare strata.

Distribution preservation targets the train-only high-confidence population.
Use deterministic stratum quotas or capped inverse-propensity weights (maximum
3), and fail if a required supported stratum has zero selected examples.

## Validation-only score-model and judge reporting

After an independently fixed score model is available, report validation-only
RMSE, MAE, and Spearman for content, organization, expression, and external
average, both overall and for the preregistered source/task-metadata/score-band/
joint-stratum/length slices. Publish slices at `n >= 20`; report Spearman as
null unless `n >= 50` and both target and prediction have at least two ranks.
Use a 95% cluster bootstrap at the prompt-group (or source) level. These are
reporting rules only; no validation metric may alter training or selection.
