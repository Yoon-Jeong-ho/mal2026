# Luna exact-R0 low-tail causal audit — 2026-08-04 run 002

## Scope and authority

- **Run ID:** `luna-r0-tail-causal-audit-v1-20260804-002`
- **Question:** why do human/reference score-1/2 axis labels become exact-R0
  score 3/4, and how much is attributable to the rubric prompt, label
  distribution, rationale input, or score-model threshold/calibration?
- **Authorization:** the user explicitly authorized many GPT-5.6-Luna reviews
  of the low-score cases and the associated external API/text transfer.
- **Population:** canonical train only and exact five-fold R0 OOF predictions.
  Validation was not loaded. `average` was not read or evaluated.
- **Privacy:** all essays, prompts, rationales, IDs, mappings, requests,
  responses, and row decisions remain under ignored restricted storage. This
  tracked record contains aggregates only.
- **GPU scope:** none.

## Inputs and execution

| Artifact | SHA-256 |
|---|---|
| canonical train | `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737` |
| exact R0 OOF | `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04` |
| actual R0 input rationale | `d3fba8cfa43d62ea17b904d65164aee9f68e173c6dde9701d9f3683532b7eab1` |
| official score-blind rationale | `d4a2be9a070c786728fde6f64f066ac9d462bc5f83305a2d9161b380abd88e55` |
| official score-conditioned rationale | `d0633e6a23ffb6ad26f4abfa12f32ab22b705ee16be3ed94fc6e2f05056170da` |
| final DPO rationale | `45dc9bfd05d60c75214221e34149ed7bff6dae0d571a90fde287ab193bb6f347` |
| `evaluation.txt` | `1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145` |
| config | `d86e076c8be9122a7fb83c28ceecb42ce5e5f50f5522cd316d0bd600690462e0` |

The audit selected 752 axis cases:

- all 426 low-to-central errors: content 144, organization 226, expression 56;
- all 146 correctly-low controls: content 69, organization 52, expression 25;
- 30 prompt-stratified correct-center controls per axis;
- 30 prompt-stratified score-5-to-central controls per axis.

Each case received three canonical-prompt blind reviews, three operational-
boundary blind reviews, and two revealed causal reviews. Blind reviews saw no
gold score, R0 score, or rationale. Revealed reviews saw the scores and five
rationale views, with the actual causal role of each view stated explicitly.
The resulting 6,016-request Batch API job
`batch_6a719ab790508190ba17017f17974caa` completed 6,016/6,016 with zero provider
failures. One HTTP-200 response was truncated at the 1,000-token output limit;
it was preserved and retried once with an 1,800-token ceiling without changing
model, prompt, inputs, or schema.

Batch usage was 8,748,426 input and 1,917,788 output tokens. Including three
smokes and the single integration retry, total usage was 8,755,862 input and
1,919,284 output tokens. No currency price is inferred.

- Request SHA-256:
  `dc71ed21e4fb8d8b0b037601f33e7705d7dcdbaaa2c357ff566399be01d14f26`
- Batch output SHA-256:
  `1015b75707032e2e64165eddd654e5becfca033aba20b50f65ee3ab54f2f6385`
- Aggregate SHA-256:
  `cf9e97c20a20926bfce4a99c983e5831d7223c3186e53029cf4ed86dbe0c76b6`
- Case-clustered analysis SHA-256:
  `ff4bb29c54f87b4b0f49f80f77fa5609e2cca7cba3f747e8294e68a37fe68a28`

Run 001 failed before writing any request or making any API call because its
control sampler assumed numeric prompt IDs while canonical IDs are `Q1`--`Q9`.
Its empty attempt directory and aggregate-only failure record were preserved.

## Exact distribution diagnosis

| Axis | Gold 1/2 count | Gold 1/2→R0 3/4 | Gold 5→R0 3/4 | R0 prediction support |
|---|---:|---:|---:|---|
| content | 213 | 144 (67.6%) | 56/56 (100%) | 1:1, 2:131, 3:1152, 4:716, 5:0 |
| organization | 278 | 226 (81.3%) | 211/211 (100%) | 1:0, 2:96, 3:1063, 4:841, 5:0 |
| expression | 81 | 56 (69.1%) | 293/293 (100%) | 1:0, 2:39, 3:598, 4:1363, 5:0 |

This is symmetric output compression, not only a rare-low-label problem: the
model emits almost no 1 and no 5, and every human score-5 case also collapses
to 3/4. On the low-to-central errors, the mean continuous upward bias was
`+0.789` content, `+1.071` organization, and `+0.968` expression. Organization
is worst despite having more low examples than expression, so raw class count
alone is insufficient as an explanation.

## Independent blind Luna scoring

The table uses one median score per case across the three repeats.

| Axis | Cases | Canonical Luna predicts low | Operational Luna predicts low | Canonical repeat exact agreement |
|---|---:|---:|---:|---:|
| content | 144 | 27.8% | 18.1% | 84.0% |
| organization | 226 | 23.9% | 15.9% | 88.5% |
| expression | 56 | 71.4% | 73.2% | 87.5% |

For content and organization, Luna independently reproduced the central score
on most human-low/R0-central cases even without seeing the R0 output or any
rationale. This implicates a mismatch between the broad written rubric and the
latent human scoring boundary, shared LLM central tendency, or both. The
judgments were stable across repeats, so additional sampling alone is unlikely
to solve it.

For expression, blind Luna recovered the human low band on roughly 72% of the
same R0 errors. Expression therefore has a substantially clearer observable
low-score signal; its remaining R0 error is more consistent with the score
representation, rationale input, or threshold than with an inherently
unreadable rubric.

The proposed operational wording did **not** repair content/organization. It
raised their median score by about `+0.125` and `+0.124`, respectively. Merely
stating that score 2 lacks an essential function and score 3 minimally contains
all functions caused Luna to infer that most functions were present. This
version must not replace the official prompt. Human-scored adjacent anchor
pairs are needed to define what “present” and “materially impaired” mean.

Prompt/topic effects exist but are not the whole cause. Organization low-to-
central errors ranged from 62.1% on Q3 to 97.1% on Q6 and 94.4% on Q8; every
prompt still had substantial error.

## Rationale diagnosis

The actual R0 score encoder used the short `rank2_ax4_random1` score-blind
rationale, not the later final DPO rationale. On low-to-central errors, Luna's
revealed reviews produced the following rationale aggregates:

| Axis | Actual R0 rationale positive | Specificity / 5 | Supports human low | Supports R0 central |
|---|---:|---:|---:|---:|
| content | 87.8% | 2.81 | 20.1% | 91.0% |
| organization | 96.0% | 2.15 | 4.6% | 97.6% |
| expression | 50.9% | 1.90 | 49.1% | 52.7% |

By contrast, the later official score-blind rationales had specificity
`4.51/4.76/4.79` and supported the human-low interpretation in
`89.6%/93.6%/98.2%` of content/organization/expression reviews. The final DPO
rationales were also specific and mostly supported the human-low assessment,
but they were generated after the score and therefore cannot have caused the
score error.

The revealed causal prompt selected `r0_rationale_overpositive` as its primary
cause in 91.3% content, 91.6% organization, and 74.1% expression observations;
model-threshold and missing-defect flags were also repeatedly high. These
figures are suggestive rather than causal proof because the human score was
visible in this condition. The stronger triangulation is: the actual input
rationale is positive, generic, and aligned with the erroneous central score;
matched-rationale versus shuffled-rationale experiments already show that the
Qwen encoder does use rationale content; and blind Luna independently exposes
different axis behavior.

## Operational score interpretation learned from the audit

The following is a diagnostic rubric, not an authorized replacement prompt:

- **1:** the axis's essential function is absent or the response is effectively
  unrecoverable.
- **2:** some relevant signal exists, but at least one essential function is
  absent or repeated global defects materially block task performance.
- **3:** every essential function is minimally present and weaknesses are local,
  traceable, and non-blocking.
- **4:** all essential functions are clear and mostly robust; only minor defects
  remain.
- **5:** performance is consistently precise, well-supported, and compelling,
  with almost no meaningful defect.

Axis-specific essential functions are: task response, explicit claim, relevant
concrete evidence, and claim-evidence link for content; identifiable global
progression and functional transitions for organization, independent of
physical line breaks; and recoverable meaning with no recurring blocking error
for expression. Every decision should state why the adjacent lower and higher
bands do not apply. The failed operational blind arm shows that these words
still require empirical human-scored anchors.

## Conclusion and next discriminating experiment

The low-tail failure is multi-factor:

1. **Output/head compression is proven:** neither low nor high tails are
   represented by R0.
2. **The current prompt is under-specified for human content/organization
   boundaries:** blind Luna also prefers 3/4 there.
3. **The actual R0 rationale likely amplifies upward bias:** it is short,
   low-specificity, overwhelmingly positive, and aligned with the wrong score.
4. **Expression differs:** a blind large model can usually see its low-score
   defects, pointing more directly to the encoder/rationale/head path.
5. **Prompt/topic and label imbalance modulate the problem but do not explain it
   alone.**

The most discriminating next experiment is an exact train-only nested OOF
three-arm comparison with identical Qwen initialization, folds, score head,
and optimizer: essay-only, essay plus the historical R0 rationale, and essay
plus the later official score-blind rationale. It must report per-axis low and
high tails as well as global RMSE. Current Luna outputs must not be used as
training labels, and the saturated `llm_as_judge.txt` score must not select an
arm. This diagnostic does not itself authorize that training experiment.

## Limitations

- Luna is an automated auditor, not independent human ground truth.
- The revealed condition can anchor on the displayed human score; blind results
  are the stronger evidence for score disagreement.
- Repeats for one case are clustered, not independent samples.
- The repeatedly exposed validation split was deliberately excluded, so no
  validation or hidden-test improvement is claimed.
