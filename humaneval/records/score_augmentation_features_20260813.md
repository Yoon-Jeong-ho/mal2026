# Human-reviewed score-augmentation features — 2026-08-13

- **Status:** exploratory feature specification; not an accepted labeling rule.
- **Purpose:** define safer train-only features and adjacent-band contrasts for
  generating candidate essays at scores 1, 2, and 5.
- **Privacy:** this tracked record contains aggregate counts and generalized
  features only. Individual writings, source identifiers, reviewer identities,
  reasons, and the local example page remain under ignored output paths.

## Evidence boundary

The review joined the fixed human-response export to `eval/train.jsonl` and
compared each human score and reason with the corresponding essay, canonical
score, and secondary reason audit. The single human-reviewed validation essay
was excluded before feature extraction.

- Unique train essays: 19.
- Reviewers per essay: 2.
- Axis judgments compared between reviewers: 57.
- Exact inter-reviewer agreement: 15.8%.
- Agreement within one point: 40.4%.
- Inter-reviewer mean absolute difference: 1.93 points.
- Written direct-score reasons: score 1 = 12, score 2 = 9, score 5 = 0.
- Reasons that were `match` or `partial` in the secondary audit and whose human
  score was within one point of the canonical half-up band: score 1 = 7,
  score 2 = 6, score 5 = 0.

The reviewers used materially different score distributions. Human judgments
therefore inform features only when the cited property is visible in the essay
and belongs to the stated axis. A human score alone is not treated as truth.
Score-5 features have lower confidence because no direct human reason was
written for a score-5 judgment.

## Candidate features

### Score 1: criterion function is mostly absent

**Content**

- The position is unresolved until late in the essay, changes, or conflicts
  with another position in the same essay.
- Reasons have little functional connection to the claim and are replaced by
  unsupported assertions, stereotypes, personal impressions, or loose
  analogies.
- Repetition does not advance the argument, and the response to the assigned
  question is not sustained.

**Organization**

- Introduction, body, and conclusion do not perform distinguishable discourse
  functions; the essay reads primarily as an unordered list.
- Abrupt insertions, position changes, and repeated passages make the intended
  sequence difficult to reconstruct.
- The conclusion fails to synthesize the argument, or the structure collapses
  toward the end.

**Expression**

- Sentence boundaries and punctuation repeatedly fail, or extremely long
  sentences repeatedly interrupt interpretation.
- Spacing, spelling, endings, grammar, and agreement errors accumulate across
  the essay.
- The defining property is repeated interference with comprehension, not one
  or two isolated errors.

### Score 2: a functional core remains but fulfillment is limited

**Content**

- A position and at least one related reason can be recovered, but support
  remains generic, asserted, or underdeveloped.
- Claim-to-example causality is weak, with overgeneralization or extended
  digression.
- The intended point is recognizable, but specificity, rebuttal, or explicit
  logical support is insufficient.

**Organization**

- Traces of opening, support, and closing—or at least a followable list—remain.
- Transitions are mechanical or abrupt, and repetition or insertion prevents
  paragraphs from fully performing their intended functions.
- The conclusion merely repeats prior material or introduces a judgment that
  was not prepared by the body.

**Expression**

- Awkward sentences and orthographic or grammatical errors recur, but the main
  meaning remains recoverable.
- Some sentences are incomplete or unnatural while a substantial portion is
  still readable.
- Unlike score 1, the errors do not continuously destroy comprehension across
  the entire essay.

### Score 5: concrete strengths with almost no recurring defect

**Content**

- The position is clear from the opening and multiple distinct reasons respond
  directly to the assigned issue.
- Each reason is developed through a mechanism, example, or consequence, with
  explicit claim-to-evidence links.
- The essay addresses a counterargument or limitation, proposes conditions or
  an alternative when relevant, and synthesizes the argument in the conclusion.

**Organization**

- The opening establishes the issue and position, and body sections have
  distinct functions.
- Reasons, counterargument or limitation, and response follow a natural order
  without duplicated material or abrupt insertion.
- The conclusion compresses the preceding argument and does not introduce a
  new issue.

**Expression**

- Sentence structure is stable, wording is precise for the issue, and style and
  endings are consistent.
- Spelling, spacing, punctuation, grammar, and agreement errors are nearly
  absent, so meaning is immediately accessible.
- A favorable overall impression is insufficient: recurring minor errors or a
  recurring weak construction should trigger consideration of score 4 first.

## Adjacent-band contrasts

### Score 1 versus score 2

The boundary is functional rather than an error count. Score 2 retains a
recoverable core in the target axis: one relevant reason for content, a broadly
followable sequence for organization, or sustained basic comprehensibility for
expression. Score 1 lacks that core or loses it repeatedly. Synthetic pairs
should cross this boundary with the smallest possible edit rather than adding
arbitrary spelling noise.

### Score 4 versus score 5

Score 4 is broadly strong but retains a recurring minor weakness or one
materially weak logical connection. Score 5 requires both concrete strengths
and the practical absence of recurring defects. A score-5 candidate should be
created by repairing verified defects in a score-4 candidate, not by adding
generic sophistication or ornate language.

## Proposed augmentation contract

1. Assign an explicit `(content, organization, expression)` target vector;
   never request an undifferentiated “score-N essay.”
2. Generate minimal-edit contrast pairs for `1↔2` and `4↔5`, changing one axis
   while holding the other two as stable as possible.
3. Use only train-split demonstrations. Abstract their boundary properties;
   do not copy individual reasons, arguments, or sentences into the prompt.
4. Treat the requested score as a provisional generation target, not a label.
5. Reject prompt leakage, demonstration copying, implausible error injection,
   topic drift, and failure to satisfy length or format constraints.
6. Use independent scoring and rationale-grounding checks for screening. Because
   the tested decoder failed to recover score 1 and collapsed score 5, its score
   must not be used as a sole acceptance gate.
7. Require human boundary review for retained `1↔2` and `4↔5` pairs, with full
   review of score-5 candidates until direct human evidence is strengthened.

## Reproducibility bindings

- Repository base used for the analysis: `243f187`.
- `eval/train.jsonl` SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Ignored human score-detail export SHA-256:
  `a4066b6fbfa37b5a152c1cbb6ecaaa0916c39950036563e25e635949fa0027cc`.
- Ignored reason-audit export SHA-256:
  `68c6117280fb28a03f954c53c88845394f3e020755c87b2cfa2559807849223f`.
- Local restricted review page:
  `outputs/humaneval/reports/cloudflare-final-20260809-001/score_feature_review.html`.

No generated essay is admitted by this record, and no validation result is
used to retune these features.
