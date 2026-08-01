# Official Terra + Luna dual-agent stack (V10)

Run ID: `iterative-official-dual-agent-v10-20260802-001`

Status: completed; strict final gate failed, exact R0 retained

Date: 2026-08-02 (Asia/Seoul)

## Decision

V10 added a materially new score-blind model source to the frozen Terra
features: three official participant outputs from GPT-5.6-Luna for every train
essay. The exact candidate inventory was committed before the Luna batch
results were downloaded. After the batch validated, a binding-only commit
filled exactly the manifest SHA, row SHA, and binding-state fields; no model,
feature, fold, metric, gate, or hyperparameter changed.

The nested development result improved exact R0 macro continuous RMSE from
`0.568780` to `0.563717`. Every axis, equal-band RMSE, both tails, 3/4
balanced accuracy, Spearman, and the paired bootstrap moved favorably.
Nevertheless, only two outer folds passed the unchanged inner AND gate. The
final macro gain was `0.005063` and 3/4 balanced-accuracy gain was `0.004180`,
both below the required `0.010`. Exact R0 remains the protocol-valid model.

V10 is same-train nested development evidence, not independent validation or
a hidden-test, leaderboard, generalization, or deployment result.

## Score-blind Luna generation

- Generation-plan commit: `8b042c5`.
- V10 scientific-inventory preregistration commit: `7ad2d6c`.
- Sealed runner commit: `470ce85`.
- Checksum-only binding commit: `38ce926`.
- Model: `gpt-5.6-luna`.
- Canonical train rows: 2,000; candidates per essay: 3; requests: 6,000.
- Accepted: 6,000; failed: 0.
- Batch aggregate usage: 8,838,953 input tokens, 2,628,953 output tokens,
  11,467,906 total tokens, and zero reasoning tokens.
- A separate one-request smoke passed strict participant JSON parsing before
  batch submission.
- Human/reference score read or prompted: false. The model predicted its own
  three axis scores and rationales from only the official system prompt,
  writing prompt, essay, and score-blind candidate instruction.
- Request SHA-256:
  `86a3f091be90775a7810161f657bbc307ff20ef3159eae5631c18de2021b0e07`.
- Validated manifest SHA-256:
  `dbdb6265bd808c6d2e08cb3c05507fd015c2a561e29da495ad2961223ee04c47`.
- Candidate rows SHA-256:
  `1397e870cffdbb66a58d7e2732fadb4e7911bc97af0e7cde17ccf506b90486ac`.
- Official system prompt SHA-256:
  `ea0454665da8e13ffb606c1b0fc7f8323bd62dbad65a9a363e463df888bbe5e9`.

All prompts, essays, mappings, provider envelopes, rationale text, and
candidate rows remain under the ignored restricted data root.

## Frozen model and protocol

The fixed 96-dimensional target-blind feature view contained the complete
39-dimensional within-model Terra view, the complete Luna counterpart, and
18 cross-model summaries: signed and absolute axis-mean delta plus pooled
six-candidate axis mean, standard deviation, minimum, and maximum. Its matrix
SHA-256 was
`eae934ac8f79e38da9a57556368e1e5f90b7785b396494c0a583f9487b32e708`.
Rationale text was not used.

The exact candidates were a fresh alpha-10/cap-0.5 residual ridge, that ridge
plus a high-confidence adjacent 3/4 flip, and that ridge plus an averaged
adjacent/cumulative-`>=4` flip. All ridge systems and optional heads started
fresh for each candidate and fold. No checkpoint was continued.

The same 5-outer by 4-inner protocol and original seven inner gates were used.
All three candidates completed before selection. The selected specification
was freshly refit on four outer-train folds and predicted the locked outer
fold once. Validation and the `average` target were not loaded; V10 itself
made zero API calls.

## Inner selections

| Outer fold | Selected | Ridge macro gain | Equal gain | Low gain | High gain | 3/4 BA gain | Outcome |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | dual-source ridge | 0.010420 | 0.017884 | 0.020130 | 0.042135 | 0.012416 | eligible |
| 1 | dual-source ridge | 0.011777 | 0.021567 | 0.021311 | 0.062317 | 0.016992 | eligible |
| 2 | R0 fallback | 0.008336 | 0.013746 | 0.008799 | 0.040064 | 0.005024 | failed only 3/4 BA |
| 3 | R0 fallback | 0.009967 | 0.020702 | 0.054603 | 0.032355 | 0.008848 | failed only 3/4 BA |
| 4 | R0 fallback | 0.013475 | 0.024492 | 0.058533 | 0.035365 | 0.008362 | failed only 3/4 BA |

The preregistered hard-flip variants did not rescue the failed folds: their
3/4 BA gains were lower than the plain residual's in every outer population.
This is negative evidence against combining the current unweighted flip rule
with the dual-source ridge.

## Final nested metrics

| Metric | Exact R0 | V10 nested selected | Improvement |
| --- | ---: | ---: | ---: |
| Macro continuous RMSE | 0.568780 | 0.563717 | +0.005063 |
| Equal-band RMSE | 0.691549 | 0.683504 | +0.008045 |
| Low-tail `{1,2}` RMSE | 0.923335 | 0.907023 | +0.016312 |
| Score-5 RMSE | 0.884190 | 0.871794 | +0.012396 |
| True-gold 3/4 balanced accuracy | 0.643313 | 0.647493 | +0.004180 |
| Macro Spearman | 0.600288 | 0.605125 | +0.004837 |

Axis RMSE gains were content `0.002443`, organization `0.003841`, and
expression `0.008905`. The 10,000-resample candidate-minus-R0 macro-RMSE
interval was `[-0.008491, -0.001721]`. The final gate passed the bootstrap,
both-tail, no-axis-regression, and Spearman checks, but failed macro-gain and
3/4-BA-gain. Score 1 remained descriptive only.

## Execution and artifacts

GPU0 passed a real 96-train/32-predict smoke, after which outer folds 0--3
ran concurrently on physical GPUs 0--3 and outer fold 4 followed on GPU0.
The append-only ledger binds the exact GPU scope and launch sequence.

- Task card SHA-256:
  `89365b65ce92002f613a9dce81d35f563d82772ca276ab4b8395a94bd894ccc7`.
- Smoke SHA-256:
  `ebbd97e754e4f5a5237a17a8e91e7a67f43910fbea9a9c2bf3934ecb2ca28214`.
- Aggregate SHA-256:
  `6a15396f23f10fe75ff5fd6d88e2542bb5d2898ada04a6b4c79bb123e524df9d`.
- Completion SHA-256:
  `d2c2828719aadbed39646c618bcd3ab2c388e63a8384a044014f2c5f77f624f0`.
- Retained restricted prediction SHA-256:
  `36059045be7b2824e8e2f775495e4dda84a817dcdcc65eb19ac7d7696956bcc9`.

Nineteen focused V10 tests passed before launch; Python compilation, JSON and
shell validation, launcher progress parsing, and scoped diff checks passed.
