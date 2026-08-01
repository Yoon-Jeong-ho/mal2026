# Official Terra 3/4 boundary heads (V8)

Run ID: `iterative-official-boundary-v8-20260802-001`

Status: completed; strict final gate failed, exact R0 retained

Date: 2026-08-02 (Asia/Seoul)

## Decision

V8 implemented the actual 3-vs-4 and cumulative `>=4` auxiliary heads named
in the user's original 20-round plan. It did not relax the V7 gate or reopen a
V7 model. Each candidate freshly fit the fixed V7 alpha-10/cap-0.5 residual,
then fit axis-specific float64 logistic heads from zero initialization using
LBFGS. The three frozen arms were an adjacent 3/4 head, a cumulative `>=4`
head, and their probability average.

The heads increased 3/4 balanced-accuracy gains on several inner populations,
but did not stabilize them on outer-design populations 2 and 3. Those two
folds again fell back to exact R0. The final nested development prediction had
macro RMSE `0.565515`, worse than V7's `0.563253`, and failed the frozen final
gate. Exact R0 remains selected.

## Immutable inputs and protocol

- Preregistration commit: `a7bb66a`.
- Train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`.
- Official score-blind Terra candidate SHA-256:
  `a1791c418c79c0b76399ddb993e862f34209c2da95b0c13f7cda87f403a24e4c`.
- V7 aggregate/completion SHA-256:
  `2cbbfde7a07db8f9844458a722541a50731193a6fd9632e6f2633282509ed68c`
  and `7ac54a9f24f564e84ba2c9f4fd135aadf1586917113e7876560ff817da4cc371`.
- Records: 2,000 train rows, five fixed folds of 400; three model-generated
  Terra scores per essay; no human/reference score was prompted to Terra.
- Validation, rationale text, and `average` were not loaded. V8 made zero
  external API calls.

V8 had no full-OOF tuning pass before preregistration. It was adaptively
motivated by the already disclosed V7 result, so it is still same-train
development evidence, not independent confirmation.

For every outer fold `O`, each other fold served as inner validation `D` and
the remaining three folds were training `S`. All three candidates completed
all four inner predictions before the unchanged seven-gate selection. Only a
selected arm was freshly refit and predicted `O` once. Checkpoints were never
reused.

## Models and execution

All arms used the same 39 official-agent score/consensus features and the same
fresh residual. Axis-specific binary heads used L2 `0.01`, 80 deterministic
LBFGS iterations, and a triangular proximity gate of radius `0.75` around
3.5. Adjacent and threshold arms applied a maximum smooth nudge of `0.15`; the
dual arm used `0.20`.

GPU0 passed a 96-train/32-predict smoke. Outer folds 0--3 ran concurrently on
physical GPUs 0--3 from epoch `1785602361.88` and completed around
`1785602383.90`; outer 4 then ran on GPU0 and completed at `1785602401.92`.
All four first-wave GPUs held about 913 MiB and were sampled at 29--34%
utilization. No conflict or integration failure occurred.

## Inner selection evidence

| Outer | Selected | Adjacent dRMSE | Adjacent dEqual | Adjacent d3/4 BA | Failed gates |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | adjacent | 0.008931 | 0.021234 | 0.017681 | none |
| 1 | adjacent | 0.009605 | 0.024244 | 0.018402 | none |
| 2 | R0 | 0.007045 | 0.017387 | 0.007145 | 3/4 BA |
| 3 | R0 | 0.007489 | 0.022940 | 0.004314 | 3/4 BA |
| 4 | adjacent | 0.013145 | 0.029191 | 0.010974 | none |

The dual arm reached the BA gate on fold 2 (`+0.010047`) but reduced macro
gain to `0.003007`, below the required `0.005`. Thus no preregistered smooth
nudge simultaneously passed both gates on folds 2 and 3.

## Final nested metrics

| Metric | R0 | V8 nested | Improvement |
| --- | ---: | ---: | ---: |
| Macro RMSE | 0.568780 | 0.565515 | +0.003265 |
| Equal-band RMSE | 0.691549 | 0.681800 | +0.009749 |
| Low-tail `{1,2}` RMSE | 0.923335 | 0.904373 | +0.018962 |
| Score-5 RMSE | 0.884190 | 0.849390 | +0.034800 |
| True-gold 3/4 BA | 0.643313 | 0.646331 | +0.003018 |
| Macro Spearman | 0.600288 | 0.604518 | +0.004230 |

All axes and both continuous tails improved, but macro and 3/4 BA gains missed
their `0.010` final thresholds. The paired candidate-minus-baseline RMSE CI
was `[-0.007379, +0.000885]`, which also crossed zero. Score-5 integer recall
gain remained zero. The final prediction artifact consequently contains exact
R0 rather than the nested V8 development prediction.

## Interpretation

The smooth boundary nudge changed too many continuous values for the amount of
integer 3/4 recovery it obtained. The V7 and V8 aggregate evidence supports
only a more selective hypothesis: a high-confidence classifier may flip the
few near-3.5 disagreements while leaving every other strong V7 correction
unchanged. Any such follow-up must be separately preregistered; V8 parameters
and artifacts are frozen.

## Verification and artifacts

Eight focused tests passed before launch; compilation, JSON parsing, and
`git diff --check` passed. Runtime SHA-256 values:

- task card: `0f9d43e06c97f6d71f6edfbdf4ec3d939b70487ef8728f9efd3f885c60974595`;
- smoke: `a44b0ce6e4754159aa8786c3aa30eba9d7ea9a4c91b880b2ee8e1355ec2c777d`;
- aggregate: `97dc01637c16a3448eb6971b03d266b38f7c1aee339add7b474db65866ecae4e`;
- completion: `58af95a4c0a97c167738a375b374edb07b11347737a110e7c0f0fd30648a9a24`;
- retained exact-R0 prediction:
  `36059045be7b2824e8e2f775495e4dda84a817dcdcc65eb19ac7d7696956bcc9`.

Each restricted outer file contains 400 rows and the final restricted file
contains 2,000. No private identifier, essay, prompt, rationale, or writing
text occurred in the public artifact tree.
