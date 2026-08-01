# Cross-fitted directional tail falsification (V6)

Run ID: `iterative-tail-directional-v6-20260802-001`

Status: completed, no candidate promoted; exact R0 OOF baseline retained

Date: 2026-08-02 (Asia/Seoul)

## Decision

V6 tested whether strictly cross-fitted, identity-default directional experts
could break the tail-versus-center trade-off observed in V1--V5. All five
outer folds completed their three preregistered candidates and independently
fell back to exact R0. The concatenated final prediction is therefore
byte-for-byte the baseline decision, with macro continuous RMSE `0.568780`.

The experiment is a negative falsification result. In particular, the
nonlinear experts consistently improved continuous score-5 RMSE and 3/4
balanced accuracy, but consistently worsened the `{1,2}` low tail and did not
recover any score-5 integer recall. The linear control made only very small
changes and lost 3/4 balanced accuracy. No candidate approached the complete
AND promotion gate.

## Frozen protocol and inputs

- Preregistration Git commit: `5e636f2`.
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Exact R0 OOF prediction SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`.
- Frozen Qwen3-Embedding-8B row SHA-256:
  `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16`.
- Score-blind evidence feature SHA-256:
  `c2770d2add3b08a46614ad4c56fddc2d6b06ba5784c39e80c95c9f419e46d7db`.
- Fixed 4,672-to-64 Rademacher projection SHA-256:
  `1ef94e113b8c32e77017f9afbc020fdfd886426fc199ef4d6fc44f8e2f8f493b`.
- Records: 2,000 train rows in five fixed folds of 400.
- Targets: `content`, `organization`, and `expression` only; `average` was
  rejected by the data contract.
- Validation was not loaded. Historical V1--V5 aggregate records informed the
  preregistration but no historical row prediction, row error, learned weight,
  checkpoint, or pseudo-target was a model input.
- No external API call was made in this run.

For each sealed outer fold `O`, each of the other four folds served once as
inner validation `D`; the remaining three original folds were `S`. Each
candidate was freshly initialized, internally cross-fitted on the original
folds of `S` to obtain out-of-fold expert-benefit labels, and predicted `D`
once. Candidate selection occurred only after all three candidates covered all
1,600 outer-train rows. When no candidate passed every gate, the fold used the
exact R0 prediction. No outer gold was available before selection freeze and
outer prediction.

The three fixed candidates were a primary nonlinear directional residual, a
more conservative nonlinear variant, and a linear safety control. They used
identity, low, high, and center routing; high corrections were allowed to cross
4.5, avoiding the structural score-5 cap found in V5.

## Execution evidence

The existing `.venv-standard` environment was used. GPU0 passed the smoke
gate before the full run. Outer folds 0--3 launched concurrently on physical
GPUs 0--3; after GPU0 completed outer 0, it processed the remaining outer 4.
The full stage completed 15 candidate evaluations and 60 fresh inner
fit/predict operations. No pre-existing process was terminated or displaced.

Command:

```bash
.venv-standard/bin/python scripts/run_iterative_tail_directional.py --launch
```

Ledger times show preflight at epoch `1785600616.599`, smoke completion at
`1785600630.064`, all first-wave outer jobs started by `1785600630.081`, and
outer 4 completed at `1785600750.116`. The aggregate and completion marker
were written at approximately `1785600760.85`.

Smoke audited 96 training rows over three original folds and 32 prediction
rows. All model state hashes changed from initialization, checkpoint reuse was
false, and the projection hash matched the preregistration.

## Inner-selection results

The table reports the unweighted mean of the five outer-train inner-OOF
comparisons. Positive deltas mean improvement over exact R0.

| Candidate | d macro RMSE | d equal-band RMSE | d low `{1,2}` RMSE | d score-5 RMSE | d 3/4 BA | d score-5 recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nonlinear primary | +0.000029 | +0.001593 | -0.008976 | +0.018202 | +0.006041 | 0.000000 |
| nonlinear conservative | -0.000145 | +0.001446 | -0.010012 | +0.019079 | +0.005792 | 0.000000 |
| linear safety control | +0.000183 | +0.002578 | +0.000361 | +0.015077 | -0.001750 | 0.000000 |

The primary and conservative candidates improved high-tail RMSE and 3/4 BA in
all five inner-OOF populations, but worsened low-tail RMSE in all five. The
linear control improved macro RMSE in all five by a mean of only `0.000183`,
far below the `0.005` promotion threshold; it also worsened 3/4 BA in all five.
Every candidate failed the required equal-band gain, macro gain, 3/4 BA gain,
and score-5 recall gain on every outer selection population. No score-5 recall
gain occurred in any of the 15 candidate cells.

The learned routing explains the trade-off. Mean inner-prediction identity
weights were `0.335528`, `0.355568`, and `0.705998` for the primary,
conservative, and linear candidates respectively. The nonlinear candidates
assigned approximately 40--42% mean mass to the low expert and 24% to the
high expert. Their average low correction was about `-0.336` and high
correction about `+0.642`; the low router did not identify low-gold examples
reliably enough to offset the global cost.

## Final metrics and gate

Because every outer fold selected the baseline fallback, final metrics equal
the exact R0 OOF metrics:

| Metric | Final value |
| --- | ---: |
| Macro continuous RMSE | 0.568780216992 |
| Macro Spearman | 0.600288438665 |
| Equal-band RMSE | 0.691549051599 |
| Low-tail `{1,2}` RMSE | 0.923334876486 |
| Score-5 RMSE | 0.884189926190 |
| True-gold 3/4 balanced accuracy | 0.643312588962 |
| 3->4 error rate | 0.305836275150 |
| 4->3 error rate | 0.356964650393 |
| Score-1 descriptive RMSE | 1.164567011818 |

The final macro improvement is zero. The 10,000-resample paired candidate
minus baseline bootstrap interval is the tautological `[0, 0]`; its upper
bound is not below zero. The final gate failed macro improvement, both tail
improvements, 3/4 BA improvement, score-5 recall improvement, and strict
bootstrap superiority. Score 1 remained descriptive only, with train counts
content 11, organization 26, and expression 4.

## Interpretation and stop rule

V6 rejects the specific hypothesis that these cross-fitted directional
experts can simultaneously improve global error, equal-band error, both tails,
3/4 separation, and score-5 recovery from the frozen R0 embeddings and the
existing score-blind evidence features. It does not prove that all tail-aware
models are ineffective.

V6 was designed after inspecting V1--V5, so this is adaptive same-train
falsification evidence rather than independent confirmation. The registered
failure action freezes V6 and ends further tuning that reuses the same train
population **and the same feature sources**. A materially new feature source,
new independently labeled data, or genuinely unseen evaluation population
requires a separately registered study; V4--V6 results and learned states must
not be reopened or retuned.

## Verification and artifacts

Focused verification:

```bash
PYTHONPATH=src .venv-standard/bin/python -m unittest -q \
  tests.test_iterative_tail_directional_models \
  tests.test_iterative_tail_directional_protocol \
  tests.test_iterative_tail_directional_runner \
  tests.test_iterative_tail_directional_selection
```

Result: 23 tests passed in 39.917 seconds. `py_compile` and `git diff --check`
also passed.

Ignored runtime evidence:

- Public aggregate root:
  `outputs/iterative-tail-directional-v6/iterative-tail-directional-v6-20260802-001`.
- Restricted row root:
  `data/processed/restricted/iterative_tail_directional_v6/iterative-tail-directional-v6-20260802-001`.
- Task-card SHA-256:
  `071babecf56f32b471af8659925f727c42c92df8535bf24f4bd346be0ffb3244`.
- Smoke SHA-256:
  `8b21562cc1d0c1ff050cf7ad0b5ebc334b79383461ea826a73fbe718232aa77d`.
- Aggregate SHA-256:
  `39d31011a4a27564f9501ee6f94c41e0b5dba490774583a3859369298b15042d`.
- Completion SHA-256:
  `3585df5a2ada491af57e8997dc277c719feda9ccefb01d50d225a7841fff075e`.

Each restricted outer prediction file contains exactly 400 rows. A recursive
public-artifact key audit found no source identifier, essay text, writing
text, or rationale field.
