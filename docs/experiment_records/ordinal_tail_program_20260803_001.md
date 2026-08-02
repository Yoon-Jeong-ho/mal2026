# Ordinal tail program — 2026-08-03 run 001

## Status and authority

- **Run ID:** `ordinal-tail-program-v1-20260803-001`
- **Parent Git SHA:** `8b6cdd06d5a7692540a631d72e9ec4d85088a96b`
- **Authorization:** the user approved the end-to-end ordinal-tail plan and asked
  that RMSE be driven as close to `0.4` as the evidence permits.
- **GPU scope:** GPUs 0–3 only; GPU 0 is the smoke-test device.
- **Selection population:** train-only nested OOF (five fixed outer folds, four
  inner folds). The repeatedly exposed validation split is descriptive only.
- **Targets:** `content`, `organization`, and `expression` are learned and
  evaluated independently. The supplied `average` field is forbidden as a
  feature, target, calibration input, or selection metric.

The `0.4` macro RMSE is a stretch target rather than an expected result. It
would require a 29.7% RMSE reduction (about a 50.5% MSE reduction) from the
exact R0 OOF baseline. Each negative result is retained instead of changing a
protocol after observing its validation performance.

## Immutable inputs

| Input | SHA-256 |
|---|---|
| `eval/train.jsonl` | `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737` |
| frozen R0 embedding rows | `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16` |
| exact R0 OOF predictions | `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04` |

The frozen-feature screen uses the existing 4,096-dimensional, checksum-bound
R0 Qwen embedding artifact. This is a recorded deviation from the initial
draft's proposed new KURE feature cache: it provides the strongest exact OOF
baseline comparison without generating another representation. KURE remains
the backbone for the promoted end-to-end candidates.

## Stage 0 diagnostic

Command:

```bash
PYTHONPATH=src .venv-standard/bin/python \
  scripts/run_ordinal_tail_diagnostics.py \
  --config configs/ordinal_tail_program.v1.json
```

Aggregate output is stored in the ignored run directory. No essay, prompt,
identifier, embedding, or row prediction is copied into this record.

| Metric | Exact R0 OOF |
|---|---:|
| macro RMSE | 0.568780 |
| equal-group RMSE | 0.691549 |
| low-tail `{1,2}` RMSE | 0.923335 |
| score-5 RMSE | 0.884190 |
| gold-3/4 balanced accuracy | 0.643313 |
| macro Spearman | 0.600288 |

Axis RMSE is `0.509024` content, `0.688392` organization, and `0.508925`
expression. Rounded R0 predictions contain no score-5 predictions in any axis
and only one score-1 prediction overall. Raw labels are fractional in 1,812
content, 1,453 organization, and 1,455 expression rows, so a raw-score MSE
auxiliary is retained rather than replacing the target with integer classes.

## Preregistered stages

1. **Fixed-feature screen.** Compare natural CE, RPS, CORAL, CORN, SLACE at
   alpha 0.5/1/2, effective-number CE at beta 0.99/0.999, and sqrt-tempered
   sampling. Weighted/sampled posteriors are corrected back to the natural
   train-fold prior for the primary RMSE. No correction methods are stacked.
2. **End-to-end KURE/cRT.** Train only the best two distinct loss families from
   the nested screen, preserving fold isolation and the public/AI-Hub
   provenance boundary.
3. **Prompt-reference NPCR.** Compare prompt- and trait-local score-difference
   learning with adjacent and skip-gap pairs; all references are fit-fold
   local.
4. **Calibration/limited ensemble.** Fit axis-wise calibration and any ensemble
   weights only from OOF predictions. Keep an unchanged R0 candidate.
5. **Final refit and descriptive validation.** Freeze the protocol, refit the
   selected candidates, and read validation only after all selection is over.

The promotion gate requires at least `0.005` macro RMSE improvement, no more
than `0.01` RMSE degradation on any axis, no low-tail or score-5 degradation,
no more than `0.01` loss in gold-3/4 balanced accuracy, no more than `0.005`
loss in Spearman, and a positive paired-bootstrap improvement lower bound.

## Environment note

The existing `.venv-standard` environment does not contain `pytest`. Package
installation is prohibited, so test modules are executed with Python's
standard `unittest` runner. This environment limitation and the failed pytest
probe are preserved in the append-only ignored run ledger.
