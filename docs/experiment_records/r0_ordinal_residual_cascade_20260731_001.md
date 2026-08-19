# R0 ordinal residual and adjacent-cascade feasibility — 2026-07-31-001

- **Status:** completed negative experiment; the residual classifier is not
  selected for deployment.
- **Question:** determine whether a held-out R0 score of 3 or 4 carries enough
  calibrated signal to justify a second-stage classifier, including the
  proposed `3 -> {1/2, 3} -> {1, 2}` and `4 -> {4, 5}` hard routes.
- **Authorization/GPU scope:** the current user explicitly requested model
  training, experiments, literature review, and subagent review. Physical GPUs
  0--3 were used; no GPU outside the repository default scope was used.
- **Git/environment:** launch Git SHA
  `a96f029b1ea640878607d4eb7bd817e6099334c1`; the pre-existing worktree was
  dirty and preserved. Existing `.venv-standard`, PyTorch `2.11.0+cu130`, and
  the maintained Hugging Face Trainer path were used. No package or environment
  was installed or created.
- **Data/privacy:** canonical train and validation checksums are respectively
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
  and `0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4`
  as bound by the canonical loader and held-out R0 aggregate.
  All row-level predictions, identifiers, rationales, and embeddings remain in
  ignored `data/processed/restricted/`; only aggregate metrics and provenance
  are recorded here. The average target was not used.

## Protocol

1. Reproduced the exact historical R0 arm with five leakage-free train folds.
   Each fold used Qwen3-Embedding-8B revision
   `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, the AI-Hub warm start,
   score-blind `rank2_ax4_random1` rationale, LoRA r=16/alpha=32, maximum length
   2048, learning rate 1e-4, four epochs, and an epoch-1--4 arithmetic ensemble.
   Each model trained on 1,600 essays and predicted only its 400 held-out
   essays. Fold seed was `2026073101`; effective batch size was 64.
2. Generated one frozen, L2-normalized 4,096-dimensional public Qwen3 embedding
   for each train/validation input. The embedding text included the original
   input contract and rationale; train base scores came only from exact OOF
   models and validation scores came from the already fixed held-out R0
   ensemble.
3. Trained a single shared residual model with three axis-specific 5-way heads,
   rather than nine independent backbones. The head received frozen embeddings
   plus the base continuous score, and used cross-entropy with a 0.25-weighted
   continuous auxiliary loss. Three seeds (`2026073101`--`03`) were compared on
   one fixed 1,600/400 train-only split. Temperature and soft blend weight,
   including a no-change weight of 0, were selected before the held-out
   validation set was evaluated once. The selected seed was then refit from a
   fresh initialization on all 2,000 train rows.

Representative commands:

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src .venv-standard/bin/python \
  scripts/run_r0_exact_oof_fold.py \
  --run-id r0-exact-oof-20260731-002 --fold <0..4> --physical-gpu <0..3>
PYTHONPATH=src .venv-standard/bin/python scripts/aggregate_r0_exact_oof.py \
  --run-id r0-exact-oof-20260731-002
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=src .venv-standard/bin/python \
  scripts/build_r0_residual_embeddings.py shard \
  --artifact-run-id r0-public-frozen-embedding-20260731-001 \
  --split <train|validation> --prediction-run-id r0-exact-oof-20260731-002 \
  --shard-index <0..3> --physical-gpu <0..3>
PYTHONPATH=src .venv-standard/bin/python scripts/build_r0_residual_embeddings.py \
  merge --artifact-run-id r0-public-frozen-embedding-20260731-001 \
  --split <train|validation>
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv-standard/bin/python \
  scripts/run_r0_ordinal_residual.py \
  --config outputs/r0-ordinal-residual-v1/configs/r0-ordinal-residual-20260731-001.json
```

For validation embedding shards, `--prediction-run-id` is deliberately omitted
because the runner binds validation to the fixed held-out R0 ensemble.

## Results

- Exact train OOF R0: macro axis RMSE `0.568780`, macro axis Spearman
  `0.600288` over 2,000 essays. Merged restricted prediction checksum:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`.
- Existing validation R0: macro axis RMSE `0.558294`; pooled RMSE across 1,200
  axis rows `0.565345`; pooled integer accuracy `0.5900`; pooled QWK `0.548544`.
- When the existing R0 emitted integer 3, only `55.16%` were truly 3
  (`562` routed axis rows). When it emitted 4, `63.29%` were truly 4 (`572`
  routed axis rows). The proposed one-direction tree would ignore `30.4%` of
  score-3 routes whose truth is 4/5 and `20.3%` of score-4 routes whose truth
  is 1/2/3.
- The residual posterior's validation conditional accuracy was `56.65%` when
  it predicted class 3 and `61.30%` when it predicted class 4. Overall
  10-bin confidence ECE was `0.04975`, but content and expression were less
  calibrated (`0.07186` and `0.07072`).
- Most importantly, leakage-free train-dev selection chose blend weight
  **`0.0`**, posterior temperature `1.0`, and seed `2026073103`. Thus the safe
  action was to leave the base R0 score unchanged; held-out RMSE improvement was
  exactly zero. The saved classifier checkpoint is an experimental artifact,
  not a deployment candidate.
- Earlier five-fold affine calibration also worsened macro continuous RMSE
  from `0.558294` to `0.561230`. The result rejects global post-hoc affine or
  threshold correction for this model lineage.

## Interpretation and decision

The regression output is under-dispersed, especially for organization, but its
global bias and slope are already close enough that recalibration cannot supply
missing ranking information. The 3/4 ambiguity is real, yet a frozen semantic
embedding plus base-score classifier did not add leakage-free RMSE signal.
Moreover, the train tails are too small for reliable standalone 1-vs-2 heads
(class-1 counts by axis: content 11, organization 26, expression 4).

Do not replace R0 and do not deploy the literal hard cascade. Any follow-up
must preserve OOF selection and should first target genuinely new tail signal
(additional adjudicated low/high examples, or a joint ordinal/continuous
fine-tune) rather than oversampling the same few tail rows or forcing calibrated
probabilities into score changes.

## Artifacts and verification

- OOF aggregate:
  `outputs/r0-exact-oof-v1/r0-exact-oof-20260731-002/merged/aggregate_metrics.json`
  (SHA-256 `6e59278351b534b7e3ced34a9eb63f3c2dcf01254fd2b635a63946a29c2c179c`).
- Frozen train/validation row checksums:
  `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16`
  and `e398d79ffeea7d34e12cfcaac79fae7f6ac3d5c17ab9d5744f521b7e38e9e5f1`.
- Residual config checksum:
  `6d8c7cd430c33ce4d5c8219f524038fc3d2f818b54f4e97d39e9ff9db1514d75`.
- Residual aggregate:
  `outputs/analysis/r0-ordinal-residual-20260731-001/aggregate.json`
  (SHA-256 `f13af12dba365d829512738f61739076f646485cc71485b5a09bc926629b2f09`).
- The first OOF launch failed before GPU training because gradient
  checkpointing was incompatible with the custom regressor. The failure was
  preserved; the replay disabled only that flag, matching the historical R0
  protocol. No scientific variable was changed.
- Contract/unit suite: 30 tests passed; Python compilation and Git whitespace
  checks passed.
