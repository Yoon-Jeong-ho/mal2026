# KURE axis-wise ordinal contrastive scoring V1 — preregistration

Status: preregistered; no V1 GPU training result inspected at this commit

Date: 2026-08-02 (Asia/Seoul)

## Authorized question

Test whether three independent KURE encoders can improve Korean-writing score
prediction by organizing each axis representation so that equal scores are
closest, adjacent scores remain moderately close, and distant scores are
farther apart.  Compare continuous-head scoring, score prototypes, interpolated
0.1/0.5 centers, and score-conditioned clustering.  Also compare direct MAL
adaptation with an AI-Hub stage-1 warm start.

This record interprets "AI-Hub stage 1" as the already completed, checksum-bound
48,016-row full-parameter KURE score pretraining artifact.  V1 does not rerun a
new AI-Hub contrastive stage.  The contrastive objective is applied during the
MAL adaptation stage in both arms, isolating the value of the existing AI-Hub
warm start.

## Immutable inputs and privacy

- Git parent before V1 implementation: `3ae5bd24cd1a255b561348eba73c6f9b082f1fc0`.
- KURE: `nlpai-lab/KURE-v1` revision
  `d14c8a9423946e268a0c9952fecf3a7aabd73bd9`.
- Local KURE config SHA-256:
  `852d42e020c7f989c2acaf30fc683b7f768e8c6d1ab17166e835442162bd825d`.
- MAL train: 2,000 rows, SHA-256
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Canonical validation: 400 rows, SHA-256
  `0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4`.
- Exact user-supplied evaluation prompt SHA-256:
  `1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145`.
- AI-Hub full-parameter completion SHA-256:
  `c91704e5a5c5f54b086552731fe87febdaee8c42273f93a5492f1f8626b47959`.
- AI-Hub full model SHA-256:
  `ffdc985d56c655c03e8964927b127b24f0c5bb7fdde8d89e944941f5419cf25a`.
- The AI-Hub artifact was trained on 48,016 manifest-bound records.  Its
  selection/refit details remain in its original completion record.
- Only `content`, `organization`, and `expression` are read. `average` is never
  read, trained, predicted, or evaluated.
- Row text, identifiers, embeddings, predictions, adapters, checkpoints, and
  logs remain under ignored `outputs/`. Only aggregate records are tracked.

The 2,000 MAL train score counts are:

| axis | score 1 | score 2 | score 3 | score 4 | score 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| content | 11 | 202 | 982 | 749 | 56 |
| organization | 26 | 252 | 708 | 803 | 211 |
| expression | 4 | 77 | 533 | 1,093 | 293 |

Score 1 is therefore descriptive only, particularly for expression. Low-tail
claims use `{1,2}` as a group.

## Fixed arms and three-model contract

Exactly two arms are evaluated:

1. `base`: pinned public KURE backbone.
2. `aihub_full_backbone`: the strict `backbone.*` tensors from the completed
   AI-Hub full-parameter model; its old three-score head is discarded.

Each arm trains three completely separate models, one for each analytic axis.
Models share configuration but not adapters, projection heads, score heads,
optimizer state, batches, labels, or checkpoints. Each axis model reads only
its own gold axis.

The backbone uses KURE's official CLS pooling followed by L2 normalization.
Fresh LoRA adapters target `query`, `key`, `value`, and `dense` with r=16,
alpha=32, dropout=0.05. A fresh 1,024→256 projection and independent continuous
and four-threshold ordinal heads are trained. Numeric mode is FP32 with TF32
matrix multiplication because prior repository evidence found non-finite KURE
training in reduced precision.

## Fixed data isolation and selection

For each axis and arm, the canonical 2,000-row train population is divided by
the existing prompt-stratified, document-isolated deterministic splitter into
1,600 selection-train and 400 selection-dev rows. Validation must not be loaded
until epoch selection and a fresh all-2,000 refit complete.

Selection trains epochs 1 through 6 and selects the epoch with the lowest
selection-dev RMSE of the fixed hybrid scorer, with higher Spearman and then
earlier epoch as deterministic ties. The chosen epoch is replayed from the
same arm initialization and seed on all 2,000 train rows. Prototypes and
clusters are rebuilt only from those 2,000 refit embeddings. Canonical
validation is then read once for a descriptive final report. It is not an
untouched confirmation set because it was exposed in older project work.

Seeds are 2026080201/02/03 for content/organization/expression. Maximum input
length is 1,536; the prior audit found train maximum 1,421, so no MAL input is
truncated. Selection epochs, scorer type, temperature, and centering are not
retuned after validation.

## Balanced batches and loss

The deterministic per-batch exact-score quota is `[2,4,5,5,4]` for scores
1..5 (batch 20), without duplicate document rows inside a batch. Regression
and CORAL per-row losses use original-distribution importance weights so the
balanced sampler does not redefine the population target. Pairwise losses use
the balanced batch deliberately. Gradient accumulation is 2, but contrastive
pairs exist only within each physical microbatch.

For normalized projected vectors `z_i`, axis scores `y_i`, cosine similarity
`s_ij`, and distance `d_ij=|y_i-y_j|`, the continuous ordinal similarity target
is:

`target(d) = cos(pi*d/4)` for `d in {0,1,2,3,4}`.

This gives identical/adjacent/two-apart/three-apart/opposite targets of
1.0/~0.707/0/~-0.707/-1. The contrastive term is a pairwise robust regression
to this target. The rank term mines, inside each current fit-only batch, the
most similar label-distant example and enforces the ordering
same-score > adjacent-score > distant-score. Same-document pairs are forbidden.
No validation or held-out embedding enters mining, a queue, a centroid, or a
cluster.

The fixed objective is:

- SmoothL1 continuous score loss, weight 1.00, beta 0.25;
- four-threshold CORAL BCE, weight 0.30;
- ordinal cosine contrastive loss, weight 0.20, temperature 0.10;
- rank/hard-negative loss, weight 0.10, rank margin 0.10.

Configured compact/soft-close/far margins are 0.05/0.10/0.20. Optimizer is
AdamW, LR 5e-5, weight decay 0.01, warmup 0.10, maximum gradient norm 1.0.

## Fixed inference comparison

Every trained axis representation reports all of the following without
validation-time calibration:

1. bounded continuous head;
2. soft expected score over train-only label centroids 1..5, T=0.10;
3. primary hybrid: 0.5 continuous + 0.5 soft prototype;
4. nearest normalized linearly interpolated label prototype on a 0.5 grid;
5. the same on a 0.1 grid;
6. score-conditioned spherical k-means, k=2 per score where support allows,
   with deterministic empty-cluster recovery and soft expected score.

Because MAL gold labels are integers, 0.1/0.5 are interpolation grids at
inference, not new supervised labels. This distinction must be retained in the
result. The primary selection metric is method 3. Methods 1, 2, and 4–6 are
fixed diagnostic ablations and may not be selected post hoc as the declared V1
winner.

## Metrics and interpretation

For each method report axis/macro continuous RMSE and Spearman, half-up integer
RMSE/accuracy/recall/one-off, `{1,2}` RMSE, score-5 RMSE, true-gold 3/4 balanced
accuracy, 3→4 and 4→3 errors, centroid support/cosine separation, and sampler
coverage/duplicates. Aggregate both arms only after all three axis results are
complete.

Historical KURE direct evaluation-prompt regression (`0.6418559395` descriptive
validation macro RMSE) is a context baseline. Exact R0 OOF (`0.5687802170`) is
reported separately and is not directly comparable to the validation number.
A V1 arm is called an internal improvement only if its three-axis selection-dev
hybrid macro RMSE is lower than the matching base arm/control. Validation
results are descriptive; they cannot establish independent generalization or
a hidden-benchmark improvement.

## Resource and run contract

- Existing `.venv-standard`; no package installation or environment creation.
- Authorized/default GPU scope: physical GPUs 0--3. GPU0 runs the smallest real
  smoke first. After it passes, the three axis jobs may run concurrently on
  GPUs 0, 1, and 2; GPU3 is available for an independent diagnostic or the next
  queued arm, but no process is displaced merely to maximize utilization.
- Output root: `outputs/kure-axis-ordinal-contrastive-v1/`.
- Preserve every smoke, failure, selected epoch, refit, aggregate, and negative
  result. Never overwrite an existing run directory.
- Commit reproducibility code/config/aggregate documentation locally; do not
  push. The user will push later.
