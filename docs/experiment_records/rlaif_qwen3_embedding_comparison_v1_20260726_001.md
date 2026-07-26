# RLAIF Qwen3-Embedding two-initialization comparison v1 — 2026-07-26-001

## Authorized task card

The user clarified that the rationale-conditioned score regressor must use the
embedding-tuned `Qwen/Qwen3-Embedding-8B` checkpoint rather than the earlier
Qwen2.5-Instruct decoder backbone, and authorized two otherwise identical
training runs:

1. initialize directly from the immutable public Qwen3-Embedding-8B snapshot;
2. initialize from the completed Qwen3-Embedding-8B model previously trained
   on 48,016 eligible AI-Hub human-feedback records, then continue on the same
   rationale-conditioned data.

Both arms use the already completed, independently selected
`rank2_ax4_random1` rationale source: 2,000 train rationales and 400 validation
rationales.  The input contains the writing prompt, student essay, and three
rationale strings.  Supervised targets are exactly `content`, `organization`,
and `expression`; the prior AI-Hub model's fourth `average` head is discarded
before continuation and no average target or prediction is used.

The public snapshot is pinned to
`Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.
The warm-start is bound to the prior completion metadata and model-state
checksum.  Its first three regression-head rows and all trained LoRA tensors
are restored over the same immutable base; the large saved copy of frozen base
weights is not redundantly loaded into memory.

## Fixed execution contract

- Last-nonpad pooling and L2-normalized Qwen3 embedding, followed by an
  unbounded three-output linear regression head and raw MSE.
- LoRA rank 16 / alpha 32 / dropout 0.05 on Q/K/V/O and gate/up/down
  projections; BF16, TF32, learning rate `1e-4`, weight decay `0.01`, warmup
  ratio `0.05`, seed `2026072601`.
- Twelve epochs, global batch 64 (`4` examples/GPU × four-process DDP × four
  accumulation steps), expected 384 optimizer updates.  Validation uses batch
  8/GPU.
- One short GPU0 gate for each initialization (four examples, one update),
  followed automatically by the fixed full DDP runs and validation on GPUs
  0--3.  GPUs 4--7 are neither queried nor used.
- Aggregate-only persistence under ignored `outputs/`; essays, rationale text,
  IDs, row predictions, and generated data remain in restricted/ignored roots.

The runner is `scripts/run_rlaif_qwen3_embedding_comparison_v1.py`.  It records
the Git SHA, exact configs/commands, environment, GPU authorization, source and
model bindings, finite-loss/state gates, 400-example validation population,
and per-axis RMSE/Spearman.  The three-axis macro is a diagnostic only, not an
average-score target.  Because the same canonical validation split informed
earlier model/rationale selection, this comparison is descriptive and will not
be used for further hyperparameter tuning.

## Result status

Implementation and GPU-free contract checks passed.  The existing environment
does not include pytest, so the seven directly relevant test functions were
imported and executed with `.venv-standard`; all seven passed, as did Python
byte-compilation and both-arm config validation.  Actual GPU0 gate, full
training, and validation evidence will be appended after the durable runner
completes.

The first actual runtime (`20260726-001`) is preserved as a negative
integration result.  The public-base four-example update passed (finite loss
`15.713469`), but the warm-start arm stopped before full execution because the
temporary four-head construction retained a four-label forward contract after
the average row was removed.  No result was selected and no scientific
variable changed.  Recovery runtime `20260726-002` constructs the declared
three-head model directly, loads the AI-Hub LoRA tensors, slices only the first
three rows of the source head, and shape-checks every restored trainable tensor.

## Completed recovery result

Recovery runtime `20260726-002` completed successfully at
`2026-07-26T12:40:59Z`.  Both GPU0 gates passed, both full arms consumed the
same 2,000 train examples for 384 DDP updates, and both evaluations covered 400
unique validation essays with one prediction per essay.  Every persisted
training/evaluation contract reports exactly the three requested fields and
`average_target_used: false`.

| initialization | content RMSE / Spearman | organization RMSE / Spearman | expression RMSE / Spearman | three-axis diagnostic RMSE / Spearman |
| --- | ---: | ---: | ---: | ---: |
| public Qwen3-Embedding base | 1.884778 / -0.103866 | 2.029515 / 0.008927 | 2.273013 / 0.001607 | 2.062436 / -0.031111 |
| AI-Hub 48,016-row Qwen3-Embedding warm-start | **0.531378 / 0.582048** | **0.710936 / 0.578077** | **0.531244 / 0.604543** | **0.591186 / 0.588222** |
| prior Qwen2.5-Instruct rationale baseline | 0.590273 / 0.543862 | 0.769708 / 0.580551 | 0.629532 / 0.368061 | 0.663171 / 0.497492 |

The warm-start is the unambiguous winner of the two Qwen3 arms.  Against the
same-rationale Qwen2.5 baseline, its diagnostic RMSE is lower by `0.071985`
(`10.85%` relative), and its diagnostic Spearman is higher by `0.090731`.
RMSE improves on all three axes: `0.058896` content, `0.058772` organization,
and `0.098289` expression.  Organization remains the largest-error axis, and
the `0.591186` diagnostic RMSE does not reach the requested `0.421300` level.

The public-base failure is a valid negative result for this fixed one-seed,
384-update schedule, not evidence that the embedding checkpoint is generally
incapable.  Unlike the warm arm it began with a random score head and no prior
score-regression LoRA state; the large validation error and near-zero rank
correlation show that this schedule did not learn a usable scorer from that
initialization.  No post-validation retuning was performed.

Each full training run took about 1,011--1,012 seconds.  Representative
steady-state observations showed all four H100s active, commonly 97--100% GPU
utilization and roughly 78.9--80.7 GiB HBM used per device; there was no OOM.
The aggregate final report is ignored at
`outputs/aggregate-reports/rlaif-qwen3-embedding-comparison-v1-20260726-002.final-summary.json`
(SHA-256 `a7b428e8adcab0e16b25b84a8d6192c8348aef46b3bec8b082678724501f8455`).
