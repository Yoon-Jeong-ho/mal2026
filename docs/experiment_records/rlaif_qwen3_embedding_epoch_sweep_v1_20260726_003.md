# Qwen3-Embedding warm-start epoch sweep v1 — 2026-07-26-003

## Authorized task card

After the Qwen3-Embedding two-initialization comparison selected the AI-Hub
48,016-row warm-start, the user explicitly authorized repeating that arm while
saving the model after every epoch and evaluating all saved epochs.

- **Arm:** only `qwen3_aihub_warmstart`; the failed public-base scorer is not
  extended.
- **Data:** the same 2,000 train writings and A.X `random1` rationales; the
  same 400 canonical validation writings are evaluated once per checkpoint.
- **Targets:** exactly `content`, `organization`, and `expression`.  The old
  AI-Hub `average` head remains discarded and no average target is read or
  emitted.
- **Training:** the prior fixed seed and optimizer schedule are unchanged:
  12 epochs, 32 optimizer updates per epoch, 384 updates total, global batch
  64, BF16/TF32, LoRA rank 16, and maximum length 2,048.
- **Checkpoints:** the reconstructable LoRA tensors and three-output head are
  saved after steps 32, 64, ..., 384.  Each checkpoint is approximately 201
  MiB and is bound to the immutable base snapshot; redundant 17-GiB frozen
  base copies and optimizer states are not stored because evaluation is the
  only declared checkpoint consumer.
- **Evaluation:** all 12 checkpoints are loaded in one DDP evaluation process
  and evaluated independently on 400 unique validation essays, one prediction
  per essay per checkpoint.  Only aggregate C/O/E RMSE and Spearman are saved.
- **Resources:** the new save/load path receives one four-example GPU0 train
  and evaluation gate, then the full train and evaluation use DDP on GPUs
  0--3.  GPUs 4--7 are neither queried nor used.

The canonical validation set was already exposed in earlier selection and is
now explicitly reused at the user's request.  Therefore the lowest-RMSE epoch
is a descriptive validation-selected checkpoint, not an untouched
generalization estimate.  No hyperparameter is changed after viewing its
curve.  Generated text, inputs, identifiers, row predictions, checkpoints, and
logs remain in ignored/restricted roots.

## Result status

The durable runner completed at `2026-07-26T13:29:15Z`.  The GPU0 checkpoint
save/reload/evaluation gate passed, full DDP training completed all 384 updates,
and exactly 12 checksummed epoch states were preserved (about 2.4 GiB total on
disk).  Every checkpoint was evaluated on the declared 400 unique validation
essays.  All reports contain only the three requested score fields and state
`average_target_used: false`.

| epoch (step) | content RMSE | organization RMSE | expression RMSE | three-axis RMSE | three-axis Spearman |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (32) | 0.519832 | 0.699950 | 0.499505 | 0.573095 | 0.623453 |
| 2 (64) | 0.517878 | 0.709132 | 0.506323 | 0.577778 | **0.635958** |
| **3 (96)** | **0.512864** | 0.695580 | **0.499289** | **0.569245** | 0.625288 |
| 4 (128) | 0.531223 | **0.690919** | 0.500508 | 0.574217 | 0.630892 |
| 5 (160) | 0.544506 | 0.709148 | 0.508675 | 0.587443 | 0.619837 |
| 6 (192) | 0.535585 | 0.713171 | 0.510540 | 0.586432 | 0.608322 |
| 7 (224) | 0.539021 | 0.710234 | 0.518678 | 0.589311 | 0.604397 |
| 8 (256) | 0.529708 | 0.714360 | 0.527207 | 0.590425 | 0.601893 |
| 9 (288) | 0.527441 | 0.705949 | 0.518117 | 0.583836 | 0.605349 |
| 10 (320) | 0.524144 | 0.710805 | 0.513706 | 0.582885 | 0.596216 |
| 11 (352) | 0.526666 | 0.712324 | 0.522655 | 0.587215 | 0.590618 |
| 12 (384) | 0.529242 | 0.713580 | 0.523874 | 0.588899 | 0.591004 |

The declared RMSE-first rule selects **epoch 3 / step 96**.  Its complete
metrics are content `0.512864 / 0.621518`, organization
`0.695580 / 0.597796`, and expression `0.499289 / 0.656549` RMSE / Spearman;
the three-axis diagnostic is `0.569245 / 0.625288`.  Epoch 2 has the highest
macro Spearman, while epoch 4 has the lowest organization RMSE.  Thus the
useful region is consistently early (epochs 1--4), rather than a single late
fluctuation.

Compared with the earlier fixed epoch-12 run (`0.591186`), epoch 3 lowers the
diagnostic RMSE by `0.021941` (`3.71%`).  The repeated epoch-12 result is
`0.588899`, only `0.002287` from that earlier run, so the checkpoint-saving
path did not materially change the endpoint.  Relative to its own epoch 12,
epoch 3 is lower by `0.019654` (`3.34%`).  Twelve epochs were not an invalid
run, but the validation curve indicates that continuing past epoch 4 gradually
reduced rank correlation and did not improve RMSE under this protocol.

The selected macro RMSE remains `0.147945` above the requested `0.421300`
level, and organization remains the largest-error axis.  Because the canonical
validation set selected the epoch, `0.569245` is not an unbiased held-out
estimate.  The selected reconstructable checkpoint is retained at
`outputs/rlaif-qwen3-embedding-epoch-sweep-v1/rlaif-qwen3-embedding-epoch-sweep-v1-full-003/epoch_checkpoints/epoch-03/`.
The aggregate final report is ignored at
`outputs/aggregate-reports/rlaif-qwen3-embedding-epoch-sweep-v1-20260726-003.final-summary.json`
(SHA-256 `4607b5ffdedcea4aeef767041952e06d53b071bcc04dfb1f5a983b6142584d5e`).
