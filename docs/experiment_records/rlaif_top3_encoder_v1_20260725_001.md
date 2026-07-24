# RLAIF top-three rationale / encoder regression v1 — 2026-07-25-001

## Authorized task card

Following completion of the RLAIF/GRPO prompt-ensemble v8 matrix, the user
authorized the remaining end-to-end stages: generate rationales from the top
three complete adapters, then train and validate one encoder per rationale
source.  The protocol is intentionally separate from the earlier API-rationale
score-regression matrix.

- **Decoder selection:** exactly the three complete `bundle` adapters sorted
  by their frozen-v6 macro score: (1) Midm-2.0-Base / `random1` (4.189100),
  (2) A.X-4.0-Light / `random1` (4.187033), and (3) A.X-4.0-Light / `all5`
  (4.184067).
- **No source combination:** `all5` is the RLAIF reward construction for its
  own adapter, not an inference ensemble.  Each adapter independently
  generates one train and one validation rationale source.  No top-three
  rationale set, prediction, or evaluation is merged or ensembled.
- **Encoder:** the prior validated best backbone,
  `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`, is
  used in three separate LoRA score-regression runs.
- **Targets:** only `content`, `organization`, and `expression`.  There is no
  fourth score target; the validation summary reports those axes and a
  three-axis macro diagnostic only.
- **Split isolation:** 2,000 canonical train writings are used for generation
  and encoder fitting; 400 canonical validation writings are generated and
  evaluated only after training.  Generated text, essays, IDs, model states,
  logs, and predictions remain in ignored restricted/output roots.
- **Resources:** one actual GPU0 generation-plus-one-update preflight; after
  its gate passes, local vLLM uses tensor parallelism over GPUs 0--3 for each
  source and encoder training/evaluation uses four-process DDP over GPUs 0--3.
  GPUs 4--7 are never queried or used.

## Fixed execution and gates

The durable launcher is `scripts/run_rlaif_top3_encoder_v1.py`.  Its immutable
runtime configs, ledgers, server attestations, and logs are under ignored
`outputs/rlaif-top3-encoder-v1/20260725-001/`; private rationale artifacts are
under ignored `data/processed/restricted/rlaif_top3_encoder_v1/`.

1. GPU0 preflight: one score-blind rationale request with strict three-axis
   JSON schema, then one Qwen2.5 encoder update.  It must produce a finite
   train metric and a saved state.
2. For each of the three sources, TP=4 vLLM generates 2,000 train and 400
   validation rationale objects.  Every output must parse, have all three
   axes, and have zero transport/schema failures.
3. For each source, DDP=4 trains Qwen2.5 for the fixed 12-epoch decoder
   rationale schedule.  It must have finite metrics and a checksummed model
   state.
4. Each encoder is evaluated once on 400 matching validation rationales, with
   exactly one prediction per writing.  The report stores only per-axis RMSE /
   Spearman and the three-axis macro diagnostics; no raw prediction is saved.

The runner stops on a declared generation, training, or evaluation hard-gate
failure and preserves the failed ignored artifact.  `full-resume` may reuse
only a completed artifact whose complete provenance is revalidated.

## Result status

Implementation and GPU-free contract checks are complete.  Runtime progress,
gate evidence, and final aggregate metrics will be appended here after the
declared GPU0 preflight and full durable runner stages finish.
