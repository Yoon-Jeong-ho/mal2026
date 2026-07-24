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
   rationale schedule.  It must have finite metrics and a checksummed final
   model state.  Intermediate full-model checkpoints are disabled because no
   declared stage consumes them; this removes redundant NFS writes without
   changing data, optimizer, update count, or final-state validation.
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

## GPU0 preflight result

The declared actual preflight completed on GPU0.  Its one score-blind Midm
`random1` rationale request was schema-valid (1/1) with zero transport/schema
failures.  The one-step Qwen2.5 three-axis encoder update completed with a
finite train loss (0.618423) and a checksummed final state.  This gate did not
evaluate validation performance and does not participate in model selection.

## Full top-three generation, encoder fitting, and validation result

The durable full runner completed at `2026-07-24T22:38:25Z` with no failed
stage.  For every selected adapter, its separate train rationale generation
completed 2,000/2,000 schema-valid records and its separate validation
generation completed 400/400.  All six generation reports passed their three
hard gates: complete records, every rationale parsed with all three axes, and
zero transport/schema failures.  Generation remained score-blind: it neither
read nor prompted the source writing scores.

The three resulting Qwen2.5-7B LoRA encoders each completed the fixed 12-epoch
schedule (384 DDP=4 updates, finite training metric, checksummed final state).
The canonical writing scores are used at this *encoder supervised-target*
stage, but the target list is exactly `content`, `organization`, and
`expression`; neither an `average` field nor a fourth prediction head is read,
trained, or emitted.  Each model was evaluated once on the matching 400
validation rationales (400 unique writings, one prediction per writing, zero
rationale-source combination).

| decoder rationale source | content RMSE / Spearman | organization RMSE / Spearman | expression RMSE / Spearman | three-axis diagnostic RMSE / Spearman |
| --- | ---: | ---: | ---: | ---: |
| rank 1 — Midm-2.0-Base bundle / `random1` | 0.592465 / 0.546808 | 0.774760 / 0.588955 | 0.632834 / 0.334114 | 0.666686 / 0.489959 |
| rank 2 — A.X-4.0-Light bundle / `random1` | **0.590273** / 0.543862 | 0.769708 / 0.580551 | **0.629532** / **0.368061** | **0.663171** / **0.497492** |
| rank 3 — A.X-4.0-Light bundle / `all5` | 0.591685 / 0.541987 | **0.769705** / 0.584176 | 0.634594 / 0.342908 | 0.665328 / 0.489690 |

The predeclared selection rule is lower three-axis diagnostic RMSE, then
higher three-axis diagnostic Spearman only for a tie.  It selects the rank-2
A.X-4.0-Light `random1` rationale-source encoder (`0.663171` diagnostic
RMSE).  The diagnostic is a post-hoc arithmetic summary of the three reported
axes, **not** an `average` writing-score target, prediction, or an ensemble.
This selects a best encoder under the fixed validation protocol; it does not
establish a generalization claim beyond the single canonical validation split
or remove the upstream LLM-judge/proxy-label limitation.
