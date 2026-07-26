# Official decoder integer-score comparison contract

This comparison is intentionally separate from the Qwen3-Embedding matrix.
It uses the pinned local `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28` and defines twelve target-data
arms:

- generative decoder, bounded-regression score head, or cumulative-ordinal
  score head;
- public initialization or a matched same-architecture AI-Hub integer-score
  pretraining state; and
- essay-only or essay-plus-selected-rationale input.

The generative target is exactly
`{"content":I,"organization":I,"expression":I}`, where every `I` is an
integer from 1 through 5. It has 125 legal outputs. It contains neither a
rationale nor `average`. The dedicated-head variants attach a fresh three-logit
bounded head or twelve-logit cumulative ordinal head to the causal decoder
backbone. They do not reuse a language-model output head as a score head.

All source scores are projected by the project-wide half-up rule. Only
`content`, `organization`, and `expression` may be indexed. Historical
continuous or four-axis runs are descriptive references and cannot initialize
a primary arm.

Model selection uses the deterministic prompt/document-group-safe 1,600/400
split derived from the 2,000 training essays. After selecting among epochs
1--4 by macro integer RMSE, integer Spearman, continuous RMSE, then earlier
epoch, the model is freshly refit on all 2,000 training essays. Canonical
validation is loaded once afterward for descriptive final evaluation only.

AI-Hub initialization is scientifically meaningful only when produced by the
same decoder architecture, model revision, and three-axis integer target
contract. It is genuine full-parameter training: GPU0 performs one optimizer
update for both selection and refit smoke gates, then GPUs 0--3 run FSDP
full-shard selection and a fresh refit stopped at the exact selected optimizer
step. Selection reads only AI-Hub `selection_dev`; canonical MAL validation is
unreachable. The export is a complete full backbone plus the matched dedicated
head where applicable. Completion, metadata, every inventory member, and the
canonical inventory digest are SHA-256 bound.

For MAL adaptation, the full AI-Hub backbone is loaded before a fresh LoRA is
attached. The matched bounded/ordinal head is retained and trained with that
LoRA; the generative arm retains its fully tuned LM head. Thus `public` versus
`aihub_matched` genuinely compares public initialization against full AI-Hub
pretraining followed by MAL LoRA, rather than comparing two LoRA warm starts.
AI-Hub pretraining never consumes target rationales. Rationale arms remain hard-gated
until the selected official rationale model has produced hash-bound train and
validation files under restricted ignored storage.

The AI-Hub producer is configured by
`configs/official_decoder_aihub_integer_score_pretrain.v1.json`; its trainer
and orchestrator are `scripts/train_official_decoder_aihub_score_pretrain.py`
and `scripts/orchestrate_official_decoder_aihub_score_pretrain.py`.

The target runner supports a GPU0 one-update smoke and full execution. Generative
evaluation uses free-running `generate` constrained by a token trie over the
125 legal outputs; teacher-forced token accuracy is never reported as score
RMSE. Each candidate epoch starts from an identical seed/state, the selected
epoch is freshly refit, and completion persistence is aggregate-only. Full
orchestration must run the three AI-Hub full pretrains first, bind their
completion, inventory, and metadata checksums into the resolved config, then
run the twelve target arms under the
repository GPU0-first / GPU0--3 policy.
