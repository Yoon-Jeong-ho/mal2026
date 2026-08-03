# Draft task card: KURE Stage3 phase-1 direct CORAL OOF

## Authorization state

- Proposed run ID: `kure-phase1-direct-oof-v1-20260803-001`
- Status: `authorized_pending_hash_binding_and_smoke`
- GPU scope if authorized: physical GPU0 for a nonselectable smoke, then
  physical GPUs 0--3 for inference-only exact five-fold OOF. The fixed mapping
  is fold 0→GPU0, fold 1→GPU1, fold 2→GPU2, fold 3→GPU3, and fold 4→GPU0.
- User authorization recorded on 2026-08-03 (Asia/Seoul), verbatim:
  `그래 goal도 다시 쭉 진행해보자. gpu 작업은 종료해놨고, 너가 끝까지 진행하면서 겨로가를 내면 된다.`
- This contextual approval authorizes the named GPU0 smoke and GPU0--3 exact
  five-fold direct OOF protocol in this card. It does not authorize
  validation-based selection, `average`, calibration, refit, deployment, or
  any later LDS scientific run.
- Every long-running repository-owned Python process in this protocol must use
  the installed `setproctitle` package and expose a `mal2026:direct:*` title
  containing its stage and, for inference workers, fold/axis when applicable.
- The stopped `...-004` scheduler and delayed `...-005` successor are checked
  explicitly. The experiment never signals either scheduler; the shared GPU
  coordination locks and fresh compute-process gate prevent overlap.

This card is required because directly decoding an intermediate Stage3 head
was not a preregistered Stage3 candidate. It is an adaptive, same-train
diagnostic prompted by the observed failure of both the original cRT head and
its integration recovery. Even a positive result cannot be described as
independent confirmation or generalization evidence.

## Fixed hypothesis

The Stage3 restricted checkpoints contain the phase-1 `coral-natural` LoRA
weights and its `score.*`, `cut_base`, and `cut_gaps` parameters in addition to
the later cRT `head.*`. cRT training froze the representation and phase-1 head.
Therefore the saved checkpoint can reconstruct the exact phase-1 model without
retraining. The diagnostic tests whether replacing the phase-1 ordinal decode
with cRT caused avoidable central collapse.

## Proposed single-candidate protocol

1. Bind the committed Stage3 config, aggregate, exact five-fold assignments,
   exact R0 OOF, KURE revision, AI-Hub warm-start artifact, and all 15
   `coral-natural` axis/fold checkpoint hashes.
2. Reconstruct each outer-fold/axis KURE+LoRA model from its checkpoint.
   Load only `lora_*`, `score.*`, `cut_base`, and `cut_gaps` as the candidate
   state. Require the saved cRT `head.*` tensors to exist for lineage auditing
   but explicitly forbid them from loading into or influencing inference.
3. Decode four CORAL cumulative logits into a five-class PMF and emit the
   continuous expectation over scores 1--5. No calibration, threshold tuning,
   blending, retraining, refitting, inverse-prior correction, or candidate
   selection is allowed.
4. A separate, pre-run data-steward stage materializes one hash-bound, restricted,
   label-free projection containing only `id`, document/prompt identifiers, prompt,
   essay, and outer-fold membership. This projection is generated before and
   independently of candidate model inference and scientific results. Candidate
   inference workers must not open, stat/hash, or otherwise access gold-bearing
   train, fold-row, or exact-R0 files before their restricted predictions are
   durably committed. They read held text and folds only from the label-free
   projection. Held axis labels may first be opened after prediction commit; the
   source `average` score is never indexed.
5. A GPU0 content-only smoke is nonselectable and cannot be reused as a
   scientific result. Only after its attested hashes pass may all five outer
   folds run once across GPUs 0--3.
6. Concatenate the five outer predictions exactly once. Report three-axis
   macro RMSE, per-axis RMSE, Spearman, `{1,2}` and score-5 RMSE, gold-3/4
   balanced accuracy, QWK, equal-group RMSE, prediction-band counts, and the
   frozen Stage6 common gate with 10,000 source-clustered bootstrap resamples.

## Promotion and stop rule

- Exact R0 remains protected unless all frozen common-gate conditions pass.
- Because this draft is outside the existing Stage6 trust chain, a gate pass
  is not automatic deployment eligibility and requires a new explicit,
  hash-bound scientific decision.
- If the common gate fails, freeze this direct-decode candidate permanently;
  do not tune or calibrate it after seeing OOF results and do not evaluate it
  on the exposed validation split.
- The next proposed scientific candidate after such a failure is one fixed
  outer-train-only LDS-weighted CORAL experiment under a separate authorized
  task card. NPCR and the fixed blend are not repeated.

## Frozen upstream evidence

- Stage3 config: `configs/kure_ordinal_oof.v1.json`
  - SHA-256: `5108e37c490482fbbf45c043f2ad4d0154048d432b4a4a163b3d102f8c6d756e`
- Stage3 aggregate:
  `outputs/kure-ordinal-oof-v1/kure-ordinal-oof-v1-20260803-001/aggregate.json`
  - SHA-256: `eb13d63d28258f331ebcefb2b79f4364ddcc9ff38eec38da533665222706e0e3`
- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- Fold manifest SHA-256:
  `9ccb63c857f80cefb1ed15ac4f60ca75f5570d044c273030d5eb0185c756e938`
- Fold rows SHA-256:
  `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16`
- Exact R0 OOF SHA-256:
  `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04`
- Stage3 cRT recovery negative aggregate SHA-256:
  `aef53a1ec26875ffe9bf88ddbb9fd7c4846402d3c392f0d928fe05364e3c289a`

## Pending before runnable status

1. A committed runnable config binding the final authorized task-card SHA and every checkpoint
   SHA, plus exact commands, seeds, environment, hardware mapping, ACL and
   telemetry evidence fields.
2. The GPU0 smoke and its hash-bound attestation.

## Completed preparation evidence

- Preparation/code commit:
  `10cb2e430eff03c40089e9be9eec1f211496c093`
- Data-steward command:
  `.venv-standard/bin/python scripts/prepare_kure_phase1_direct_input.py --config configs/kure_phase1_direct_oof.v1.json`
- Preparation-request config SHA-256:
  `54fc72c254939adba8f46cbc8b5881b52a36ec8f2eb230b3344043015b63392c`
- Generator SHA-256:
  `02fc15af87bc022a18e4fd54e9ed414708997f76792c61e6bb956d97683ad0c4`
- Restricted label-free projection:
  - SHA-256: `602e6e39f21ce13abbcc3463af6b01b100da096da4b07d90e279cf43012a5170`
  - 2,000 rows; exactly 400 rows per frozen outer fold.
  - Exact top-level schema: `id`, `document_id`, `prompt_num`, `prompt`,
    `essay`, and `outer_fold`; no score, average, gold, or label field.
- Restricted manifest SHA-256:
  `a208b0b538703fa0410ec52b8cf9f7332cdac6489912a358ff42d8a36503d98f`
- Both restricted artifacts and every parent through the restricted anchor have
  zero world permission bits. The project filesystem reports mode `0770` for
  these ordinary files despite the generator requesting `0660`; privacy checks
  therefore enforce the repository boundary (owner/group only) rather than an
  unsupported exact mode.
- The generator ran exactly once. No GPU, model inference, metric computation,
  validation row, or scientific result was used in this preparation stage.

## Independent code-review evidence

- Final disposition: **APPROVE for code preparation only; not approval to run**.
- Focused unit tests: 18/18 passed.
- Python compilation and Bash syntax checks passed.
- The pending launcher dynamically failed before GPU query, lock acquisition,
  or output creation.
- The review found no remaining critical, high, medium, or low code issue after
  fixes for held-label isolation, checkpoint lineage, fail-closed scheduler and
  telemetry handling, live-job-only cleanup, no-clobber publication, ACL checks,
  prediction-band reporting, and failure-ledger preservation.
- LSP/type-check tools were absent from the existing environment and were not
  installed, consistent with the repository environment policy.
