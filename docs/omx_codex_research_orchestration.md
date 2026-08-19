# MAL2026 OMX/Codex research orchestration

This is the MAL2026 implementation of the durable two-tier authority in
[`codex_execution_policy_v2.md`](codex_execution_policy_v2.md). `AGENTS.md`
remains authoritative for privacy, Git, data, and hardware boundaries. Keep
per-run state only in an append-only ledger or aggregate experiment record.

## Authority and progression

- One lead owns end-to-end progress: task-card validation, resource scheduling,
  recovery, gate records, stage transition, and final reporting. Investigators
  and reviewers return evidence; they neither block the lead nor require the
  lead to wait for another user turn.
- An approved task card fixes the stage command and its next full or pilot
  command. A passing smoke is a gate, not the deliverable: replaying it
  successfully requires the lead to run that predeclared next stage and report
  its completion.
- Integration recovery is autonomous within the task card's recovery envelope.
  It includes dependency, launcher, API-envelope, schema-parsing,
  serialization, test-harness, transport, and resource-setup failures. Apply
  the policy's focused-test, same-smoke replay, and three-iteration limit.
- A scientific gate covers data/split, labels, prompt content, score
  rubric/objective, model/version, training or decoding hyperparameters,
  acceptance metrics, validation exposure, selection, SFT, DPO, and GRPO.
  Record the decision and obtain user authorization unless the named experiment
  plan explicitly preapproves that exact decision.

## Resource, privacy, and session rules

- The default MAL2026 scope is GPUs 0--3, using GPU0 first for preflight and
  smoke. The current user may explicitly expand a named task card to specified
  GPUs among 4--7. Once authorized, the lead may perform the minimum read-only
  availability check needed and use only those named GPUs. Never terminate,
  displace, or alter a pre-existing process, and never infer its ownership.
  Keep every task card within its declared GPU scope and record the exact GPU
  scope and the user's authorization in the run ledger.
- Preserve privacy and aggregate-only reporting. Do not record raw essays,
  restricted inputs, identifiers, credentials, prompts, rationales, or keys.
- Do not publish or stage Git changes. External APIs, new data, destructive
  actions, and resource-boundary or cost conflicts require approval.
- User-owned tmux sessions may be monitored read-only only when requested;
  never attach, inspect pane contents, send keys, kill, rename, or otherwise
  control them. An agent-created background session must have a distinct
  `mal2026-` project-prefixed name and be reported.

## Required records

Before execution, the lead verifies a task card containing the policy-required
deliverable, completion predicate, recovery envelope, test ladder, permitted
inputs, command/output paths, resource scope (GPUs 0--3 by default or
user-authorized named GPUs among 4--7), GPU-scope authorization, and escalation
criteria. Append aggregate evidence and each transition to the run ledger; do
not edit a durable policy document with current status.

Use this minimum transition record:

```json
{
  "run_id": "string",
  "stage": "named approved stage",
  "event": "start|repair|smoke_pass|next_stage_complete|blocked",
  "failure_family": "none|named integration family",
  "repair_iteration": 0,
  "evidence_ref": "aggregate-only path or digest",
  "command_ref": "task-card command identifier",
  "resource_scope": "none|GPU0|GPUs 0-3|user-authorized named GPUs among 4-7",
  "gpu_scope_authorization": "default|exact current-user authorization reference",
  "decision": "continue|escalate|complete"
}
```
