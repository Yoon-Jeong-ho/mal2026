# Codex execution policy v2

This durable policy prevents approved MAL2026 work from stopping at a passing
smoke. It complements `AGENTS.md`; its privacy, Git, data, GPU, and approval
boundaries remain controlling. Store per-run state only in an append-only
ledger or aggregate experiment record, never as current-state prose here.

## Two-tier authority

### 1. Integration recovery: lead proceeds autonomously

Within an already approved stage and its task card, the lead owns completion.
Integration failures include dependencies, launchers, API envelopes, schema
parsing, serialization, test harnesses, transport, and resource setup. The
lead must inspect the failure, implement a bounded repair, run a focused unit
or no-GPU test, replay the same smoke, and, after it passes, automatically run
the predeclared next full or pilot stage with its fixed command and report
completion. A smoke is a gate, not the final deliverable.

Allow at most three evidence-distinct repair iterations per failure family.
Escalate only when the same failure signature recurs twice after a verified
patch, three repairs produce no new diagnostic signal, a boundary or cost
issue appears, or the repair would change a scientific variable. Preserve the
failure evidence and report the escalation reason.

### 2. Scientific decision gates: authorization is required

The following are scientific variables: data or split changes, labels, prompt
content, score rubric/objective, model/version, training or decoding
hyperparameters, acceptance metrics, validation exposure, selection, SFT,
DPO, and GRPO. Before changing one, record the decision and obtain user
authorization, unless a named experiment plan explicitly preapproves that
exact variable and scope. Do not use integration recovery to bypass a
scientific gate.

Destructive actions, external APIs, new data, resource-boundary conflicts, and
unapproved external cost always require approval. Preserve privacy rules, do
not publish Git changes, and keep every task card within a declared GPU scope.
The default MAL2026 scope is GPUs 0--3, with GPU0 first for preflight and smoke.
The current user may explicitly expand a named task card to specified GPUs
among 4--7. Once authorized, the lead may perform the minimum read-only
availability check needed and use only those named GPUs. Never terminate,
displace, or alter a pre-existing process, and never infer its ownership.
Record the exact GPU scope and the user's authorization in the run ledger.

## Roles and tmux

One lead owns end-to-end progress and does not wait for another user turn after
an approved stage. Investigators and reviewers return scoped evidence and may
recommend action, but cannot block the lead's allowed recovery or transition.

For user-owned tmux, agents may monitor read-only only when requested. They
must never attach, inspect pane contents, send keys, kill, rename, or control
those sessions. An agent-created background session must use a distinct
`mal2026-` project-prefixed name and be reported.

## Concise templates

### Task card

```text
Run ID / named approved plan:
Stage and deliverable:
Exact completion predicate:
Permitted inputs and privacy boundary:
Fixed smoke command; fixed next full or pilot command:
Command and output paths:
Resource scope (GPUs 0-3 default with GPU0 first for smoke, or user-authorized named GPUs 4-7):
GPU-scope authorization (default or exact current-user authorization):
Allowed integration-recovery envelope:
Test ladder (focused unit or no-GPU test -> same smoke -> next stage):
Scientific variables explicitly preapproved:
Escalation criteria and ledger path:
```

### Append-only ledger entry

```text
Timestamp | run ID | stage | event | failure family/signature | repair # |
evidence reference | command/output path | resource scope and authorization |
decision | deviation
```

### Lead completion report

```text
Deliverable: [path or aggregate result]
Completion predicate: [pass/fail evidence]
Recovery: [none or family, repairs, focused-test and smoke results]
Next stage: [fixed command] -> [completed result]
Ledger: [append-only record path]
Escalation: [none or exact approval needed]
```
