# OpenAI explanation repeat-distribution v5 remediation — 2026-07-20-001

**Scope:** aggregate-only runtime remediation on physical GPUs 4--7. No GPU
0--3 was queried or used. No validation input, selection artifact, training,
or full-scale run was created before its declared gates.

## Observed v4 diagnosis

- The closed v4 pilot recorded 2,860 transport-or-schema failures in 2,895
  calls, zero eligible distributions, and zero watchdog faults.
- Aggregate-only server-log classification found 1,963 context rejections:
  GPU 4: 496, GPU 5: 489, GPU 6: 488, GPU 7: 490. The pinned llama.cpp
  allocation divided `--ctx-size 4096` across `--parallel 4`, leaving 1,024
  tokens per slot; rejected requests were 1,024--1,377 tokens.
- The remaining 897 failures are irrecoverably unclassified in v4 because its
  client reduced HTTP, response-envelope, JSON, and schema failures to one
  boolean. This record does not inspect raw v4 payloads or responses.
- V4 also submitted from a global 16-worker pool without endpoint admission
  control, used no retry policy, and did not bind request token budget to the
  actual per-slot context.

## v5 repair provenance

The first v5 lineage used one server slot per GPU (`--parallel 1`) and one
4,096-token slot, a 192-token JSON-only completion budget, a 256-token safety
margin, fixed-byte transport-only retries (at most two attempts), server
`/props` and CUDA-visibility attestation, independent per-GPU queues, and
aggregate response categories. Both `--reasoning off` and
`enable_thinking=false` remain required.

- `configs/openai_explanation_repeat_distribution.v5.pilot.json`:
  `ff43cef169a82b407c10f52645d74035c0f1db93c19a38d39d8ea7b694ddac68`
- `scripts/preflight_openai_repeat_v5_synthetic.py`:
  `436c0a405aa0eba893125c0654135ffc3f2be300fab80e95962a0ef2c908c5a4`
- `scripts/run_openai_explanation_repeat_distribution_v5.py`:
  `76846b611d343c7762da2d8e88ce78c35b22fd993b73ca97da990b17b807d980`

The exact-v5 fixed synthetic preflight at
`outputs/debug/openai-repeat-v5-synthetic-20260720-002/aggregate.json` passed:
80/80 schema-valid calls, zero final transport/schema categories, all required
rubric fields parsed, deterministic and cross-GPU agreement, invalid/identity/
padded controls, retry invariant, server liveness, and clean shutdown. Its
fixed boundary prompt token count was 3,279 on every GPU, within
`3279 + 192 + 256 <= 4096`.

## Failed real smoke and v5.1 amendment

The v5 three-essay train-only smoke (`openai-repeat-v5-20260720-003`) failed
closed. It made 105 calls across all candidate/repeat/control paths; its
aggregate context budget passed (prompt-token range 302--1,216), but 91 final
failures were classified as `semantic_abstain_without_failed_gate`. There were
no HTTP failure categories and no retry was taken. The smoke therefore did not
create a selection artifact and its chained 96-essay pilot did not start.

This showed an ambiguous abstention instruction in the real request template.
It was not treated as permission to relax a gate. A new immutable v5.1 lineage
adds the explicit contract: `scored` if and only if all hard gates are true;
`abstain` if and only if at least one hard gate is false; all score fields are
always present and ignored for abstention.

- `configs/openai_explanation_repeat_distribution.v5_1.pilot.json`:
  `c1f58ae66b9aee8e7b64570ff3ee1a34dbc0eda11c157b480d7d91e952b959b0`
- `scripts/run_openai_explanation_repeat_distribution_v5_1.py`:
  `e8c51edde480f2025273f7b112af5a8f94c98db6ad65b5aadddbc3eaf3d4d329`
- `scripts/run_openai_explanation_repeat_distribution_v5_1.sh`:
  `eb51f8bf0ea40c498de1f75a31e928d20d21240f973c004e2bfdb43dfa24d5cc`

The v5.1 fixed synthetic rerun is the current gate. It must pass with the same
zero-failure and control criteria before a new three-essay smoke is permitted.
Only a zero-failure v5.1 smoke may automatically launch the capped 96-essay
pilot. This record will be updated with those aggregate outcomes.

## v5.1 result and v5.2 current gate

V5.1 synthetic passed, but its three-essay smoke failed closed with 105/105
`schema_value` classifications. Aggregate inspection identified a wrapper bug:
the emitted v5.1 schema version was not propagated to the response parser. No
pilot was launched.

V5.2 is a new immutable lineage that binds the parser to the emitted schema;
it retains every v5.1 runtime, context, retry, and control gate.

## v5.2 GPU0--3 migration and stopped smoke

**Resource boundary.** Physical GPUs 4--7 became externally owned and were
strictly off-limits. This migration did not query, modify, or use them. GPU0
was temporarily owned for the synthetic and real smoke gates; GPUs 1--3 were
not queried, yielded, or modified because the required GPU0 smoke gate failed
before a full launch could be authorized. Their utilization-only jobs therefore
remain untouched.

The migrated, train-only v5.2 configuration is
`configs/openai_explanation_repeat_distribution.v5_2.gpu0_3.json`
(`daff36e1ffee5f7ee022f543cd988c65614fb4a20de8601741715e04ec904a97`).
It preserves candidate isolation, validation-request count zero, selection
artifact prohibition, five deterministic plus five dispersion repeats per
candidate, the 192-token completion cap, 256-token margin, and two-attempt
transport-only retry. It uses project-owned GPUs 0--3 only, isolated
localhost ports 18200--18203, and `--parallel 4` with total context 16,384 so
every slot retains the repaired 4,096-token budget. The runner and synthetic
preflight hashes are respectively
`f5fe6ba16bfbfea8c14145cc9baf05cff4830176bd8b22316c0e6d58b961b72e` and
`799765fa3c47a5ae9fc9ebcae2f307102453181d965cbde83990a0d9b8858527`.

The GPU0-only synthetic preflight passed at
`outputs/debug/openai-repeat-v5_2-gpu0-synthetic-20260720-001/aggregate.json`
(aggregate SHA-256
`0b9bb5a41deb0fb64a758170c3ec7d065d91d4ae98da70a80d78f6a6f856df73`):
20/20 schema-valid synthetic calls, zero final transport/schema categories,
all controls, retry invariant, server liveness, and a fixed boundary of 3,290
tokens satisfying `3290 + 192 + 256 <= 4096`. It used the pinned GGUF
`b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`,
llama.cpp `571d0d540df04f25298d0e159e520d9fc62ed121` / `b10068`, the project
`.venv-standard` interpreter, and a physical NVIDIA H100 80GB GPU0. No real
or validation data was accessed in that gate.

The following real, restricted train-only smoke ran after the synthetic pass:

```text
scripts/run_openai_explanation_repeat_distribution_v5_2_gpu0_3.sh \
  openai-repeat-v5_2-gpu0-3-20260720-002 smoke 3 0
```

It failed closed. Its aggregate report SHA-256 is
`b3398eca04a5ab0112a9fd44e739cab3fbc5e65a5953617b242be0ab0b9569a3` and
its server-attestation SHA-256 is
`fc74580045ed37f5000bf362bb6205e7f9a2e971529f2573aba184aeaca1124b`.
The 105 calls comprised 45 deterministic, 45 dispersion, and 15 controls;
validation rows loaded and validation requests were both zero. The prompt
range (350--1,264) remained inside the 4,096-token slot budget and watchdog
faults were zero. However, 32 final responses were classified as
`semantic_abstain_without_failed_gate`; evidence-valid rates were 0.80,
0.666667, and 0.666667 for candidates 1--3. Deterministic agreement, duplicate
identity, padded-verbosity, evidence-validity, and the zero
transport/schema-failure hard gates consequently failed. No selection artifact,
SFT, DPO, GRPO, validation request, GPU1--3 priority/yield action, or
2,000-essay full run was created.

The smoke's single-GPU cross-GPU check is explicitly recorded as not evaluated;
this does not promote or waive the multi-GPU stability requirement. The full
2,000-essay x 3-candidate run remains blocked by the failed zero-failure smoke
gate. Git SHA at migration execution was
`86902f1e3a077b1178d1297a1dcccf10e929453d`.

After this terminal failure, no scoring was resumed. A GPU-free static
hardening pass added per-endpoint four-slot admission, explicit prior-report
checks for future smoke/full launches, and a priority-plan resolver that
records the GPU0 fallback when the documented GPU1--3 score-utilization jobs
are non-yieldable; it never sends them a signal or overwrites their artifacts.
The current runner, launcher, resolver, and inherited synthetic-wire hashes
are respectively
`31459b013c93482735472fc5822a06996cedbe25cf0f9a40d321ea446dfe36a5`,
`ac5bdf37ecb1e1166df0fc1b3b84bf81be832124c80acc9fc8dd04c6656881aa`,
`0e0f585cbd35f370be6648535127f5ccaa7fb3b615056d89ddd55f8af17107b3`, and
`25cc2149df96234dfa856a6186cd777be342b71306202b95802135c4a819710b`.
