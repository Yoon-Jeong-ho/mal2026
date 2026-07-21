# RLAIF/GRPO prompt-ensemble v6 tensor-parallel policy gate — 2026-07-22

- **Status:** no v6 policy update, Qwen reward call, adapter, validation
  generation, or score comparison has started.
- **Fixed science:** v5 JSON-object completion format and 192-character
  canonical parser; the opaque train-only 320-group pilot subset; all reward
  prompts/seeds/mapping; optimizer; and frozen-v6 validation remain unchanged.

## Resource-only repair

v5's JSON-object rollout parsed 64/64 but required 301.769 s on one policy
GPU. v6 applies the requested tensor-parallel layout to the **policy vLLM**:
GPUs 0--1 form tensor-parallel size 2, GPU 2 trains the policy on one rank,
and GPU 3 hosts the Qwen judge. The global GRPO generation batch remains 64
(`16 policy examples x 4 samples`); the single trainer rank uses batch size 16
instead of two DDP ranks of 8. This changes compute placement only, not the
population, number of policy completions, reward calls per completion, samples,
or optimizer updates.

The v6 pre-training gate is unchanged: 16 train-only prompts x four samples,
64/64 canonical parses, and no more than 240 s wall time. It tests the exact
TP2 dynamic-LoRA policy server before any trainer launch. A failure is retained
as an operational negative and does not produce a score claim.

## Passed TP2 batch gate and preserved launcher repair (`20260722-015`)

The exact TP2 JSON-object gate passed: 64/64 canonical parses, 38.666 s total
wall time, and request p50/p95/max 30.478/36.144/38.664 s. This is an 87%
wall-time reduction against the prior one-GPU JSON-object gate (301.769 s).
No Qwen reward or policy update was made during that gate.

The immediately following one-update gate exited before model loading because
the old validation rule expected its judge attestation on GPU 1, whereas v6
correctly hosts Qwen on GPU 3 to coexist with the two-GPU policy server. No
completion, reward, optimizer update, or adapter was written. The validator
now binds v6 one-update gates to its declared GPU-3 judge topology and a fresh
runtime lineage is required; the TP2 resource, data, and scoring protocol is
unchanged.
