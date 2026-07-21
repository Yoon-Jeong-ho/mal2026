# RLAIF/GRPO prompt-ensemble v7 allocator repair — 2026-07-22

- **Status:** the first v7 launcher (`20260722-017`) reached only the policy
  server synthetic health request and was stopped before Qwen health, policy
  rollout, reward, update, validation generation, or score comparison.  A
  fresh v7 lineage is prepared after the runner repair below.
- **Fixed science:** the score-blind train-only 320-group pilot population,
  opaque selection/holdout, four completions per essay, five Qwen prompt
  forms/seeds, `all5` and deterministic `random1` reward estimators, JSON
  object/canonical 192-character policy contract, optimizer, frozen-v6
  evaluator, and GPU 0--3 boundary are unchanged from v6.

## Preserved v6 runtime result (`20260722-016`)

The TP2 policy batch gate passed 64/64 canonical parses in 35.940 seconds.
Both one-update actual-input arm gates then completed with four valid policy
completions, zero judge schema/transport failures, and nonzero reward
variation.  The 320-group `all5` pilot subsequently reached global step 8,
with 192/192 valid policy completions and 960 aggregate Qwen calls, before
the next float32 backward pass raised `OutOfMemoryError` while requesting
1.26 GiB on the policy-training process.  At failure it reported 71.67 GiB
allocated and 5.68 GiB reserved-but-unallocated by PyTorch.  The ignored
aggregate record is
`outputs/rlaif-grpo-prompt-ensemble-v6/rlaif-grpo-prompt-ensemble-v6-midm2_base-bundle-all5-pilot-016/training_failed_runtime.json`.

No final adapter was exported and no frozen validation text was opened or
scored.  This is a runtime memory failure, not evidence about the judge,
reward quality, training-data amount, or post-RL score.  The runner stopped
the v6 Qwen and TP2 policy services cleanly; the failed log and aggregate
record remain preserved in ignored output roots.

## v7 change and exact retry boundary

v7 changes only the policy-trainer process allocator setting to
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.  It is declared in the
new versioned config and recorded in the stage ledger.  The setting permits
the CUDA caching allocator to grow segments when rollout batches have varying
sequence lengths; it does not change rows, group cardinality, generated text,
judge prompts, rewards, optimizer parameters, checkpoints, or validation.

The interrupted `-017` startup also exposed a runner-only inconsistency: it
unconditionally added `--enforce-eager` to Qwen reward/evaluation commands,
despite the already hash-bound frozen-v6 config declaring
`enforce_eager: false` and documenting successful CUDA-graph scoring.  The
runner and both attestation validators now read that fixed setting.  Thus the
fresh retry uses CUDA graphs for Qwen exactly as the frozen v6 configuration
declares; this is a correction to resource execution, not a prompt, model,
reward, data, or validation-protocol change.
The official vLLM engine-argument documentation states that
`--enforce-eager` disables CUDA graphs, whereas false uses the hybrid
CUDA-graph/eager path for performance and flexibility:
<https://docs.vllm.ai/en/v0.10.0/configuration/engine_args.html>.

The retry uses a fresh ignored lineage (`20260722-018`) and repeats the
existing TP2 64-completion gate and two one-update arm gates before the
all-five then random-one 320-group pilot.  A second allocator-independent
memory failure will be preserved; no data reduction or reward/prompt retuning
will be selected from validation.

The reproducibility source revision for the `-018` command is
`0b7de49` (`Add TP2 RLAIF GRPO prompt ensemble pipeline`); it contains the
versioned v7 config, allocator setting, no-eager Qwen runner path, and all
contract tests.  The runner command is
`MAL2026_RLAIF_CONFIG=configs/rlaif_grpo_prompt_ensemble.v7.json MAL2026_RLAIF_RUNTIME_ID=20260722-018 PYTHONPATH=src .venv-standard/bin/python scripts/run_rlaif_grpo_prompt_ensemble_v1.py midm-pilot`.

## Static verification before launch

- `python -m json.tool configs/rlaif_grpo_prompt_ensemble.v7.json`
- `python -m py_compile src/mal2026/rlaif_grpo.py src/mal2026/rlaif_evaluation.py scripts/run_rlaif_grpo_prompt_ensemble_v1.py`
- `PYTHONPATH=src python -m unittest tests/test_rlaif_grpo_contract.py -v`
- `git diff --check`

The GPU-free contract suite validates the score-blind five-form setup,
rationale-only schema, v6 TP2 batch cardinality, v7's sole allocator
difference, and the future full-run TP2-policy/GPU3-judge topology.
