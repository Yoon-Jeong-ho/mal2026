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

## Fresh v7 two-arm Midm pilot (`20260722-018`)

The fresh lineage completed all declared gates on GPUs 0--3.  It used the
same source SFT adapter for both arms, 320 opaque train-only groups from the
declared 1,920 eligible rows, four policy completions per group, and no source
writing scores or validation source text.  The TP2 policy rollout batch gate
completed 64/64 canonical parses in 36.376 seconds (limit: 240 seconds).
Both one-update actual-input gates, the 80-update `all5` pilot, and the
80-update deterministic `random1` pilot passed.

Training-signal aggregates (not post-RL quality metrics) were:

| arm | groups / updates | policy completions | Qwen calls | parse-valid | mapped reward mean / sd | zero group-sd fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all5` | 320 / 80 | 1,280 | 6,400 | 1,280/1,280 | 0.384271 / 0.313104 | 0.0 |
| `random1` | 320 / 80 | 1,280 | 1,280 | 1,280/1,280 | 0.376432 / 0.391739 | 0.0 |

For `random1`, deterministic prompt-form assignments were 259
`axis_fidelity`, 274 `balanced_rationale`, 259 `communication_quality`, 243
`diagnosis_calibration`, and 245 `grounding_specificity`.  The frozen source
adapter SHA-256 was unchanged before and after both runs.  The ignored
completion records retain the detailed provenance; no raw writing, prompt, or
rationale was added to this record.

### Frozen-v6 validation result

Each arm generated one deterministic rationale for each of 400 held-out
essays.  Frozen v6 then made 20,000 calls per arm (five independently posed
judge forms × ten replications), with 20,000 schema-valid scored observations,
zero abstentions, and zero transport/schema failures per arm.  The evaluation
never supplied candidate/source writing scores to the policy or judge.

The frozen SFT baseline macro mean is **3.867967**.  The paired outcomes are:

| arm | macro mean | content | organization | expression | paired macro delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all5` | 3.864400 | 3.857600 | 3.853700 | 3.881900 | -0.003567 [-0.075983, 0.067033] |
| `random1` | 3.909850 | 3.935150 | 3.858100 | 3.936300 | +0.041883 [-0.029300, 0.112800] |

`all5` is a null/slightly negative macro result.  `random1` is directionally
positive but its primary macro interval includes zero; it is therefore **not a
verified macro improvement**.  Its content-axis paired delta is +0.095400
(95% CI [0.016200, 0.177800]), while organization is -0.038000 (95% CI
[-0.115900, 0.039700]).  This preserves both the promising content signal and
the unresolved macro result without choosing a reward estimator from
validation.

The five frozen prompt forms retain material calibration variation: for
`all5`, macro means ranged from 3.618417 (`grounding_specificity`) to 4.033750
(`axis_fidelity`), with axis ranges 0.472000 (content), 0.295750
(organization), and 0.479750 (expression).  The analogous `random1` ranges
were 0.451000, 0.299000, and 0.471500.  Thus a lower score here is not evidence
that the supervised 2,000-example corpus should be reduced; it is evidence to
keep the fixed full comparison and later assess reward disagreement/low-margin
filtering as a separately declared ablation.

Aggregate evidence is retained only under ignored roots:

- `outputs/rlaif-grpo-prompt-ensemble-v7/20260722-018/aggregate/midm2_bundle_v7_full_batch_gate.json`
- `outputs/rlaif-grpo-prompt-ensemble-v7/rlaif-grpo-prompt-ensemble-v7-midm2_base-bundle-{all5,random1}-pilot-018/training_complete.json`
- `data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/rlaif_grpo_v7/rlaif-grpo-prompt-ensemble-v7-midm2_base-bundle-{all5,random1}-pilot-validation-001/aggregate_judge_report.json`
