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

## Fixed full Midm bundle result (`20260722-019`)

After a fresh GPU0-first preflight, the unchanged full recipe completed the
Midm bundle arm for both estimators: 1,920 opaque train-only groups, 480
updates, and four policy completions per group.  The full `all5` arm made
38,395 aggregate Qwen calls, with 7,679/7,680 valid policy completions
(99.987%); `random1` made 7,678 calls, with 7,678/7,680 valid policy
completions (99.974%).  Both parse rates exceed the declared 0.98 gate.  The
zero-reward-standard-deviation fractions were 0.0 (`all5`) and 0.011458
(`random1`), both within the declared 0.8 limit.  Reference adapter digests
were identical before and after each arm; source/candidate writing scores were
not read or prompted.

Frozen-v6 evaluation used the same 400 held-out essays and five-form,
ten-replication, 20,000-observation protocol per arm as the pilot.  Both arms
had zero abstentions, 20,000/20,000 valid scores, and zero transport/schema
failures.  Relative to the frozen SFT macro baseline (3.867967), both are now
clear positive results:

| arm | macro mean | content | organization | expression | paired macro delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all5` | 4.149600 | 4.145400 | 4.096950 | 4.206450 | +0.281633 [0.216633, 0.347800] |
| `random1` | 4.100667 | 4.117900 | 4.026400 | 4.157700 | +0.232700 [0.166100, 0.301117] |

Every requested axis improves with a positive 95% paired interval.  For
`all5`, deltas are +0.305650 content [0.234350, 0.378950], +0.200850
organization [0.123000, 0.276750], and +0.338400 expression [0.254900,
0.424250].  For `random1`, they are +0.278150 [0.206100, 0.352700],
+0.130300 [0.045850, 0.213650], and +0.289650 [0.198250, 0.380900],
respectively.  `all5` is higher in this Midm bundle comparison, but the
predeclared remaining-model run retains both arms rather than selecting from
this validation result.

This full result reverses the pilot's macro-inconclusive outcome and is direct
evidence **against reducing the supervised/eligible RL population**.  The
fixed all-model matrix therefore proceeds without changing data, prompt forms,
reward settings, or validation protocol.  Aggregate-only evidence is retained
under ignored roots:

- `outputs/rlaif-grpo-prompt-ensemble-v7/rlaif-grpo-prompt-ensemble-v7-midm2_base-bundle-{all5,random1}-full-019/training_complete.json`
- `data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/rlaif_grpo_v7/rlaif-grpo-prompt-ensemble-v7-midm2_base-bundle-{all5,random1}-validation-001/aggregate_judge_report.json`

## Fixed full A.X-4.0-Light bundle result (`20260722-019`)

The first non-Midm full-matrix comparison also completed without a protocol
change.  Both arms start from the same A.X-4.0-Light bundle SFT adapter and
use 1,920 opaque, score-blind train-only groups, 480 updates, and four policy
completions per group.  All 7,680 policy completions parsed for each arm.  The
`all5` arm made 38,400 Qwen calls (mapped reward mean/sd 0.459991/0.280672);
the deterministic `random1` arm made 7,680 calls (0.431619/0.362615).  Their
zero-reward-standard-deviation fractions were 0.0 and 0.011458,
respectively.  Reference adapter hashes were unchanged, source/candidate
writing scores were not read or prompted, and raw prompts/completions were not
persisted outside ignored roots.

Frozen-v6 again evaluated 400 held-out essays with five independently posed
prompt forms and ten replications (20,000 schema-valid scored observations per
arm).  Both arms had zero abstentions and zero transport/schema failures.  The
paired comparison to the frozen A.X SFT baseline is:

| arm | macro mean | content | organization | expression | paired macro delta (95% bootstrap CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all5` | 4.186950 | 4.157050 | 4.141350 | 4.262450 | +0.420767 [0.348783, 0.496533] |
| `random1` | 4.192283 | 4.105850 | 4.197500 | 4.273500 | +0.426100 [0.355700, 0.499650] |

All requested-axis intervals are positive.  `all5` has content +0.471550
[0.386950, 0.558650], organization +0.271550 [0.188250, 0.354800], and
expression +0.519200 [0.412850, 0.630500].  `random1` has +0.420350
[0.336950, 0.506950], +0.327700 [0.245300, 0.408750], and +0.530250
[0.427400, 0.638700], respectively.  This is a second independently trained
model in which both reward estimators improve all three axes; it is not used to
select an arm while the predeclared matrix remains in progress.

Aggregate-only evidence is retained under ignored roots:

- `outputs/rlaif-grpo-prompt-ensemble-v7/rlaif-grpo-prompt-ensemble-v7-ax4_light-bundle-{all5,random1}-full-019/training_complete.json`
- `data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/rlaif_grpo_v7/rlaif-grpo-prompt-ensemble-v7-ax4_light-bundle-{all5,random1}-validation-001/aggregate_judge_report.json`

## Preserved A.X content runtime interruption and fresh resume (`20260722-019` → `-020`)

The next declared arm, A.X-4.0-Light `content`/`all5`, was interrupted at
global step 444/480.  It had completed 7,104 score-valid policy completions
and 35,520 logical Qwen observations before a returned Qwen OpenAI envelope
had a non-`stop` finish reason (`envelope_finish`).  The reward callable
correctly refused to turn that response into a reward, the runner preserved
`training_failed_runtime.json`, stopped all servers, and did not export an
adapter or open/evaluate validation text for that partial arm.  This is an
execution-envelope failure, not a metric or data-quality observation.

The repaired caller now applies the already-declared bounded
`max_transport_attempts: 3` specifically to this scoreless response-envelope
case: it reissues the identical private request, keeps parsed-invalid scores
and abstentions non-retriable, and records aggregate-only extra retry counts.
It does not alter prompts, data, seeds, response schemas, reward mapping,
optimizer settings, or the frozen-v6 evaluator.  vLLM documents that an
`error` finish reason is retryable while a `length` finish is incomplete;
the caller retains the strict requirement for a usable `stop`/schema-valid
score and preserves a failure if the bounded reissues are exhausted:
<https://docs.vllm.ai/en/v0.17.0/api/vllm/entrypoints/openai/responses/serving/>.

The fresh `20260722-020` runner lineage verifies and reuses the already
complete A.X bundle two-arm frozen evaluations, then starts fresh ignored arm
directories for the incomplete content task and all later matrix tasks.  It
does not overwrite the partial `-019` artifact.  Static compilation and the
15-test contract suite passed; the suite includes simulated first-attempt
envelope failure/success and all-attempt failure bounds before the resumed
long run.
