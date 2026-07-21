# API-rationale SFT and score-regression matrix — 2026-07-21

- **Final status (2026-07-21):** completed. The durable runner lineage is
  `outputs/api-rationale-sft-score-regression-v1/20260721-001/full-resume-015`;
  the aggregate-only final summary is
  `outputs/aggregate-reports/api-rationale-sft-score-regression-v1-20260721-001.final-summary.json`.
  All dependent stages passed their declared completion gates. The frozen
  validation split was used for user-authorized decoder selection, so the
  downstream comparisons in this record are descriptive validation results,
  not a new untouched-test claim.

- **Status:** public snapshots, native-template checks, actual-input data-contract checks, GPU-free code tests, the one-update GPU0 representative SFT preflight, the post-failure four-GPU numerical-recovery gate, and all 12 full decoder SFTs passed. The Phi bundled validation generations `-001` (384 tokens) and `-002` (uniform 512 tokens) each preserved a 399/400 parse-valid result with the same deterministic one-row `finish=length` hard-gate failure; neither is used downstream. The uniform bundled `-003` recovery (512 tokens plus schema-level 192-character limit per axis) passed 400/400 for all three bases. Phi content-only `-001` then preserved 399/400 with the same category, so all axis-only systems were regenerated uniformly in `-002` with the same schema bound; all nine passed 400/400 and each base's three axis outputs merged at 400/400. No completed artifact is overwritten. Qwen judge DP=4 `-001` failed during CUDA-graph warmup, and its fresh eager `-001` judge-output attempt still selected the default FlashInfer FP8 block-scale GEMM that required unavailable `ninja`; neither produced a usable judge distribution. The repaired eager `-002` bundled A.X judge then passed all 20,000 calls, but the next axis-triplet invocation exposed a local loader task-provenance bug before its output stage began. The fresh continuation reuses only that verified A.X bundle report and corrects the axis-triplet loader binding; model/topology/batching/prompts/seeds/scoring remain fixed, and the documented FlashInfer FP8 linear path remains disabled with a synthetic one-token server gate before restricted rationales are opened.
- **Score-regression recovery:** the first Qwen direct `-001` train persisted an invalid model after its second logged interval showed a non-finite gradient norm; the existing finite-summary check was insufficient because Trainer's default NaN filter hid later losses. Its evaluation also exposed a separate config-validator defect: evaluation revalidated a completed training config as if its output directory had to be fresh. Both artifacts are preserved and excluded. The `-002` run exposed the nonfinite values rather than masking them, but standard nonfused AdamW and gradient clipping alone did not prevent the bfloat16 Qwen forward/backward failure. All six fresh `-003` regressors therefore preserve the data, models, objectives, epoch schedules, and Qwen global batch (64), while using float32 base/LoRA/head compute; Qwen reduces each rank's batch from 4 to 2 and raises accumulation from 4 to 8 solely to retain that global batch within memory. Evaluation accepts the completed training root but still requires a fresh evaluation output.
- **KURE DDP recovery:** all three completed Qwen `-003` regressors are reusable. KURE's direct `-003` runner failed after one update because two reviewed model parameters were unused by that backbone's forward route while DDP unused-parameter detection was disabled. KURE alone therefore receives fresh `-004` output lineages with standard Trainer `ddp_find_unused_parameters=true`; no data, model, label, optimizer, schedule, or metric rule changes.
- **Authorization:** direct user authorization covers the complete decoder and encoder training/evaluation matrix, use of GPUs 0–3, local model acquisition for the named public models, validation evaluation, aggregate result documentation, and a scoped Git commit/push after verification.
- **Purpose:** train Korean writing-rationale generators on all validated API candidates; compare their frozen-validation rationale quality to the API baseline with the validated score-only LLM judge; then train score regressors with no explanation, API rationale, or the selected decoder's rationale input.

## Frozen input and split contract

- API candidate source: `openai-rationale-terra-full-20260719-001`, validated train artifact checksum `d69dd9bc349c117a55b4bf652507135f084c9045d95d5939da3980223893aecf` (6,000 candidates / 2,000 essays) and frozen validation artifact checksum `21c7b97e6faf6d8092b4a27e35b60083f9b9b60861493867061816fcb12f9d83` (1,200 / 400).
- Writing source: train checksum `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737` (2,000) and frozen validation checksum `0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4` (400).
- Decoder SFT reads every train candidate's three diagnosis texts, but never reads or prompts the candidate score. It produces rationale-only JSON without candidate or writing scores.
- Score-regression labels come only from the canonical writing-source `content`, `organization`, and `expression` scores. The average and every API-candidate score are excluded from model input and labels.
- Training uses all 2,000 train essays (and all 6,000 API candidate rows where the API-rationale condition applies). The 400 validation essays are not opened for training, checkpoint selection, early stopping, or hyperparameter changes. The user explicitly requested final validation comparison for decoder selection; that selection use is recorded and downstream validation results are therefore descriptive rather than a new untouched test claim.

## Declared model and task matrix

1. **Rationale generators:** `skt/A.X-4.0-Light` at `ba21c20ea1b31ded1ec3e2fb432335077dc4be98`, `microsoft/Phi-4-mini-instruct` at `cfbefacb99257ffa30c83adab238a50856ac3083`, and `K-intelligence/Midm-2.0-Base-Instruct` at `35479c5fc9a18a5db7cc6dbadcf1db68db7beab0`.
2. **Four decoder tasks per base:** one bundled model emits all three axis rationales; three axis-specific models emit content, organization, or expression rationale only. Thus there are 12 SFT runs. Every base's four tasks are evaluated on all 400 frozen-validation essays.
3. **Decoder validation:** maintained vLLM batch generation plus the already validated v6 score-only rationale judge, with five fixed prompt forms and ten fixed seeds per generated rationale. The API candidates' completed frozen-validation v6 aggregate is the comparison baseline. The three bundled outputs are directly comparable across all axes; the predeclared downstream decoder winner is therefore the bundled model with the maximum macro mean judge score (ties: lower prompt-form range, then lexical base ID). The nine single-axis generators are evaluated and reported on their requested axis, but cannot replace the one all-axis decoder required by the later score-regression condition.
4. **Score regressors:** three input conditions—essay only, essay plus API rationale, and essay plus the selected decoder's rationale—are trained with each encoder: the prior local `Qwen/Qwen2.5-7B-Instruct` snapshot at `a09a35458c702b33eeacc393d103063234e8bc28`, and `nlpai-lab/KURE-v1` at `d14c8a9423946e268a0c9952fecf3a7aabd73bd9`. This is six regression runs.
5. **Score validation:** every score-regression run predicts the three analytic scores for all 400 frozen-validation essays. Report per-axis and macro RMSE and tie-aware Spearman correlation; no row-level prediction is persisted.

## Runtime, templates, and gates

- All full jobs use exactly physical GPUs 0–3 together under the maintained TRL `SFTTrainer` or Transformers `Trainer`; only GPU0 is used for the smallest actual-input preflight. vLLM is used for all decoder generations and LLM-as-judge calls. The frozen decoder bases and LoRA adapters use float32 compute with TF32 enabled for the decoder SFT matrix: this recorded numerical-recovery change replaces the failed bf16 setup below and is not a data/objective/hyperparameter selection change.
- The local public-model snapshots are written beneath already-ignored `outputs/model-cache/`. Before training, each tokenizer's native `apply_chat_template` must render the project synthetic contract and a prompt/completion sample; the model-specific rendered template hash, special-token boundary check, and `trust_remote_code` decision are recorded only in aggregate provenance.
- Decoder hard gates: canonical counts/checksums, zero invalid rows, valid chat template, finite Trainer metrics, completed adapter, 400/400 parseable frozen-validation outputs, and zero judge transport/schema failures. The 50-score judge distribution is not used to filter training candidates.
- The bundled decoder generation budget is 512 new tokens plus a schema-level 192-character maximum per axis in the explicitly versioned `-003` lineage. This is a uniform closure recovery after a preserved Phi request reached the cap in both `-001` (384 tokens) and `-002` (512 tokens); all three bundled bases, and the selected base's train generation, use the same repaired contract. A Phi content-only `-001` job also reached `finish=length`; all axis-only systems therefore use the exact same schema bound with their fixed 192-token budget in fresh `-002` lineage. The local installed XGrammar compiler accepted that strict JSON schema with the Phi tokenizer before the fresh run.
- Encoder hard gates: canonical counts/checksums, no candidate-score access, finite Trainer metrics, completed state checksum, 400/400 predictions, finite RMSE/Spearman, and no row-level outputs.
- No external API call, new writing dataset, selection artifact, reward-model/RL job, raw text, prompt, rationale, identifier, candidate score, or prediction is committed. Generated train/validation rationales required by the downstream stage remain only in the ignored restricted root.

## Execution order and recovery envelope

1. Implement/configure contracts and run GPU-free tests.
2. Download/verify public model snapshots, inspect template support, then run one GPU0 one-update actual-input maintained-`SFTTrainer` preflight for one representative decoder task. This is the sole small training gate; generation and judging are exercised directly on the declared full runs rather than adding separate duplicate smokes.
3. Run the 12 decoder SFT/evaluation jobs sequentially on GPUs 0–3, aggregate results, and select the declared best decoder without retuning.
4. Generate the selected decoder's train rationales once, then run the six score regressors sequentially on GPUs 0–3 and evaluate all 400 validation essays.
5. A transport, template, serialization, data-contract, Trainer, or resource-setup failure may be repaired and replayed up to three times at the same stage. A failed scientific hard gate is preserved as a result and stops the dependent stage; it is not silently retuned.

## Transition ledger

```json
{"run_id":"api-rationale-sft-score-regression-v1","stage":"task_card","event":"start","failure_family":"none","repair_iteration":0,"evidence_ref":"this aggregate-only task card","command_ref":"new v1 launchers","resource_scope":"none","decision":"continue"}
{"run_id":"api-rationale-sft-v1-phi4_mini-bundle-gpu0_smoke-001","stage":"gpu0_actual_sft_preflight","event":"smoke_pass","failure_family":"none","repair_iteration":1,"evidence_ref":"outputs/api-rationale-sft-v1/api-rationale-sft-v1-phi4_mini-bundle-gpu0_smoke-001/training_complete.json (aggregate only: one record, one update, finite train loss 2.410813570022583)","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py gpu0-smoke","resource_scope":"GPU 0 only","decision":"continue to declared full matrix"}
{"run_id":"api-rationale-sft-v1-ax4_light-bundle-001","stage":"full_decoder_sft","event":"failed_numerical_gate","failure_family":"bf16_backward_nonfinite","repair_iteration":1,"evidence_ref":"preserved ignored full-numerical-gate-failed-001 ledger/log: seven logged non-finite gradient norms followed by masked zero losses; no generation, judge, or downstream stage was run","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full","resource_scope":"GPUs 0–3","decision":"stop and repair numerical setup"}
{"run_id":"api-rationale-sft-v1-ax4_light-bundle-numeric_recovery-001","stage":"four_gpu_numerical_recovery","event":"pass","failure_family":"resolved_bf16_backward_nonfinite","repair_iteration":2,"evidence_ref":"outputs/api-rationale-sft-v1/api-rationale-sft-v1-ax4_light-bundle-numeric_recovery-001/training_complete.json (all 6,000 source candidates loaded, five full distributed updates, finite train loss 2.9774662017822267)","command_ref":"torch.distributed.run --nproc_per_node=4 scripts/train_api_rationale_sft.py --config ...numeric_recovery...json","resource_scope":"GPUs 0–3","decision":"rerun declared full matrix with float32 decoder compute"}
{"run_id":"api-rationale-sft-v1-ax4_light-bundle-001","stage":"full_decoder_sft","event":"pass","failure_family":"none","repair_iteration":2,"evidence_ref":"outputs/api-rationale-sft-v1/api-rationale-sft-v1-ax4_light-bundle-001/training_complete.json (6,000 records, 188 distributed updates, finite train loss 1.4493048533480217, fixed config and float32 adapter provenance)","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full","resource_scope":"GPUs 0–3","decision":"preserve completed adapter; generation is the next dependent stage"}
{"run_id":"api-rationale-generation-v1-ax4_light-bundle-validation-001","stage":"vllm_generation_server","event":"failed_transport_gate","failure_family":"flashinfer_jit_requires_unavailable_ninja","repair_iteration":1,"evidence_ref":"preserved ignored outputs/api-rationale-sft-score-regression-v1/20260721-001/full/logs/server-generation-ax4_light-bundle-validation.log; no health response and no generated output","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full","resource_scope":"GPUs 0–3","decision":"stop before generation; repair only the vLLM execution backend/sampler path"}
{"run_id":"api-rationale-generation-v1-vllm-native-sampler-gate-006","stage":"vllm_generation_server","event":"repair_pass","failure_family":"resolved_flashinfer_jit_requires_unavailable_ninja","repair_iteration":2,"evidence_ref":"ignored outputs/api-rationale-sft-score-regression-v1/20260721-001/vllm-native-sampler-gate-006.log (four-rank TP health endpoint returned HTTP 200; CUDA graphs remained enabled; all GPUs 0–3 held approximately 76.7 GiB after health)","command_ref":"vLLM TP=4 with FlashAttention backend, native PyTorch sampler, disabled FlashInfer autotune/custom all-reduce fusion","resource_scope":"GPUs 0–3","decision":"resume exact matrix with a fresh runtime ledger and reuse only the completion record that exactly matches the fixed SFT config"}
{"run_id":"api-rationale-sft-score-regression-v1-full-resume-001","stage":"runner_setup","event":"failed_transport_gate","failure_family":"missing_runtime_logs_directory","repair_iteration":1,"evidence_ref":"preserved ignored outputs/api-rationale-sft-score-regression-v1/20260721-001/full-resume-001/runner.log (server log file could not be opened; no vLLM subprocess or generation request was started)","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3 idle","decision":"preserve fresh runtime; create the server-log parent before the subprocess and relaunch a new runtime"}
{"run_id":"api-rationale-generation-v1-ax4_light-bundle-validation-001","stage":"full_decoder_generation","event":"pass","failure_family":"none","repair_iteration":2,"evidence_ref":"ignored decoder generation aggregate (400 expected/observed/parse-valid, zero transport or schema failures, adapter completion checksum bound)","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3 TP=4","decision":"preserve completed validation generation"}
{"run_id":"api-rationale-sft-score-regression-v1-full-resume-002","stage":"resource_teardown","event":"failed_transport_gate","failure_family":"cuda_release_not_complete_before_next_stage","repair_iteration":1,"evidence_ref":"preserved ignored outputs/api-rationale-sft-score-regression-v1/20260721-001/full-resume-002/runner.log (generation completed and server shutdown was requested; immediate next-stage idle check found GPU 0 still allocated)","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3","decision":"preserve completed generation; add a bounded 90-second idle-release wait and verify/reuse its aggregate before a new runtime"}
{"run_id":"api-rationale-generation-v1-phi4_mini-bundle-validation-001","stage":"full_decoder_generation","event":"failed_hard_gate","failure_family":"response_finish_length","repair_iteration":1,"evidence_ref":"preserved ignored generation aggregate: 400 expected/observed, 399 parse-valid, one finish-category failure; no score field was read or prompted","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3 TP=4","decision":"do not judge or select this incomplete output; regenerate every bundled base in fresh -002 lineage with the same 512-token closure budget"}
{"run_id":"api-rationale-generation-v1-phi4_mini-bundle-validation-002","stage":"bundled_generation_recovery","event":"failed_hard_gate","failure_family":"deterministic_response_finish_length","repair_iteration":2,"evidence_ref":"preserved ignored generation aggregate: 400 expected/observed, 399 parse-valid, one finish-category failure; its private source ID matched -001 in memory but was not written to this record","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3 TP=4","decision":"do not judge or select this incomplete output; preserve it and add a schema-level closure bound uniformly"}
{"run_id":"api-rationale-generation-v1-*-bundle-*-003","stage":"bundled_generation_recovery","event":"planned_uniform_repair","failure_family":"resolved_response_finish_length_pending_execution","repair_iteration":3,"evidence_ref":"Phi completed -001 rationales had per-axis character maxima 145/114/119; the exact strict JSON schema with maxLength=192 compiled under installed XGrammar and the Phi tokenizer","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3 TP=4","decision":"apply 512 tokens plus maxLength=192 per axis uniformly to every bundled validation output and selected-base train output; preserve all -001/-002 artifacts"}
{"run_id":"api-rationale-generation-v1-ax4_light-bundle-validation-003","stage":"bundled_generation_recovery","event":"pass","failure_family":"resolved_response_finish_length","repair_iteration":3,"evidence_ref":"ignored generation aggregate: 400 expected/observed/parse-valid, zero transport/schema failures; per-axis observed character maxima 157/126/141","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3 TP=4","decision":"retain as the comparable bundled A.X validation output"}
{"run_id":"api-rationale-generation-v1-phi4_mini-bundle-validation-003","stage":"bundled_generation_recovery","event":"pass","failure_family":"resolved_deterministic_response_finish_length","repair_iteration":3,"evidence_ref":"ignored generation aggregate: 400 expected/observed/parse-valid, zero transport/schema failures; per-axis observed character maxima 192/115/192, all within the declared schema bound","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3 TP=4","decision":"retain as the comparable bundled Phi validation output; continue the remaining declared decoder matrix"}
{"run_id":"api-rationale-sft-v1-phi4_mini-content-001","stage":"full_decoder_sft","event":"pass","failure_family":"none","repair_iteration":1,"evidence_ref":"ignored training completion: 6,000 records, 188 distributed updates, finite train loss 1.151180496875276","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3","decision":"continue to its declared validation generation"}
{"run_id":"api-rationale-generation-v1-phi4_mini-content-validation-001","stage":"axis_generation","event":"failed_hard_gate","failure_family":"response_finish_length","repair_iteration":1,"evidence_ref":"preserved ignored generation aggregate: 400 expected/observed, 399 parse-valid, one finish-category failure; completed outputs had p99 122 and maximum 144 rationale characters","command_ref":"scripts/run_api_rationale_sft_score_regression_v1.py full-resume","resource_scope":"GPUs 0–3 TP=4","decision":"preserve the incomplete artifact; regenerate every base/axis uniformly in fresh -002 lineage with the already compiled strict maxLength=192 schema"}
{"run_id":"api-rationale-sft-score-regression-v1-full-resume-006","stage":"qwen_v6_judge_server","event":"failed_transport_gate","failure_family":"cuda_graph_warmup_requires_unavailable_ninja","repair_iteration":1,"evidence_ref":"preserved ignored full-resume-006 server log: vLLM DP=4 process exited before /health and before any judge request; root cause was a worker CUDA-graph warmup compiler invocation with `[Errno 2] No such file or directory: ninja`","command_ref":"vLLM Qwen3.6-35B-A3B-FP8 DP=4, FlashAttention, native sampler, prefix caching, CUDA graphs","resource_scope":"GPUs 0–3","decision":"start a fresh resume runtime; preserve topology/model/batching/prompts/seeds and use `--enforce-eager` solely for the Qwen judge server"}
{"run_id":"api-rationale-judge-v1-ax4_light-bundle-validation-001","stage":"qwen_v6_judge_execution","event":"failed_transport_gate","failure_family":"flashinfer_fp8_blockscale_gemm_requires_unavailable_ninja","repair_iteration":2,"evidence_ref":"preserved ignored full-resume-007 server/client logs: eager DP=4 health passed, but the first request selected FlashInfer FP8 block-scale GEMM and every resulting observation was invalid; no usable score distribution or selection was created","command_ref":"vLLM Qwen3.6-35B-A3B-FP8 DP=4, FlashAttention, eager, native sampler, prefix caching","resource_scope":"GPUs 0–3","decision":"preserve the append-only judge output; create fresh judge -002 outputs and set the documented `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` fallback, with a synthetic one-token inference gate before restricted requests"}
{"run_id":"api-rationale-judge-v1-ax4_light-bundle-validation-002","stage":"qwen_v6_judge_execution","event":"pass","failure_family":"resolved_flashinfer_fp8_blockscale_gemm_requires_unavailable_ninja","repair_iteration":3,"evidence_ref":"ignored aggregate judge report: 20,000/20,000 scored and schema-valid observations, zero abstentions or transport/schema failures, five prompt forms","command_ref":"vLLM Qwen3.6-35B-A3B-FP8 DP=4 with FlashInfer FP8 block-scale linear path disabled","resource_scope":"GPUs 0–3","decision":"reuse only after exact completion/config verification"}
{"run_id":"api-rationale-judge-v1-ax4_light-axis_triplet-validation-002","stage":"axis_triplet_judge_loader","event":"failed_contract_gate","failure_family":"axis_triplet_loaded_as_bundle_task","repair_iteration":1,"evidence_ref":"preserved ignored full-resume-008 client log: the axis-triplet output was rejected before output creation because the loader incorrectly required task=bundle; no judge observation was written","command_ref":"scripts/judge_api_rationales_v6.py","resource_scope":"GPUs 0–3 server already running","decision":"repair loader to bind `task=config.system_kind`; preserve and reuse only the independent completed bundled judge report"}
{"run_id":"api-score-regression-v1-qwen25_7b-direct-001","stage":"score_regression_train_eval","event":"failed_numerical_and_contract_gate","failure_family":"nonfinite_trainable_parameters_and_completed_training_output_revalidated_as_fresh","repair_iteration":1,"evidence_ref":"preserved ignored full-resume-009 training/evaluation logs: after the second logged interval gradient norm became non-finite, saved trainable tensors were non-finite; subsequent evaluation failed before prediction because its loader required the completed training output to be absent","command_ref":"scripts/train_api_score_regression.py and scripts/evaluate_api_score_regression.py under four-rank Trainer","resource_scope":"GPUs 0–3","decision":"do not use the invalid -001 model; start all six fresh -002 runs with the recorded numerical guard and corrected evaluation validation"}
```

## Pre-execution evidence

## Verified intermediate results

All values below are aggregate-only results from the ignored durable runner
ledger and completion records. A generation result is recorded only after all
400 frozen-validation outputs satisfy the parse and transport gates.

| Decoder base | Task | SFT records / steps | Finite train loss | Validation generation |
|---|---:|---:|---:|---|
| A.X-4.0-Light | bundled three-axis rationale | 6,000 / 188 | 1.449305 | comparable uniform `-003` passed: 400/400 parse-valid; 0 transport/schema failures (per-axis max chars 157/126/141) |
| A.X-4.0-Light | content-only rationale | 6,000 / 188 | 1.165183 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |
| A.X-4.0-Light | organization-only rationale | 6,000 / 188 | 1.241129 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |
| A.X-4.0-Light | expression-only rationale | 6,000 / 188 | 1.138780 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |
| Phi-4-mini-instruct | bundled three-axis rationale | 6,000 / 188 | 1.356893 | comparable uniform `-003` passed: 400/400 parse-valid; 0 transport/schema failures (per-axis max chars 192/115/192) |
| Phi-4-mini-instruct | content-only rationale | 6,000 / 188 | 1.151180 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |
| Phi-4-mini-instruct | organization-only rationale | 6,000 / 188 | 1.186378 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |
| Phi-4-mini-instruct | expression-only rationale | 6,000 / 188 | 1.190981 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |
| Midm-2.0-Base-Instruct | bundled three-axis rationale | 6,000 / 188 | 1.267099 | comparable uniform `-003` passed: 400/400 parse-valid; 0 transport/schema failures |
| Midm-2.0-Base-Instruct | content-only rationale | 6,000 / 188 | 1.083011 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |
| Midm-2.0-Base-Instruct | organization-only rationale | 6,000 / 188 | 1.198948 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |
| Midm-2.0-Base-Instruct | expression-only rationale | 6,000 / 188 | 1.194543 | uniform axis `-002` passed: 400/400 parse-valid; 0 transport/schema failures |

- All four required public snapshots were retrieved to the already-ignored `outputs/model-cache/` root at the immutable revisions above. The three decoder snapshots and KURE loaded their configuration with `trust_remote_code=false`; the Phi snapshot contains repository Python files but the installed integrated `Phi3ForCausalLM` architecture loaded without executing them.
- Each decoder tokenizer's **native** `apply_chat_template` rendered synthetic system/user/assistant boundaries and a distinct generation prompt. Aggregate rendered-template hashes were recorded in the local preflight only; no writing or rationale text was emitted. The resolved architectures were Qwen2 for A.X-4.0-Light, Phi3 for Phi-4-mini-instruct, and Llama for Mi:dm.
- Actual restricted input checks passed: 2,000 train / 400 validation writing records and 6,000 / 1,200 candidates, exactly three candidates per essay and all three rationale axes. Decoder-side source rows carried no loaded writing scores; candidate score values were never accessed. The score-regression stage alone loads the canonical source analytic labels.
- The actual maintained-`SFTTrainer` GPU0 gate used the Phi-4-mini bundled task, loaded the declared candidate source but limited training to one record and one update, and completed with finite metrics. Its adapter and all run artifacts remain ignored; only its aggregate completion record is cited above. A first wrapper launch was preserved separately without entering the runner because its outer log redirection pre-created the protected runtime root; the runner now permits a pre-existing root containing only that outer `runner.log` and continues to reject every stage artifact reuse.
- The first full decoder attempt exposed a numerical integration failure rather than a model-quality result: with bf16 base/adapter compute, the A.X bundled SFT logged seven non-finite gradient norms and subsequent zero losses. The run and partial adapter root were renamed into ignored failure-preservation locations; no dependent output was produced. GPU0 diagnostics reproduced the issue after five optimizer updates with finite forward losses, and a five-update all-four-GPU recovery using float32 base and LoRA parameters completed with finite loss and finite gradient norm. This establishes the recorded float32 decoder-compute repair for the declared rerun; it does not consume validation data or alter the data, targets, candidate-score exclusion, epochs, learning rate, batch, seed, or model-selection rule.
- The repaired full A.X bundled SFT then completed all 6,000 candidate records with finite metrics. The following TP=4 vLLM server failed before its health gate because this fixed local vLLM installation attempted an optional FlashInfer JIT path that requires the unavailable `ninja` executable. No generation request or restricted output was written. The replacement is a transport-only setup repair: FlashAttention and native PyTorch sampling replace that optional FlashInfer path; vLLM tensor parallelism, compilation, CUDA graphs, prefix caching, and LoRA serving remain enabled. A real four-GPU `/health` gate passed with the repaired command before the fresh full-resume launcher was permitted. The fresh runtime may reuse the already completed A.X adapter only when `training_complete.json` exactly matches the immutable full config and verifies float32 LoRA provenance; all other SFT jobs remain new and every generation/judge/regression artifact remains fresh.

## Final verified results

The final aggregate verifier confirmed the following completed gates without
opening or recording row-level writing, rationale, candidate-score, or
prediction data:

- 12/12 decoder SFT completion records have finite metrics and adapters.
- The three bundled and nine axis-only decoder validation generations are each
  400/400 parse-valid; each axis-triplet merge is 400/400.
- Each of the six v6 judge reports contains 20,000/20,000 schema-valid scored
  observations (five prompt forms × ten seeds × three axes × 400 essays), with
  zero abstentions, transport failures, and schema failures.
- The selected bundled decoder produced 2,000/2,000 parse-valid train
  rationales. All six score regressors have finite completed state and 400/400
  validation predictions.

### Score-only rationale-judge comparison

The fixed v6 judge sees the essay, target axis/rubric, and one score-free
rationale only. It receives neither a writing score nor a candidate score.
The values below are means over its 50 fixed independent observations per
rationale; `Δ API` is relative to the frozen API-rationale validation baseline
on that axis.

| Decoder base | System | Content mean / Δ API | Organization mean / Δ API | Expression mean / Δ API | Macro mean |
|---|---|---:|---:|---:|---:|
| A.X-4.0-Light | bundled | 3.685500 / -0.466933 | 3.869800 / -0.291717 | 3.743250 / -0.699400 | 3.766183 |
| A.X-4.0-Light | axis triplet | 3.867950 / -0.284483 | 3.930350 / -0.231167 | 3.968550 / -0.474100 | 3.922283 |
| Phi-4-mini-instruct | bundled | 2.999900 / -1.152533 | 3.313200 / -0.848317 | 2.836150 / -1.606500 | 3.049750 |
| Phi-4-mini-instruct | axis triplet | 2.979400 / -1.173033 | 3.423850 / -0.737667 | 2.986450 / -1.456200 | 3.129900 |
| Midm-2.0-Base-Instruct | bundled | 3.839750 / -0.312683 | 3.896100 / -0.265417 | 3.868050 / -0.574600 | **3.867967** |
| Midm-2.0-Base-Instruct | axis triplet | 3.882450 / -0.269983 | 3.936900 / -0.224617 | 3.946150 / -0.496500 | 3.921833 |

Under the predeclared all-axis downstream rule, the selected model is the
**Midm-2.0-Base-Instruct bundled decoder** (macro 3.867967). The separately
trained axis triplets are reported as requested but are not eligible to replace
the required single bundled decoder. Every decoder system is below the API
baseline on every judged axis; the selection is therefore a relative choice
among bundled student decoders, not evidence of parity with the API rationales.

### Score-regression validation

Each number is aggregate-only on the 400 frozen-validation essays. Lower RMSE
and higher tie-aware Spearman are better.

| Encoder | Input condition | Content RMSE / rho | Organization RMSE / rho | Expression RMSE / rho | Macro RMSE | Macro rho |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B | essay only | 0.579966 / 0.578227 | 0.757022 / 0.581470 | 0.630082 / 0.298266 | 0.655690 | 0.485988 |
| Qwen2.5-7B | essay + API rationale | 0.551263 / 0.650042 | 0.692677 / 0.740603 | 0.612106 / 0.449791 | **0.618682** | **0.613479** |
| Qwen2.5-7B | essay + selected decoder rationale | 0.596269 / 0.567039 | 0.783287 / 0.608711 | 0.644938 / 0.340942 | 0.674831 | 0.505564 |
| KURE-v1 | essay only | 0.595548 / 0.499952 | 0.790135 / 0.509398 | 0.626451 / 0.241549 | 0.670711 | 0.416966 |
| KURE-v1 | essay + API rationale | 0.561570 / 0.638424 | 0.717917 / 0.692567 | 0.577778 / 0.339876 | **0.619088** | **0.556956** |
| KURE-v1 | essay + selected decoder rationale | 0.599535 / 0.536824 | 0.787686 / 0.537757 | 0.612376 / 0.131407 | 0.666532 | 0.401996 |

### Interpretation and next decision

- API rationales improve both encoders over essay-only input (Qwen macro RMSE
  0.655690 → 0.618682 and macro rho 0.485988 → 0.613479; KURE 0.670711 →
  0.619088 and 0.416966 → 0.556956).
- The selected Midm bundled decoder does **not** reproduce that gain: its macro
  RMSE is worse than essay-only for both Qwen (0.674831) and KURE (0.666532).
  Qwen macro rho rises modestly over essay-only (0.505564 vs. 0.485988), but
  KURE macro rho declines (0.401996 vs. 0.416966).
- Thus this result is a negative transfer result for the current SFT decoder
  rationale condition. The judge's relative preference among student outputs is
  insufficient evidence of downstream score-regression utility, and the
  score-only API-rationale condition is the empirical reference to preserve
  before any reward-model or reinforcement-learning stage. No post-validation
  retuning was performed.
