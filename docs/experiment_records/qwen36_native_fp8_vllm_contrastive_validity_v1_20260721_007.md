# Rationale-only judge contrastive-validity calibration — 2026-07-21

- **Status:** completed; the predeclared contrastive-validity gate passed. No reward-model labels, selection artifact, or RL job was created.
- **Authorization:** direct user authorization to carry out the first two corrective stages: construct deterministic degraded rationale controls and test whether the pointwise judge independently scores the original higher.
- **Question:** the completed v6 collection established execution reliability, not whether a high rationale-quality score is valid. This calibration measures contrastive sensitivity only; it does not create reward-model labels or select training examples.
- **Inputs and isolation:** use a deterministic calibration-only holdout of 80 train essay groups (240 generated candidates) from the validated train-only artifact. The selected candidate-key set is committed only by a digest in restricted outputs and must be excluded from later reward-model training or preference selection. No validation candidate or validation source text is opened; no source writing score is read or supplied to the judge.
- **Controls:** for every original rationale-only candidate, score three in-memory degraded controls independently against the same v6 judge: (1) axis rotation, (2) pairing the unchanged rationale with a different calibration essay, and (3) axis-appropriate but generic boilerplate. No raw text, prompts, identifiers, or completions are persisted or tracked.
- **Fixed judge protocol:** exactly the completed v6 candidate projection, score-only JSON response contract, five prompt emphases, ten fixed seeds, temperature 0.15, and Qwen/Qwen3.6-35B-A3B-FP8 FP8 model. This produces 50 independent pointwise scores for each original/control condition.
- **Planned calls:** 240 base candidates × 4 conditions × 50 = 48,000 train-calibration observations. GPU0 preflight uses one essay group (3 base candidates; 600 calls), then one DP=4 endpoint on GPUs 0–3 runs the full calibration.
- **Validity metrics:** pair same prompt type and seed between original and control; report per-axis strict-win/tie/loss rates, mean score margin, candidate-level confidence intervals, and aggregate rates. No side-by-side candidate comparison prompt is used.
- **Predeclared non-promotion gate:** require zero transport/schema failures plus candidate-level original-over-control strict-win rates of at least 0.75 for generic controls and 0.60 for axis-rotation and cross-essay controls, for every axis. A failed validity gate is a negative result: preserve it and do not use v6 scores as an unqualified RL reward target.
- **Resource envelope:** GPU0 only for preflight; GPUs 0–3 together for full collection; never GPUs 4–7. Use the existing v6 CUDA-graph runtime (192 sequences/DP rank, 768 in-flight requests, 65,536 batch tokens, 0.90 memory target).
- **Runner:** `scripts/run_qwen36_native_fp8_vllm_contrastive_validity_v1.sh gpu0`, then `full` after the declared preflight gate.

## Pre-execution evidence

- Static checks passed: Python compilation, Bash syntax, whitespace check, and three GPU-free contrastive-contract checks. The existing versioned collector-contract checks are replayed separately before launch; the contrastive checks explicitly bind the fixed v6 configuration.
- Actual-input population check passed without generating a score: the GPU0 stage resolves 1 source group / 3 base candidates / 12 conditions / 600 calls; the full stage resolves 80 / 240 / 960 / 48,000. Invalid source-or-candidate rows were 0; source writing scores read or prompted was `false`; validation source text opened was 0.

## Transition ledger

```json
{"run_id":"qwen36-native-fp8-rationale-contrastive-v1-train-20260720-gpu0_smoke-001","stage":"implementation_and_actual_input_preflight","event":"start","failure_family":"none","repair_iteration":0,"evidence_ref":"aggregate-only static and population checks recorded above","command_ref":"run_qwen36_native_fp8_vllm_contrastive_validity_v1.sh gpu0","resource_scope":"none","decision":"continue"}
{"run_id":"qwen36-native-fp8-rationale-contrastive-v1-train-20260720-gpu0_smoke-001","stage":"gpu0_execution_preflight","event":"smoke_pass","failure_family":"none","repair_iteration":0,"evidence_ref":"outputs/aggregate-reports/qwen36-native-fp8-rationale-contrastive-v1-20260721-001.gpu0-to-full.gate-summary.json","command_ref":"run_qwen36_native_fp8_vllm_contrastive_validity_v1.sh gpu0","resource_scope":"GPU0","decision":"continue"}
{"run_id":"qwen36-native-fp8-rationale-contrastive-v1-train-20260720-full-001","stage":"full_contrastive_calibration","event":"next_stage_complete","failure_family":"none","repair_iteration":0,"evidence_ref":"outputs/aggregate-reports/qwen36-native-fp8-rationale-contrastive-v1-20260721-001.final-analysis.json","command_ref":"run_qwen36_native_fp8_vllm_contrastive_validity_v1.sh full","resource_scope":"GPUs 0-3","decision":"complete"}
```

## GPU0 preflight result

- Completed 600/600 independent pointwise observations (3 base candidates × 4 conditions × 5 prompt forms × 10 seeds), with 600 schema-valid scores, 0 abstentions, and 0 transport/schema failures.
- All matching and completion hard gates passed. Each predeclared contrastive transform passed all three axes in this intentionally small preflight. This permits the planned full calibration; it is not the final validity conclusion.

## Full result

- One DP=4 vLLM endpoint on GPUs 0–3 completed 48,000/48,000 observations in 635 seconds (75.59 observations/s). All observations were schema-valid and scored; abstentions and transport/schema failures were both zero. All completion and pairing hard gates passed.
- Original rationale over generic boilerplate: candidate-level strict-win rates were 99.58% / 100.00% / 100.00% for content / organization / expression (threshold 75%).
- Original rationale over axis rotation: 98.33% / 97.92% / 98.75% (threshold 60%).
- Original rationale over a different essay: 99.58% / 99.58% / 99.17% (threshold 60%).
- The weakest prompt-form-specific rate after averaging its ten seeds was still 95.83% across every transform/axis cell, so this result was not caused by just one of the five prompt forms. Mean candidate-level original-minus-control margins ranged from 2.39 to 3.35 score points.
- The fixed v6 judge therefore passes this **controlled contrastive sensitivity** test: it strongly penalizes generic, wrong-axis, and wrong-essay diagnoses. This does not establish human-preference calibration or fine-grained ranking among plausible rationales; it is evidence against the earlier failure mode, not an unconditional approval for RL.
- Aggregate-only analysis is stored at `outputs/aggregate-reports/qwen36-native-fp8-rationale-contrastive-v1-20260721-001.final-analysis.json`. It contains no essays, rationales, identifiers, prompts, completions, source writing scores, validation text, selection artifact, or training labels.
