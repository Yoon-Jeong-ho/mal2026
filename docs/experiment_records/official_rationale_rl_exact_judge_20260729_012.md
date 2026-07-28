# Official rationale RL with the exact judge prompt — run 012

## Scope and provenance

- Status at creation: launch-ready after capacity repair; results are written only to ignored output and restricted-data paths.
- Code Git SHA: `bf8355d33cab66eaa177a780bb72b3a09e8693dd`.
- Authorized GPU scope: physical GPUs 0–3 only (four NVIDIA H100 80GB HBM3, driver 580.105.08). GPUs 4–7 are neither queried nor used.
- Environment: existing `.venv-standard`; Python runner `.venv-standard/bin/python`; PyTorch 2.11.0+cu130, Transformers 5.14.1, TRL 0.29.1, vLLM 0.25.1.
- Judge prompt SHA-256: `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d` (`llm_as_judge.txt`).
- Evaluation prompt SHA-256: `1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145` (`evaluation.txt`).
- DPO config SHA-256: `86e5adfb2691e1641b58110c58ee3856dc119c1f53d85d581dc1f6204f2da026`.
- GRPO config SHA-256: `55bdf71b1ba5528948bea4db2c0a615f8d5b90c85fa5f5431e75e6c390ccfa1d`.
- Train source checksum observed by the preceding smoke: `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Seeds: DPO policy 2026072707; GRPO policy 2026072708; exact judge 42.
- Privacy: train split only for preference/reward construction. Validation is not used for preferences or reward. No row-level writing, identifier, score, prediction, or rationale is recorded here.

## Command and output

```bash
PYTHONPATH=src:. .venv-standard/bin/python scripts/run_official_rationale_rl_experiment.py \
  --run-id official-rationale-rl-experiment-v1-exact-judge-20260729-012 \
  --dpo-config configs/official_rationale_dpo.llm_as_judge_txt.v2.json \
  --grpo-config configs/official_rationale_grpo.llm_as_judge_txt.v2.json \
  --scope all --grpo-task bundle --grpo-phase pilot --grpo-phase full
```

- Aggregate/run output: `outputs/official-rationale-rl-v1/orchestration/official-rationale-rl-experiment-v1-exact-judge-20260729-012/`.
- Restricted row output: `data/processed/restricted/official_rationale_rl_v1/official-rationale-rl-experiment-v1-exact-judge-20260729-012/`.
- DPO rollout topology: tensor parallel 4 on GPUs 0–3, `max_model_len=5120`, `max_num_seqs=256`, `max_num_batched_tokens=32768`, CUDA graphs enabled (`enforce_eager=false`).
- GRPO topology: rollout TP2 on GPUs 0–1, trainer on GPU2, exact Q4 reward on GPU3.

## Preserved negative evidence and repair

Run 011 passed its 32-group smoke and one-update DPO training, then failed during the full bundle rollout after all 6,000 initial candidates and six retries had been served. The immediate failure was `policy rollout remained truncated after bounded length retry`; at least one strict JSON object remained incomplete at the former 2,800-token retry ceiling.

The run-011 smoke judged 128 candidates with mean 12-cell total 59.7421875/60 and population standard deviation 0.7628697. It retained 13 preference rows from 32 groups; the zero-variance group fraction was 0.59375 and passed the frozen 0.8 operational gate. These numbers diagnose a strongly ceiling-saturated judge and are not a claim of rationale quality.

The repair does not expand the frozen rationale field bound: every axis remains limited to 384 characters and is schema-validated. It raises the bundle completion transport ceiling from 2,400 to 4,000 tokens and the vLLM context from 4,096 to 5,120 tokens. The largest observed exact prompt was 1,023 tokens, so the configured upper bound is 1,023 + 4,000 = 5,023, within the 5,120-token server context. Single-axis DPO rollouts use 1,200 completion tokens. A retry keeps the same capacity-safe ceiling and replaces only incomplete candidate slots.

## Preflight evidence

- 46 targeted unit tests passed.
- `git diff --check` passed.
- CPU dry plan verified the exact judge binding, DPO/GRPO 5,120-token server contract, and the authorized GPU topology without querying GPUs.

Final metrics and deviations must be appended after the durable runner completes or fails.
