# Official rationale RL exact-judge run 013

## Reproducibility record

- Launch code Git SHA: `a2c6847c2db2b64088875d09404695b465493e64`.
- Authorized hardware: physical GPUs 0–3 only, four NVIDIA H100 80GB HBM3. GPUs 4–7 are neither queried nor used.
- Existing environment: `.venv-standard`; PyTorch 2.11.0+cu130, Transformers 5.14.1, TRL 0.29.1, vLLM 0.25.1.
- Exact judge prompt SHA-256: `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`.
- Evaluation prompt SHA-256: `1950b3f837bf39003214e718f972531d442060e3c611bef1da7e1145`.
- DPO config SHA-256: `86e5adfb2691e1641b58110c58ee3856dc119c1f53d85d581dc1f6204f2da026`.
- GRPO config SHA-256: `55bdf71b1ba5528948bea4db2c0a615f8d5b90c85fa5f5431e75e6c390ccfa1d`.
- Train source SHA-256: `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Seeds: DPO 2026072707, GRPO 2026072708, judge 42. A transport-truncated replacement uses the frozen alternate seed `original_seed + 1000003`.
- Data boundary: train split only for preference/reward construction; validation is excluded. Row artifacts remain restricted and ignored.

```bash
PYTHONPATH=src:. .venv-standard/bin/python scripts/run_official_rationale_rl_experiment.py \
  --run-id official-rationale-rl-experiment-v1-exact-judge-20260729-013 \
  --dpo-config configs/official_rationale_dpo.llm_as_judge_txt.v2.json \
  --grpo-config configs/official_rationale_grpo.llm_as_judge_txt.v2.json \
  --scope all --grpo-task bundle --grpo-phase pilot --grpo-phase full
```

Aggregate output is under `outputs/official-rationale-rl-v1/orchestration/official-rationale-rl-experiment-v1-exact-judge-20260729-013/`; restricted rows are under the matching run ID in `data/processed/restricted/official_rationale_rl_v1/`.

The only deviation from run 012 is the deterministic alternate seed for a malformed replacement request. Prompts, judge, input rows, four-candidate group size, 384-character-per-axis schema, 4,000-token bundle ceiling, 5,120-token vLLM context, score projection, and training objectives are unchanged. This repairs a retry that otherwise reproduced the same invalid stochastic sample. Run 012's failure and all prior negative evidence remain preserved.

Preflight: 46 targeted tests and `git diff --check` passed. Final metrics and deviations must be appended after the durable runner completes or fails.

## Terminal result

Run 013 failed at 06:18:43 KST in `dpo-smoke-judge`. The rollout smoke completed, but one exact-Q4 response hit the frozen 1,800-token judge output ceiling and ended in the middle of a JSON string (`finish_reason=length`). The server remained healthy and reported no truncation of its input context; the failure was the response token cap in `q4_score`, not the policy replacement-seed repair.

The next integration repair preserves every complete 1,800-token-or-shorter judgment. Only an incomplete `length` response is deterministically retried with the same prompt, temperature 0, and seed 42 at a 3,600-token ceiling so the same judgment can close its JSON object. An invalid `stop` response remains a hard failure. Prompt text, JSON schema, score projection, judge model, and scientific protocol are unchanged.
