# Official rationale RL exact-judge run 019

- Launch code Git SHA: `7c930aae6fad9a5e1fc2fe3125b8b2484041560b`.
- Authorized GPUs: physical 0–3 only, four NVIDIA H100 80GB HBM3; GPUs 4–7 are neither queried nor used.
- Existing `.venv-standard`: PyTorch 2.11.0+cu130, Transformers 5.14.1, TRL 0.29.1, vLLM 0.25.1.
- SHA-256: judge prompt `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`; evaluation prompt `1950b3f837bf39003214e718f972531d442060e3c611bef1da7e1145`; DPO config `a97c14756a2ed30229763ee9623854acb2c22c53b00dd1eaed6b83691d8b0e99`; GRPO config `749afaf708bd13834965e0e749ccc96d9e6e44fd79176c92e791b22ef1ef4980`; train source `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Seeds: DPO 2026072707, GRPO 2026072708, judge 42; replacement attempt seed is `original_seed + attempt*1000003 + slot_index`.
- Train-only preferences/reward; validation excluded; row artifacts restricted and ignored.

```bash
PYTHONPATH=src:. .venv-standard/bin/python scripts/run_official_rationale_rl_experiment.py \
  --run-id official-rationale-rl-experiment-v1-exact-judge-20260729-019 \
  --dpo-config configs/official_rationale_dpo.llm_as_judge_txt.v2.json \
  --grpo-config configs/official_rationale_grpo.llm_as_judge_txt.v2.json \
  --scope all --grpo-task bundle --grpo-phase pilot --grpo-phase full
```

Outputs use matching ignored orchestration/restricted roots. Relative to run 018, the runtime-incompatible Hangul regex is removed. Invalid candidates use at most four independent `n=1` replacement attempts; valid originals remain fixed. Malformed `stop` JSON is an invalid candidate. Aggregate request counts include all attempts and candidate counts identify original invalid slots. Compact XGrammar, 384-character post-check, 7,100/2,400 capacities, 8,192 context, prompts, judge, data, candidate contract, and objectives are unchanged. Run-018 negative evidence is preserved.

Preflight: 46 targeted tests and `git diff --check` passed. Append terminal aggregate results/deviations after completion or failure.

## Bundle-only terminal result and protocol correction

The bundle rollout completed all 6,000 train groups and 24,000 candidates. It
required 90 semantic-invalid candidate replacements across 91 bounded retry
requests, with no length retry. The exact Q4 judge completed all 24,000
judgments at mean `59.598875/60` and population standard deviation
`1.6062037234`.

Preference assembly retained 2,770 groups and excluded 3,230 all-tie groups.
The zero-variance fraction was `0.5383333333`; the retained mean best-minus-
worst difference was `2.8859206/60`, and the frozen reward-variance gate
passed.

The initial executor also began the predeclared axis-triplet judge. The user
then clarified that the deployed evaluator consumes one bundled three-axis
JSON and that splitting axes would change the target contract. The owned
axis-triplet judge was stopped. Its partial judgments were not checkpointed,
are not used for training or selection, and remain only as server logs. This
is a user-authorized protocol correction, not a negative model result.

Bundle DPO was then trained with TRL `DPOTrainer` over 2,770 retained rows on
DDP world size 4. Per-rank gradient accumulation was 4, preserving configured
global accumulation 16. Training completed 174 updates in 590.6284 seconds;
train loss was `0.6677307`, final logged loss `0.6620018`, reward accuracy
`0.5769231`, and reward margin `0.1463470`. The frozen reference-adapter hash
was unchanged. The trained adapter is:

`outputs/official-rationale-rl-v1/orchestration/official-rationale-rl-experiment-v1-exact-judge-20260729-019/models/dpo-official-bundle-ddp4-full-user-aligned-001/adapter/`

Frozen-validation comparison and final rationale selection are recorded in
`official_rationale_dpo_bundle_validation_20260729_020.md`.
