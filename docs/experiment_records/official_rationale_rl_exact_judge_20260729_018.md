# Official rationale RL exact-judge run 018

- Launch code Git SHA: `2a9e8edb7995d4e0a93ddbe09ba838839ec873b7`.
- Authorized GPUs: physical 0–3 only, four NVIDIA H100 80GB HBM3; GPUs 4–7 are neither queried nor used.
- Existing `.venv-standard`: PyTorch 2.11.0+cu130, Transformers 5.14.1, TRL 0.29.1, vLLM 0.25.1.
- SHA-256: judge prompt `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`; evaluation prompt `1950b3f837bf39003214e718f972531d442060e3c611bef1da7e1145`; DPO config `a97c14756a2ed30229763ee9623854acb2c22c53b00dd1eaed6b83691d8b0e99`; GRPO config `749afaf708bd13834965e0e749ccc96d9e6e44fd79176c92e791b22ef1ef4980`; train source `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Seeds: DPO 2026072707, GRPO 2026072708, judge 42; fallback invalid-slot replacement uses `original_seed + 1000003 + slot_index`.
- Train-only preferences/reward; validation excluded; row artifacts restricted and ignored.

```bash
PYTHONPATH=src:. .venv-standard/bin/python scripts/run_official_rationale_rl_experiment.py \
  --run-id official-rationale-rl-experiment-v1-exact-judge-20260729-018 \
  --dpo-config configs/official_rationale_dpo.llm_as_judge_txt.v2.json \
  --grpo-config configs/official_rationale_grpo.llm_as_judge_txt.v2.json \
  --scope all --grpo-task bundle --grpo-phase pilot --grpo-phase full
```

Outputs use matching ignored orchestration/restricted roots. Relative to run 017, the existing Korean-text validity rule is enforced during XGrammar decoding with `pattern: ".*[가-힣].*"` per rationale field instead of relying on one-shot rejection sampling. The 384-character limit, compact JSON, 7,100/2,400 capacities, 8,192 context, prompts, judge, data, candidate count, and objectives are unchanged. Run-017 negative evidence is preserved.

Preflight: local XGrammar compilation passed for exact single-axis and three-axis schemas; 46 targeted tests and `git diff --check` passed. Append terminal metrics/deviations after completion or failure.

## Terminal result

Run 018 failed at 08:02:32 KST in the rollout smoke. Although the schema with the Hangul regex compiled, at runtime XGrammar produced extremely long sequences and at least one malformed `stop` JSON response. The 32-group smoke required 55 requests and about six minutes instead of the usual approximately 25 seconds. Schema compilation was therefore insufficient evidence of runtime compatibility, and the regex experiment is preserved as a negative result and removed.

The next integration repair restores the prior schema and compact XGrammar behavior. Each invalid slot receives up to four independent `n=1` replacement attempts using seed `original_seed + attempt*1000003 + slot_index`; every valid original candidate remains frozen. Malformed `stop` JSON is also treated as an invalid candidate rather than a full-stage transport failure. Aggregate retry-request counts include every bounded attempt, while retry-candidate counts identify original invalid slots.
