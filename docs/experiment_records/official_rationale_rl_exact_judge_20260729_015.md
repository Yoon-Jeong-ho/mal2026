# Official rationale RL exact-judge run 015

- Launch code Git SHA: `6e1aff02fd77ceac9c6b464c6462e0e091c2de1a`.
- Authorized GPUs: physical 0–3 only, four NVIDIA H100 80GB HBM3; GPUs 4–7 are neither queried nor used.
- Existing `.venv-standard`: PyTorch 2.11.0+cu130, Transformers 5.14.1, TRL 0.29.1, vLLM 0.25.1.
- SHA-256: exact judge prompt `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`; evaluation prompt `1950b3f837bf39003214e718f972531d442060e3c611bef1da7e1145`; DPO config `86e5adfb2691e1641b58110c58ee3856dc119c1f53d85d581dc1f6204f2da026`; GRPO config `55bdf71b1ba5528948bea4db2c0a615f8d5b90c85fa5f5431e75e6c390ccfa1d`; train source `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Seeds: DPO 2026072707, GRPO 2026072708, judge 42; malformed slot replacement uses `original_seed + 1000003 + slot_index`.
- Train-only preference/reward construction; validation excluded; row artifacts remain ignored and restricted.

```bash
PYTHONPATH=src:. .venv-standard/bin/python scripts/run_official_rationale_rl_experiment.py \
  --run-id official-rationale-rl-experiment-v1-exact-judge-20260729-015 \
  --dpo-config configs/official_rationale_dpo.llm_as_judge_txt.v2.json \
  --grpo-config configs/official_rationale_grpo.llm_as_judge_txt.v2.json \
  --scope all --grpo-task bundle --grpo-phase pilot --grpo-phase full
```

Outputs use the matching run ID under the ignored orchestration and restricted roots. Relative to run 014, only malformed policy candidates change recovery shape: each incomplete slot is independently resampled with `n=1`; all valid initial candidates are retained. Prompt, four-valid-candidate contract, schema, generation parameters, judge, score projection, data, and training objectives are unchanged. The deterministic judge length retry introduced before run 014 remains active. All run-014 negative evidence is preserved.

Preflight: 46 targeted tests and `git diff --check` passed. Append terminal aggregate results and deviations after completion or failure.
