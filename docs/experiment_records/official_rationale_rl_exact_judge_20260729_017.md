# Official rationale RL exact-judge run 017

- Launch code Git SHA: `117798753e318e51df1931618aaf84b40ab26e30`.
- Authorized GPUs: physical 0–3 only, four NVIDIA H100 80GB HBM3; GPUs 4–7 are neither queried nor used.
- Existing `.venv-standard`: PyTorch 2.11.0+cu130, Transformers 5.14.1, TRL 0.29.1, vLLM 0.25.1.
- SHA-256: judge prompt `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`; evaluation prompt `1950b3f837bf39003214e718f972531d442060e3c611bef1da7e1145`; DPO config `a97c14756a2ed30229763ee9623854acb2c22c53b00dd1eaed6b83691d8b0e99`; GRPO config `749afaf708bd13834965e0e749ccc96d9e6e44fd79176c92e791b22ef1ef4980`; train source `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Seeds: DPO 2026072707, GRPO 2026072708, judge 42; malformed slot replacement uses `original_seed + 1000003 + slot_index`.
- Train-only preferences/reward; validation excluded; restricted rows remain ignored.

```bash
PYTHONPATH=src:. .venv-standard/bin/python scripts/run_official_rationale_rl_experiment.py \
  --run-id official-rationale-rl-experiment-v1-exact-judge-20260729-017 \
  --dpo-config configs/official_rationale_dpo.llm_as_judge_txt.v2.json \
  --grpo-config configs/official_rationale_grpo.llm_as_judge_txt.v2.json \
  --scope all --grpo-task bundle --grpo-phase pilot --grpo-phase full
```

Outputs use matching ignored orchestration/restricted roots. Relative to run 016, only invalid-candidate handling changes: schema-complete samples without Hangul or beyond the frozen 384-character bound are counted separately and their slots are independently replaced. All complete valid candidates remain untouched. The XGrammar compact JSON, 7,100/2,400 output capacities, 8,192 context, prompts, judge, data, four-valid-candidate requirement, and objectives are unchanged. Run-016 negative evidence is preserved.

Preflight: 46 targeted tests, Python compile, and `git diff --check` passed. Append terminal aggregate results/deviations after completion or failure.
