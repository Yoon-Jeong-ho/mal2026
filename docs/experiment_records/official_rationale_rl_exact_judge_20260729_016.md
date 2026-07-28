# Official rationale RL exact-judge run 016

- Launch code Git SHA: `b79a6586c67c03c48e472c4afabece9321c433b3`.
- Authorized GPUs: physical 0–3 only, four NVIDIA H100 80GB HBM3; GPUs 4–7 are neither queried nor used.
- Existing `.venv-standard`: PyTorch 2.11.0+cu130, Transformers 5.14.1, TRL 0.29.1, vLLM 0.25.1.
- SHA-256: judge prompt `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`; evaluation prompt `1950b3f837bf39003214e718f972531d442060e3c611bef1da7e1145`; DPO config `a97c14756a2ed30229763ee9623854acb2c22c53b00dd1eaed6b83691d8b0e99`; GRPO config `749afaf708bd13834965e0e749ccc96d9e6e44fd79176c92e791b22ef1ef4980`; train source `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Seeds: DPO 2026072707, GRPO 2026072708, judge 42; malformed slot replacement uses `original_seed + 1000003 + slot_index`.
- Train-only preferences and reward; validation excluded; restricted row artifacts remain ignored.

```bash
PYTHONPATH=src:. .venv-standard/bin/python scripts/run_official_rationale_rl_experiment.py \
  --run-id official-rationale-rl-experiment-v1-exact-judge-20260729-016 \
  --dpo-config configs/official_rationale_dpo.llm_as_judge_txt.v2.json \
  --grpo-config configs/official_rationale_grpo.llm_as_judge_txt.v2.json \
  --scope all --grpo-task bundle --grpo-phase pilot --grpo-phase full
```

Outputs use matching ignored orchestration/restricted run roots. Relative to run 015, vLLM structured output is explicitly XGrammar with arbitrary JSON whitespace disabled, bundle completion transport capacity is 7,100 tokens, single-axis capacity is 2,400, and server context is 8,192. The decoded schema remains capped at 384 characters per axis. Largest audited prompt 1,023 plus completion 7,100 equals 8,123, below context. All prompts, candidates, judge, scores, data, and training objectives remain unchanged. Run-015 negative evidence is preserved.

Preflight: local vLLM 0.25.1 configuration source verified `disable_any_whitespace` support for XGrammar; 46 targeted tests, dry-plan 8,192-token assertions, and `git diff --check` passed. Append terminal metrics/deviations after completion or failure.

## Terminal result

Run 016 failed at 07:22:20 KST in the 32-group rollout smoke. The server successfully applied XGrammar, `disable_any_whitespace=true`, and the 8,192-token context. A schema-complete candidate then violated the post-schema scientific contract because at least one rationale field contained no Hangul. The prior generator treated any such sampled candidate as a fatal stage error rather than replacing only the invalid slot.

The next integration repair classifies schema-complete but non-Korean/over-384 candidates separately from transport truncation, preserves every valid candidate, and uses the same independent `n=1` frozen-seed replacement for only invalid slots. Aggregate records expose semantic and length retries separately. The required four valid Korean candidates and all other protocol elements remain unchanged.
