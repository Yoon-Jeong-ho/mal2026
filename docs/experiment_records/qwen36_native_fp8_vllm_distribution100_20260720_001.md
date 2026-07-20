# Native-FP8 vLLM 100-score distribution collection — 2026-07-20

- **Status:** superseded at the user's request before completion; not eligible for selection or training.
- **Authorization:** direct user instruction to score each generated train and frozen-validation rationale 100 times.
- **Purpose:** collect candidate-isolated pointwise LLM-judge score distributions; no selection, SFT, DPO, or GRPO is authorized by this run.
- **Git SHA:** `86902f1e3a077b1178d1297a1dcccf10e929453d`
- **Configuration:** `configs/qwen36_native_fp8_vllm_distribution100.v1.json` (SHA-256 `18c4d430be0d69ad478f0ec368538e983a18543a6734141c1de3fbbe71a4133e`).
- **Model/runtime:** local `Qwen/Qwen3.6-35B-A3B-FP8`, revision `95a723d08a9490559dae23d0cff1d9466213d989`; vLLM 0.25.1; one self-contained DP=4, TP=1 endpoint on physical GPUs 0–3; 64 sequences per DP rank, 256 client requests in flight, 4,096-token context, eager mode, prefix caching, fixed `--generation-config vllm`.
- **Input lineage:** validated generated-candidate batch SHA-256 `4ef414dd35b831092fcea24c7770a5f18fbe0df1d4e6aa74d55613b8cad71e2e`; train scoped artifact SHA-256 `d69dd9bc349c117a55b4bf652507135f084c9045d95d5939da3980223893aecf`; validation scoped artifact SHA-256 `21c7b97e6faf6d8092b4a27e35b60083f9b9b60861493867061816fcb12f9d83`.
- **Sampling:** 5 prompt layouts × 5 rubric permutations × 4 fixed seeds at temperature 0.15 = exactly 100 independent score requests per generated rationale.
- **Planned calls:** train 600,000; frozen validation 120,000.  Validation artifacts are written separately and are prohibited from influencing selection or training.
- **Command:** `scripts/run_qwen36_native_fp8_vllm_distribution100_full.sh` in durable tmux session `mal2026-dist100-full-001`.
- **Artifacts:** ignored restricted score-observation/distribution directories and `outputs/native-fp8-vllm-distribution100/qwen36-native-fp8-dist100-train-20260720-full-001/`.
- **Smoke evidence:** GPU0 and DP4 each completed one real candidate × 100 scores with 100/100 schema-valid scored observations and no transport/schema failures. Aggregate gate: `outputs/aggregate-reports/qwen36-native-fp8-dist100-20260720-005.dp4-smoke-to-full-train.gate-summary.json`.
- **Known deviation:** vLLM eager mode disables torch.compile/CUDA Graph capture to avoid the observed unbounded startup path; this is held fixed for train and validation.
- **Stop record:** at `2026-07-20T11:05:10+00:00`, the user requested that the source writing score not be shown to the judge.  The train process was interrupted cleanly after 60,186 of 600,000 observations (60,180 scored, 6 abstentions, 0 transport/schema failures; 601 complete candidates plus 86 partial observations).  These restricted partial observations remain preserved for audit only and must not be combined with the score-blind v2 results.
