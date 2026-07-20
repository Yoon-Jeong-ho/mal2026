# Native-FP8 vLLM score-only five-prompt ten-repeat collection — 2026-07-20

- **Status:** static contract checks passed; GPU0 actual-candidate smoke is next.
- **Authorization:** direct user instruction to remove semantic abstention from the reward-model label protocol while retaining five prompt versions × ten repeats.
- **Motivation:** v3 mixed semantic uncertainty with data validity, yielding a 11.61% abstention rate and a failed score-coverage gate. v3 partial observations remain audit-only and are not used here.
- **Protocol:** no source writing score is read or supplied. Every structurally valid candidate receives three required 1–5 explanation-quality scores. An unsupported, vague, off-axis, unnatural, or unhelpful explanation receives a lower score; it never receives a semantic abstention. Transport/JSON/schema failures remain fail-closed and are retried once, then recorded as failures.
- **Five prompt versions:** all have the same payload order, three axes, response schema, model settings, and ten seeds. Their only difference is one review emphasis: balanced rubric, evidence grounding, diagnosis calibration, actionability, or communication quality. This isolates prompt emphasis rather than confounding it with payload ordering.
- **Response schema:** exact JSON with only `schema_version` and the three integer axis scores; there is no `verdict` or `hard_gates` field.
- **Git SHA:** `a8063d0c7ce462a4d55dfec96971176133041450` before the v4 working-tree protocol additions.
- **Configuration:** `configs/qwen36_native_fp8_vllm_essay_only_score5x10.v4.json` (SHA-256 `2e859d3ff825688b0314429847707be4d163e5223d181f29c00173042a515fc7`).
- **Model/runtime:** local `Qwen/Qwen3.6-35B-A3B-FP8`, revision `95a723d08a9490559dae23d0cff1d9466213d989`; vLLM 0.25.1; one DP=4, TP=1 endpoint on GPUs 0–3; 96 sequences per DP rank, 384 client requests in flight, 49,152 batch tokens per DP rank, 0.90 GPU-memory target, 4,096-token context, eager mode, and prefix caching.
- **Sampling:** five prompt versions × ten fixed seeds at temperature 0.15 = exactly 50 observations per rationale; train planned 300,000 calls, then separately frozen validation planned 60,000 calls.
- **Inputs:** same validated generated-candidate batch and isolated train/validation candidate artifacts as v3; validation cannot influence prompt choice, selection, SFT, DPO, or GRPO.
- **Commands:** `scripts/run_qwen36_native_fp8_vllm_essay_only_score5x10_v4_smoke.sh gpu0`, then `dp4`, then `scripts/run_qwen36_native_fp8_vllm_essay_only_score5x10_v4_full.sh`.
- **Static evidence:** `py_compile`, score-only v4 config contract, GPU-free distribution tests, shell syntax checks, and `git diff --check` passed. No raw text, prompts, identifiers, or completions are retained in this tracked record.
