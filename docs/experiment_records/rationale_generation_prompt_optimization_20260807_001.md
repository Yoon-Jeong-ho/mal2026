# Rationale generation prompt optimization (2026-08-07)

## Contract

- Run: `rationale-prompt-optimization-v1-20260807-001`
- Git SHA at execution: `243f18742b46adb551a0b4ffe31130eaf6d8e46d`
- Source: restricted train split, fixed 15-row sample; every score 1--5 occurs exactly three times in every axis.
- Sample SHA-256: `4bf235c8e0801ac0eeebdbc8223c00018ccd8783e7d98410ff52484ad446829c`
- Generation: `gpt-5.6-luna` and `gpt-5.6-terra`, Responses API, strict JSON schema, reasoning effort `none`, maximum 1,800 output tokens.
- Judge: Qwen3.6-35B-A3B Q4_K_M GGUF, SHA-256 `b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`.
- Judge prompt: exact `llm_as_judge.txt` bytes, SHA-256 `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`.
- Seed/decoding: Judge temperature `0.0`, top-p `1.0`, seed `42`; provider generation aliases used their fixed default sampling behavior.
- Hardware: four NVIDIA H100 80GB GPUs. Baseline used GPUs 4--7; optimization rounds used the user-authorized GPUs 0--3 after a real GPU0 smoke.
- Long Python runners set an explicit `setproctitle`.

The generator and judge both received the same Decimal `ROUND_HALF_UP` integerized human/reference score. The comparison therefore measures conditional rationale fidelity, not emitted-score prediction.

## Version preservation

The canonical `rationale_generation_prompt.txt` was not overwritten. Every candidate is a separate file.

| Version | Prompt SHA-256 | Luna | Terra | Two-model mean | Worst model |
|---|---|---:|---:|---:|---:|
| v0 | `dfc4b9091e3067732f52fab7ccd7c019373938e65e30e6fcc9123cd0656c9e14` | 4.9667 | 4.9722 | 4.9694 | 4.9667 |
| v1 | `46f373d904deab495f5cded89ce837722807e2e901e9aeefd1bc92fbe4708537` | 4.9611 | 4.8944 | 4.9278 | 4.8944 |
| v2 | `140d9e46ab97202db032d80f892785af00c8fb03b40593cdf773185eecc5d7c1` | 4.9556 | 4.9722 | 4.9639 | 4.9556 |
| v3 | `b71ee648b9a6707c1e0156681adb9c4d47a3a4a4b751aa2cb90d0bc8808981c6` | **4.9833** | **5.0000** | **4.9917** | **4.9833** |

Selection maximized the worse model first and the two-model mean second. `rationale_generation_prompt_v3.txt` won both criteria.

## Iteration evidence

- v1 over-suppressed visible defects at high reference scores. The judge penalized over-positive or factually broad claims.
- v2 acknowledged visible defects more directly, but this sometimes made a reference score of 5 look inconsistent with its rationale.
- v3 selected the two strongest observable grounds at a high reference score, avoided both global error-free claims and gratuitous minor-error inventories, and made organization evidence follow visible functional transitions rather than assumed paragraph boundaries.

Terra v3 received 5 on all 180 judge cells. Luna v3 received 177 fives and three fours; its remaining deductions were one organization-groundedness cell and two high-score consistency cells.

## Commands and artifacts

Generation used:

```text
.venv-standard/bin/python scripts/test_rationale_generation_prompt_openai.py {prepare,run,summarize} --run-id <generation-run> --prompt-file rationale_generation_prompt_vN.txt
```

Judge execution used:

```text
.venv-standard/bin/python scripts/run_balanced_rationale_q4_judge.py all --campaign <judge-run> --source-run <generation-run> --gpu-scope 0,1,2,3 --gpu-authorization <recorded-user-authorization>
```

Aggregate comparison: `outputs/rationale-prompt-optimization-v1/rationale-prompt-optimization-v1-20260807-001/comparison.json`.

## Limitations

- The same 15 train rows and same judge were reused across prompt selection rounds, so 4.9917 is an in-sample prompt-selection result and is likely optimistic.
- A perfect conditional Judge score does not establish score prediction, encoder RMSE, or SFT performance.
- Before using v3 for a full generation campaign, it should be checked once on a disjoint score-balanced train-only holdout without further prompt tuning.
