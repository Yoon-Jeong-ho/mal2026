# Rationale v3 dual-teacher tail generation (2026-08-07)

## Status and decision

- **Status:** completed
- **Decision:** the frozen-v3 pilot passed its preregistered aggregate gates, so the
  same prompt and protocol were expanded to the complete train and validation
  splits.
- **Purpose:** create score-free three-axis rationale targets with both
  `gpt-5.6-luna` and `gpt-5.6-terra`, while increasing target diversity for rows
  containing rare integer score bands 1, 2, or 5.
- **Authorization:** the user approved external OpenAI API generation with both
  teachers and exact-Q4 judging on GPUs 0--3. Validation was generated only after
  the train pilot passed and was not used to choose the prompt or teacher.

## Frozen contract

- Generation prompt: `rationale_generation_prompt_v3.txt`
  (`b71ee648b9a6707c1e0156681adb9c4d47a3a4a4b751aa2cb90d0bc8808981c6`)
- Judge prompt: `llm_as_judge.txt`
  (`91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`)
- Judge model: Qwen3.6-35B-A3B Q4_K_M GGUF
  (`b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`)
- Score conditioning: each human/reference axis score was converted with Decimal
  `ROUND_HALF_UP` to an integer in 1--5 and supplied only to the teacher.
- Student target: exactly three rationale strings; no score appears in the SFT
  target file. Labels and judge diagnostics are isolated in restricted provenance.
- Row multiplicity, per teacher:
  - any axis at score 1: four variants;
  - otherwise any axis at score 2 or 5: two variants;
  - otherwise: one variant.
- Candidate variants changed only evidence-selection emphasis. They did not change
  the score, rubric, source text, output schema, or frozen system prompt.

Canonical source hashes:

| Split | SHA-256 |
|---|---|
| train | `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737` |
| validation | `0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4` |

## Generation result

| Split / teacher | Requests | Accepted | Excluded after one repair |
|---|---:|---:|---:|
| train / Luna | 2,856 | 2,856 | 0 |
| train / Terra | 2,856 | 2,853 | 3 |
| validation / Luna | 576 | 576 | 0 |
| validation / Terra | 576 | 575 | 1 |
| **Total** | **6,864** | **6,860** | **4** |

The initial validator rejected 409 surface-form outputs for non-verbatim quotation,
foreign-script contamination, or score-reference leakage. One unchanged-rubric
direct repair recovered 405. The four remaining invalid outputs were preserved as
negative artifacts and excluded. All accepted full-run candidates contain no
detected foreign script or score leak. No exact rationale duplicates were found.

Source-row multiplicities were 1,214/751/35 rows at 1/2/4 candidates per teacher
for train and 240/152/8 for validation. The resulting candidate-level band counts
show the intended tail expansion:

| Split | Axis | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 |
|---|---|---:|---:|---:|---:|---:|
| train | content | 88 | 860 | 2,460 | 2,077 | 224 |
| train | organization | 208 | 1,043 | 1,680 | 1,935 | 843 |
| train | expression | 32 | 356 | 1,495 | 2,650 | 1,176 |
| validation | content | 16 | 196 | 497 | 398 | 44 |
| validation | organization | 40 | 236 | 333 | 378 | 164 |
| validation | expression | 16 | 72 | 306 | 545 | 212 |

Mean within-row trigram similarity was 0.2132/0.2202 for train Luna/Terra and
0.2166/0.2306 for validation Luna/Terra, indicating that the added variants were
not near-duplicate copies under this diagnostic.

## Exact-Q4 judge result

All 6,860 accepted candidates were judged in one four-replica GPU0--3 campaign.
The model stayed loaded across all 16 participant groups. Peak utilization was
100% on every GPU and peak memory was 22,659 MiB per GPU.

| Teacher | Candidates | Macro | Consistency | Groundedness | Score-1/2 cell rate | Gate |
|---|---:|---:|---:|---:|---:|---|
| Luna | 3,432 | 4.9533 | 4.8876 | 4.9877 | 0.2380% | pass |
| Terra | 3,428 | 4.9741 | 4.9454 | 4.9885 | 0.1167% | pass |

Reference-band macro means were:

| Teacher | Band 1 | Band 2 | Band 3 | Band 4 | Band 5 |
|---|---:|---:|---:|---:|---:|
| Luna | 4.9363 | 4.9644 | 4.9756 | 4.9671 | 4.8465 |
| Terra | 4.9675 | 4.9768 | 4.9872 | 4.9831 | 4.9121 |

The preregistered phrase “absent or very rare” for judge scores 1/2 was
operationalized as at most 0.5% of judge cells before re-aggregation; both teachers
passed. Low `score_rationale_consistency` cells remain diagnostic rather than a
hard deletion rule because manual pilot audit showed that the judge sometimes
re-graded the supplied human/reference score, especially for score 5.

## SFT handoff

Run `rationale-v3-tail-sft-20260807-001` produced two non-destructive choices:

| Split | Mechanically valid | Quality-filtered |
|---|---:|---:|
| train | 5,709 | 5,673 |
| validation | 1,151 | 1,148 |

- Quality tiers over all candidates: 6,709 clean, 112 review, and 39 severe
  non-consistency-dimension issues.
- The quality-filtered view removes only exact within-source duplicates and a
  candidate with any Q4 `domain_match`, `groundedness`, or `specificity` score at
  most 2. It does **not** filter on score-rationale consistency alone.
- No exact duplicate was removed; the 39 severe candidates account for the entire
  difference between the valid and quality-filtered views.
- Quality-filtered train teacher counts remain balanced: 2,836 Luna and 2,837
  Terra.
- Target rows contain only `candidate_key`, restricted `source_id`, and the three
  rationales. Integer scores and judge scores occur only in the separate restricted
  provenance files.

Restricted handoff:
`data/processed/restricted/rationale_v3_tail_sft/rationale-v3-tail-sft-20260807-001/`

Aggregate handoff:
`outputs/rationale-v3-tail-sft/rationale-v3-tail-sft-20260807-001/aggregate.json`

## Reproducibility

- Repository Git SHA at execution: `243f18742b46adb551a0b4ffe31130eaf6d8e46d`
- Environment: `.venv-standard`, Python 3.12.3
- Hardware: four NVIDIA H100 80GB HBM3 GPUs, driver 580.105.08
- Long-running Python processes used `setproctitle`.
- Runner hashes:
  - generation: `df065f563acd988ea8ed634fe98882794492d884d816906d3fc4d7efc36a4d47`
  - Q4 judge campaign: `abd20f578e4af9e758ae0c54fad82e6961a8e0c82ea9d637f35701a365e77552`
  - SFT assembly: `8094f8046ddb87e371921b56ad40f9fad2d8f8d67c797c8d486fe0f6441950fa`

Run IDs:

- `rationale-v3-tail-train-luna-20260807-001`
- `rationale-v3-tail-train-terra-20260807-001`
- `rationale-v3-tail-validation-luna-20260807-001`
- `rationale-v3-tail-validation-terra-20260807-001`
- `rationale-v3-tail-full-q4-judge-20260807-001`
- `rationale-v3-tail-sft-20260807-001`

Representative commands (the model and run ID were changed for each of the four
generation runs):

```bash
.venv-standard/bin/python scripts/generate_v3_tail_rationale_batch.py prepare \
  --run-id rationale-v3-tail-train-luna-20260807-001 \
  --model gpt-5.6-luna --scope train
.venv-standard/bin/python scripts/generate_v3_tail_rationale_batch.py smoke \
  --run-id rationale-v3-tail-train-luna-20260807-001 --model gpt-5.6-luna
.venv-standard/bin/python scripts/generate_v3_tail_rationale_batch.py submit \
  --run-id rationale-v3-tail-train-luna-20260807-001 --model gpt-5.6-luna
.venv-standard/bin/python scripts/generate_v3_tail_rationale_batch.py poll \
  --run-id rationale-v3-tail-train-luna-20260807-001 --model gpt-5.6-luna
.venv-standard/bin/python scripts/generate_v3_tail_rationale_batch.py download \
  --run-id rationale-v3-tail-train-luna-20260807-001 --model gpt-5.6-luna
.venv-standard/bin/python scripts/generate_v3_tail_rationale_batch.py retry-direct \
  --run-id rationale-v3-tail-train-luna-20260807-001 --model gpt-5.6-luna
.venv-standard/bin/python scripts/generate_v3_tail_rationale_batch.py analyze \
  --run-id rationale-v3-tail-train-luna-20260807-001 --model gpt-5.6-luna
```

The exact multi-run judge and assembly arguments are recorded in their restricted
manifests. No API key, prompt/essay text, rationale, source identifier, or judge
evidence is included in this record.

## Limitations

- The Q4 score measures conditional fidelity between a supplied integer human
  score and a rationale. It does not measure deployment-time score prediction.
- High teacher/judge agreement does not establish downstream SFT improvement.
  The valid and quality-filtered train views should therefore be compared under a
  separately frozen training/evaluation protocol.
- Generated validation targets are descriptive evaluation references only and
  must not be used for prompt, teacher, checkpoint, or hyperparameter selection.
