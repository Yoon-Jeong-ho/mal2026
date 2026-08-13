# Prompt experiment index

This index separates prompt source files from aggregate experiment records.
Restricted essays, demonstrations, human reasons, identifiers, requests, and
model responses are intentionally excluded from Git. Generated predictions and
run logs remain under ignored `data/processed/restricted/` and `outputs/`
locations.

## Decoder score prompts

| Study | Prompt conditions | Main aggregate result | Decision | Record |
|---|---|---|---|---|
| Official zero-shot baseline | Exact [`evaluation.txt`](../evaluation.txt) | Reused as the unchanged P0 baseline across decoder studies | Retain as baseline; not assumed optimal | [Human-audited comparison](experiment_records/decoder_human_audited_prompt_validation_20260809_001.md) |
| Local five-shot distribution | `balanced5`: one train demonstration per score; `central5`: three score-3 and two score-4 demonstrations | Five local models showed a ranking/calibration trade-off. Qwen3.5 central had raw RMSE 0.8215; Qwen3.5 balanced had Spearman 0.4958 but worse RMSE 1.0118 | Do not use few-shot scores as synthetic-label truth | [Record](experiment_records/decoder_fewshot_validation_20260731_001.md) |
| External five-shot distribution | The same fixed balanced and central rotations on Solar-Open2 INT4, GPT-5.6 Luna, and GPT-5.6 Terra | Solar central had the best individual raw RMSE, 0.7227; Terra central had Spearman 0.4971. All models remained weak on scores 1, 2, and 5 | Auxiliary or disagreement signal only | [Record](experiment_records/decoder_fewshot_external_20260731_001.md) |
| Explicit public score bands | Exact official P0 versus [`public_spec_score_band_prompt.v1.txt`](../configs/public_spec_score_band_prompt.v1.txt) | P1 improved Qwen raw RMSE 0.9948→0.9157 but worsened Solar 0.7646→0.8259 and collapsed Solar predictions toward score 3 | Model-specific mixed/negative result; not promoted | [Record](experiment_records/decoder_prompt_band_ablation_20260801_001.md) |
| Human-audited score rules | P0, public-band P1, and [`human_audited_score_prompt.v1.txt`](../configs/human_audited_score_prompt.v1.txt) | Human-audited P2 improved rank correlation but worsened raw RMSE to 0.9885 and emitted no exact score 1 or 5 | Negative result; do not replace P0 or P1 | [Record](experiment_records/decoder_human_audited_prompt_validation_20260809_001.md) |
| Solar train-only search | Eight tracked rule/few-shot prompt families under `configs/solar_prompt_search.v*.json` | The fixed train-only target RMSE below 0.4 was not reached | Completed negative search; no candidate promoted to validation | [Record](experiment_records/solar_prompt_search_20260801_001.md) |

### Reproduction entry points

- Local few-shot:
  [`configs/decoder_fewshot_validation.v1.json`](../configs/decoder_fewshot_validation.v1.json),
  [`scripts/run_decoder_fewshot_validation.py`](../scripts/run_decoder_fewshot_validation.py),
  and [`src/mal2026/decoder_fewshot_validation.py`](../src/mal2026/decoder_fewshot_validation.py).
- External few-shot:
  [`configs/decoder_fewshot_external.v1.json`](../configs/decoder_fewshot_external.v1.json),
  [`scripts/run_decoder_fewshot_external.py`](../scripts/run_decoder_fewshot_external.py),
  and [`src/mal2026/decoder_fewshot_external.py`](../src/mal2026/decoder_fewshot_external.py).
- Explicit band ablation:
  [`configs/decoder_prompt_band_ablation.v1.json`](../configs/decoder_prompt_band_ablation.v1.json),
  [`scripts/run_decoder_prompt_band_ablation.py`](../scripts/run_decoder_prompt_band_ablation.py),
  and [`src/mal2026/decoder_prompt_band_ablation.py`](../src/mal2026/decoder_prompt_band_ablation.py).
- Human-audited ablation:
  [`configs/decoder_human_audited_prompt_validation.v1.json`](../configs/decoder_human_audited_prompt_validation.v1.json)
  and [`scripts/run_decoder_human_audited_prompt_validation.py`](../scripts/run_decoder_human_audited_prompt_validation.py).

## Rationale prompts

| Study or contract | Prompt files | Main result or role | Record |
|---|---|---|---|
| Score-conditioned rationale prompt optimization | [`rationale_generation_prompt_v0.txt`](../rationale_generation_prompt_v0.txt) through [`rationale_generation_prompt_v3.txt`](../rationale_generation_prompt_v3.txt) | On the fixed 15-row train-only selection sample, v3 had the best worst-model judge mean, 4.9833, and two-model mean, 4.9917. This is an in-sample conditional-rationale result, not score prediction evidence | [Record](experiment_records/rationale_generation_prompt_optimization_20260807_001.md) |
| Score-blind rationale generation | [`Rationale_evaluation_training.txt`](../Rationale_evaluation_training.txt) | Generates grounded content, organization, and expression rationales without emitting scores | [End-to-end re-audit](experiment_records/evaluation_prompt_end_to_end_reaudit_20260729_001.md) |
| Rationale-aware score input | [`rationale_to_score.txt`](../rationale_to_score.txt) | Defines the encoder input contract that checks rationales against the essay rather than treating them as labels | [Public-spec program](experiment_records/official_prompt_alignment_v1_20260727_001.md) |
| Score-blind/conditioned SFT matrix | `configs/evaluation_prompt_rationale_sft.*.json` | Preserves separate score-blind and score-conditioned training/evaluation arms | [End-to-end re-audit](experiment_records/evaluation_prompt_end_to_end_reaudit_20260729_001.md) |
| RLAIF prompt ensemble | `configs/rlaif_grpo_prompt_ensemble.v1.json` through `v8.json` | Eight tracked reward/prompt variants with individual aggregate records | [`v1` record](experiment_records/rlaif_grpo_prompt_ensemble_v1_20260722_009.md) through [`v8` record](experiment_records/rlaif_grpo_prompt_ensemble_v8_20260722_022.md) |

The canonical score-conditioned rationale file
[`rationale_generation_prompt.txt`](../rationale_generation_prompt.txt) remains
byte-identical to v0. Candidate variants are preserved separately rather than
overwriting that baseline.

## Human-review-derived augmentation guidance

The aggregate 1/2/5 feature proposal and adjacent-band generation contract are
recorded in
[`humaneval/records/score_augmentation_features_20260813.md`](../humaneval/records/score_augmentation_features_20260813.md).
It is an exploratory generation specification, not a validated labeling rule.

## Interpretation boundary

- Validation results in these records are descriptive because the fixed split
  has been observed repeatedly; they must not be used for further unrecorded
  prompt retuning.
- A prompt-selected score is not a synthetic ground-truth label.
- Tail behavior at scores 1, 2, and 5 remains the main decoder weakness.
- Prompt variants and negative results are retained so that future work does
  not repeat failed or model-specific approaches.
