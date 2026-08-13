# Decoder score-band prompt ablation — 2026-08-01-001

- **Status:** completed; mixed/negative result, not promoted as the default.
- **Run ID:** `decoder-prompt-band-ablation-v1-20260801-001`.
- **Question:** does making the observed meaning of scores 1--5 explicit improve
  zero-shot decoder scoring relative to the user-supplied official prompt?
- **Privacy:** individual essays, identifiers, prompts containing essays, and
  model responses remain under ignored restricted/output roots. This record
  contains aggregate statistics only.

## Train-only score-band analysis

The analysis read all 2,000 canonical training rows and zero validation rows.
Continuous axis labels were converted to descriptive integer bands with
`Decimal(str(score)).quantize(1, ROUND_HALF_UP)`.

| Axis | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 |
|---|---:|---:|---:|---:|---:|
| Content | 11 | 202 | 982 | 749 | 56 |
| Organization | 26 | 252 | 708 | 803 | 211 |
| Expression | 4 | 77 | 533 | 1,093 | 293 |

Higher bands were somewhat longer and contained somewhat more enumeration,
evidence, and transition markers, but the univariate Spearman associations
with raw scores were only about 0.1--0.3. These are descriptive correlations,
not causal or usable scoring thresholds. Physical paragraph count was also
unusable because 1,994/2,000 serialized essays were single-line. Accordingly,
the revised prompt explicitly forbids length, marker counts, and physical line
count as mechanical rules and defines organization through discourse function.
Rare score-1 bands, especially expression (`n=4`), make their aggregate profile
unstable.

Reproducible aggregate analysis:

- `src/mal2026/train_score_band_profile.py`
- `scripts/analyze_train_score_bands.py`
- `tests/test_train_score_band_profile.py`
- `data/reports/train_score_band_profile_v1.json`
  (`SHA-256 d2094de208dc8142be3d5a79e58f0bac2e1350c4e2f05bfa212643a507ca3804`)
- canonical train input SHA-256
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.

## Prompt and fixed protocol

- `official_p0` is the exact user-supplied `evaluation.txt`
  (`SHA-256 1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145`).
  It was not overwritten.
- `axis_band_p1` is a new, zero-shot prompt derived from the public task
  specification and aggregate training evidence. It is **not organizer-authored**.
  It defines separate 1--5 anchors for content (task response, claim, evidence,
  logical connection), organization (discourse function and development), and
  expression (clarity, vocabulary, and language conventions). It requires an
  adjacent-band check, forbids matching the dataset score distribution or using
  3 as a default, and requests strict score-plus-rationale JSON with no average.
- No individual labeled demonstrations were included in either arm. This avoids
  the score-distribution prior previously observed with balanced and central
  five-shot prompts.
- The same two arms were run on a deterministic 400-row train probe and all 400
  validation rows. The train probe is descriptive rather than an independent
  prompt-selection set because the prompt was informed by aggregate full-train
  analysis. Validation had been observed by earlier experiments, so validation
  results are locked/descriptive rather than an unbiased model-selection result.
- Both arms used the same essay, model, seed (`2026080101`), integer 1--5 JSON
  schema, `temperature=0`, and 512-token initial output ceiling. Only responses
  truncated by length were retried at 2,048 tokens.
- Primary metrics were macro raw RMSE against continuous axis labels and macro
  Spearman. Secondary diagnostics included half-up integer RMSE, score
  histograms, exact tail recall, triplet accuracy, prompt flip/direction, and a
  paired essay bootstrap with 10,000 replicates.

Prompt/config evidence:

- `configs/public_spec_score_band_prompt.v1.txt`
  (`SHA-256 1c126b26b4ff8b99a6dd4d14235a9e8eaa07b7e9f3d8f28040e2fdccd35a4e4e`)
- `configs/decoder_prompt_band_ablation.v1.json`
  (`SHA-256 f6d9efa80df2e58d81d9de919dce085c49257fec5fd234ab8066b562cbaa799e`)
- restricted train-probe manifest SHA-256
  `9e097e12b66bdeda193d09163a905dc1759b99f01168708d22db18128d838b06`.

## Models and runtime

| Model | Revision/runtime | GPU execution |
|---|---|---|
| `Qwen/Qwen3.6-35B-A3B-FP8` | revision `95a723d08a9490559dae23d0cff1d9466213d989`; local vLLM 0.25.1 | TP=4, GPUs 0--3 |
| `nota-ai/Solar-Open2-250B-Nota-INT4` | official image `upstage/vllm-solar-open2@sha256:b34b7dd42d986bafb8dcfb285758de2fbfd55e7cf28a1b5ba6b858e6b2b7c8c3`; vLLM 0.22.0 | TP=4 + EP, GPUs 0--3 |

Both used CUDA graphs (`enforce_eager=False`), prefix caching, chunked prefill,
12,288-token context, 32,768 batched tokens, up to 64 sequences, and 0.9 GPU
memory utilization. The Qwen run issued 1,600 full requests, made 203
length-only retries, and retained one unparseable train-probe P0 response. The
Solar run issued 1,600 full requests plus two smoke and 14 internal retry
requests; it used 2,190,860 prompt and 515,805 completion tokens and retained
one unparseable validation P1 response.

Launch Git SHA was `a96f029b1ea640878607d4eb7bd817e6099334c1`; the pre-existing dirty worktree
was preserved. The existing `.venv-standard` and runner were used; no package
or environment was created. Runner commands were:

```text
PYTHONPATH=src .venv-standard/bin/python scripts/run_decoder_prompt_band_ablation.py --config configs/decoder_prompt_band_ablation.v1.json --stage prepare
PATH="$PWD/.venv-standard/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src .venv-standard/bin/python scripts/run_decoder_prompt_band_ablation.py --config configs/decoder_prompt_band_ablation.v1.json --stage qwen-run
PATH="$PWD/.venv-standard/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src .venv-standard/bin/python scripts/run_decoder_prompt_band_ablation.py --config configs/decoder_prompt_band_ablation.v1.json --stage solar-run
PYTHONPATH=src .venv-standard/bin/python scripts/run_decoder_prompt_band_ablation.py --config configs/decoder_prompt_band_ablation.v1.json --stage aggregate
```

## Aggregate results

`raw RMSE` compares integer decoder predictions with continuous canonical
scores. `int RMSE` compares them with half-up integer gold. Score-3 and score-4
rates are macro averages over the three axes.

### Train probe (descriptive)

| Model | Prompt | n | Raw RMSE | Spearman | Int RMSE | Score 3 | Score 4 | Triplet acc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B | P0 official | 399 | 1.0155 | 0.4841 | 1.1238 | 46.95% | 16.71% | 3.01% |
| Qwen3.6-35B | P1 bands | 400 | **0.8873** | **0.5102** | **0.9915** | 44.33% | 26.83% | 9.00% |
| Solar-Open2-250B | P0 official | 400 | **0.7667** | **0.4778** | **0.8572** | 59.42% | 29.83% | 12.50% |
| Solar-Open2-250B | P1 bands | 400 | 0.8422 | 0.3780 | 0.9496 | 79.92% | 8.33% | 12.50% |

### Validation (locked/descriptive)

| Model | Prompt | n | Raw RMSE | Spearman | Int RMSE | Score 3 | Score 4 | Triplet acc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35B | P0 official | 400 | 0.9948 | **0.4931** | 1.0848 | 46.83% | 17.33% | 3.50% |
| Qwen3.6-35B | P1 bands | 400 | **0.9157** | 0.4855 | **1.0108** | 45.17% | 24.33% | 6.75% |
| Solar-Open2-250B | P0 official | 400 | **0.7646** | **0.4650** | **0.8491** | 59.33% | 31.08% | 11.25% |
| Solar-Open2-250B | P1 bands | 399 | 0.8259 | 0.3930 | 0.9287 | 79.37% | 8.69% | 10.28% |

Paired validation bootstrap estimates for P1 minus P0 were:

- Qwen raw-RMSE delta `-0.0790`, 95% interval
  `[-0.1089, -0.0497]`; Spearman delta `-0.0074`, interval
  `[-0.0444, 0.0293]`.
- Solar raw-RMSE delta `+0.0635`, 95% interval `[0.0294, 0.0968]`;
  Spearman delta `-0.0745`, interval `[-0.1313, -0.0205]`.

For Qwen, validation axis RMSE changed from P0 to P1 as follows: content
`0.7933 -> 0.7980`, organization `0.8264 -> 0.8620`, and expression
`1.3646 -> 1.0872`. The aggregate gain therefore came almost entirely from
repairing severe expression underscoring, while content/organization worsened
slightly and rank correlation did not improve.

For Solar, every validation axis worsened: content `0.6347 -> 0.7161`,
organization `0.8279 -> 0.8859`, and expression `0.8311 -> 0.8758`. P1 caused
a strong collapse toward score 3.

Pooled validation exact recall by rounded gold band also shows that neither
prompt solved the extremes. Qwen P1 improved score-2 recall from 53.4% to
60.3% and score-4 recall from 23.7% to 29.8%, but score-5 recall fell from
3.8% to 0%. Solar P1 improved score-2 recall from 25.0% to 39.5%, but score-4
recall fell from 41.6% to 10.9% and score-5 recall from 1.0% to 0%.

## Decision and limitations

The explicit band prompt is **not a universal improvement**. Identical wording
moved Qwen scores upward enough to reduce expression RMSE, but made Solar much
more conservative and concentrated around score 3. The direction repeated on
the train probe and validation, indicating model-specific prompt response, but
the validation result remains descriptive because it has previously been
observed.

Therefore:

1. Do not replace `evaluation.txt` globally.
2. Do not use P1 as the Solar synthetic-label or augmentation judge.
3. If prompt specialization is pursued, choose and calibrate it using
   train-only out-of-fold predictions; use a genuinely untouched set only once
   for final confirmation.
4. Decoder scores still lack reliable score-1/5 behavior. Until independently
   calibrated, they are safer as rationale, pairwise preference, disagreement,
   or uncertainty signals than as absolute synthetic labels.
5. The earlier labeled central five-shot arms remain better in absolute RMSE
   (Solar `0.7227`, Qwen `0.8376`) but carry a demonstrated prompt-prior and
   tail-calibration trade-off; they are not label truth either.

Aggregate evidence:

- combined result:
  `outputs/analysis/decoder-prompt-band-ablation-v1-20260801-001/aggregate.json`
  (`SHA-256 5011b929ad292a4623d40fca842d61f7ae5eb1b3f786d83e654990b1a9916921`)
- Qwen result SHA-256
  `65e87c3f006b1023ec81182e0df72e5b83d611794d7a1f4b308261248c53cbc9`
- Solar result SHA-256
  `04113f7dae0b9a1392f253c3432d5472e4895c03b6873337c123faf7afb72c2e`
- ignored runtime ledger and logs:
  `outputs/decoder-prompt-band-ablation-v1/decoder-prompt-band-ablation-v1-20260801-001/`.
