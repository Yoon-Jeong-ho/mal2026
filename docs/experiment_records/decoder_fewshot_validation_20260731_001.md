# Decoder few-shot validation — 2026-07-31-001

- **Status:** completed.
- **Run ID:** `decoder-fewshot-validation-v1-20260731-001`.
- **Question:** without training, does changing only the score distribution of
  five train-only demonstrations change decoder score concentration on the
  400-row validation split?
- **Privacy:** validation prompts, essays, identifiers, demonstrations, and
  model responses remain under ignored restricted/output roots. This record
  contains aggregate statistics only.

## Fixed protocol

- The scoring instruction is the user-supplied `evaluation.txt` verbatim
  (`SHA-256 1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145`).
- Each response is constrained to exact JSON containing integer scores 1--5
  and Korean rationales for `content`, `organization`, and `expression`.
  No `average` field is requested or evaluated.
- Demonstrations come only from canonical training data. Their score is the
  half-up integer form of the canonical score; their rationale comes from the
  score-blind evaluation-prompt rationale artifact
  (`SHA-256 d4a2be9a070c786728fde6f64f066ac9d462bc5f83305a2d9161b380abd88e55`).
  No validation label is used for prompt construction, shot selection, or
  ordering.
- Both arms use five shots. `balanced5` contains one example of each score
  1--5 on every axis. `central5` contains three score-3 and two score-4
  examples on every axis. Five deterministic order rotations are assigned to
  exactly 80 validation essays each, independently of labels.
- Decoding is deterministic (`temperature=0`, seed `2026073104`) with vLLM
  structured output, prefix caching, CUDA graphs, 12,288-token context,
  32,768 batched tokens, and up to 64 sequences. The initial output ceiling is
  512 tokens.
- Protocol/config evidence:
  `outputs/analysis/decoder-fewshot-validation-v1-20260731-001/protocol.json`,
  config SHA-256
  `07169afcb230939d4d54411e34cd05a748342e5da1d53e9a710bd993011305b9`,
  and restricted shot-manifest SHA-256
  `a9431b7b432526bf30659dd33d812f35c967286e0a107903a63b93eb70bfe111`.

## Models and runtime

| Model | Revision | GPU execution |
|---|---|---|
| `skt/A.X-4.0-Light` | `ba21c20ea1b31ded1ec3e2fb432335077dc4be98` | TP=1, GPU 0 |
| `K-intelligence/Midm-2.0-Base-Instruct` | `35479c5fc9a18a5db7cc6dbadcf1db68db7beab0` | TP=1, GPU 1 |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | TP=1, GPU 2 |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | TP=1, GPU 3 |
| `Qwen/Qwen3.6-35B-A3B-FP8` | `95a723d08a9490559dae23d0cff1d9466213d989` | TP=4, GPUs 0--3 |

- Existing `.venv-standard`: Python 3.12.3, PyTorch 2.11.0+cu130,
  Transformers 5.14.1, vLLM 0.25.1, CUDA 13.0; four NVIDIA H100 80 GB
  devices. The Qwen3-8B and Qwen3.5-9B snapshots were downloaded under the
  ignored project model cache after the user explicitly requested those
  models. No package or environment was created.
- Launch Git SHA: `a96f029b1ea640878607d4eb7bd817e6099334c1` with a dirty worktree,
  which was preserved.
- Runner template:
  `CUDA_VISIBLE_DEVICES=<scope> .venv-standard/bin/python
  scripts/run_decoder_fewshot_validation.py --config
  configs/decoder_fewshot_validation.v1.json --stage model --model-key <key>`.

## Results

`raw RMSE` compares integer decoder predictions with canonical continuous
validation scores. `int RMSE` compares with half-up integer gold. Spearman is
computed against the continuous gold. Score-3/4 rates are macro averages over
the three axes.

| Model | Shots | n | Raw RMSE | Spearman | Int RMSE | Score 3 | Score 4 | Parse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A.X-4.0-Light | balanced | 400 | 0.9997 | 0.2063 | 1.0196 | 22.50% | 60.00% | 100% |
| A.X-4.0-Light | central | 400 | 0.9122 | 0.1806 | 0.9555 | 53.50% | 42.08% | 100% |
| Midm-2.0-Base-Instruct | balanced | 399 | 0.9584 | 0.4034 | 0.9587 | 15.46% | 68.59% | 99.75% |
| Midm-2.0-Base-Instruct | central | 400 | 0.9056 | 0.3690 | 0.9152 | 23.92% | 67.08% | 100% |
| Qwen3-8B | balanced | 400 | 0.9732 | 0.3711 | 0.9762 | 20.75% | 60.33% | 100% |
| Qwen3-8B | central | 400 | 0.8750 | 0.3332 | 0.9178 | 48.00% | 49.17% | 100% |
| Qwen3.5-9B | balanced | 400 | 1.0118 | **0.4958** | 1.0104 | 20.75% | 49.92% | 100% |
| Qwen3.5-9B | central | 400 | **0.8215** | 0.4406 | **0.8667** | 46.83% | 49.17% | 100% |
| Qwen3.6-35B-A3B-FP8 | balanced | 400 | 0.9058 | 0.4789 | 0.9226 | 26.00% | 47.92% | 100% |
| Qwen3.6-35B-A3B-FP8 | central | 400 | 0.8376 | 0.4663 | 0.8777 | 36.75% | 44.58% | 100% |

The validation gold macro distribution is 37.17% score 3 and 43.67% score 4.
A constant-3 continuous-gold baseline has macro raw RMSE 0.8559; constant 4
has 0.9628. Thus only the central Qwen3.5 and Qwen3.6 arms beat constant 3,
and only by 0.0344 and 0.0183 RMSE respectively. The strongest rank signal is
instead the more widely distributed Qwen3.5 balanced arm, showing a material
calibration/ranking trade-off.

For all five models, balanced demonstrations reduce score-3 concentration;
the reduction ranges from 8.46 to 31.00 percentage points. They do not make
the prediction histogram balanced: much of the mass moves to score 4, and
raw RMSE worsens in every model. The effect is especially clear for Qwen3.5:
balanced shots improve exact accuracy on gold 1/2 axes from 9.6% to 20.0% and
on gold 5 axes from 0.0% to 51.4%, but worsen global raw RMSE from 0.8215 to
1.0118. Qwen3.6 is less prompt-sensitive and retains the strongest combined
tail behavior among the tested arms, but still has raw RMSE 0.9058 in the
balanced condition.

An explicitly post-hoc, label-free mean of the five central-arm model scores
has macro raw RMSE 0.7582 and Spearman 0.4849 on all 400 rows. This was not a
predeclared primary arm and is only diagnostic evidence that model diversity
reduces some variance; it is not a basis for validation-tuned selection.

## Negative results and integration recovery

- The first four single-GPU launches failed before smoke because FlashInfer
  JIT could not locate the already installed `.venv-standard/bin/ninja`.
  Failure logs were retained. Prepending the existing environment's `bin` to
  `PATH` repaired the integration without installing software or changing a
  scientific variable. All five two-request smoke gates then passed.
- Initial 512-token exact-JSON truncations occurred for Midm (2), Qwen3-8B
  (33), Qwen3.5-9B (4), and Qwen3.6 (3). Only those rows were deterministically
  regenerated with the same prompt, shots, order, schema, seed, and model.
  Qwen3-8B resolved at 1,024 tokens; Qwen3.5 and Qwen3.6 resolved at 2,048.
- One Midm balanced response entered a tab-whitespace loop and remained
  truncated at 2,048 tokens. The raw failed response is retained, is not
  coerced into a score, and is excluded from metrics; consequently that arm
  reports n=399 and 99.75% parse coverage.

## Interpretation

The central-score concentration is not unavoidable: decoder predictions move
substantially when the demonstration distribution changes. However, diverse
few-shot examples alone do not yield a competitive or calibrated scorer. They
mainly trade central accuracy for tail recall and remain far too sensitive to
shot priors. These outputs are useful as a disagreement/uncertainty signal or
as inputs to a calibration/routing model, but should not replace the trained
encoder scorer or become synthetic labels without an independently specified
filter and calibration protocol.

Aggregate result:
`outputs/analysis/decoder-fewshot-validation-v1-20260731-001/aggregate.json`
(`SHA-256 0b59157da22808e82c001f196260eafa8a29f592366557c619e65e35b5101eb8`).
The ignored runtime ledger and logs are under
`outputs/decoder-fewshot-validation-v1/decoder-fewshot-validation-v1-20260731-001/`.
