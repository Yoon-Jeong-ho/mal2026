# Evaluation-prompt end-to-end re-audit (2026-07-29/30, run 001)

## Scope, authority, and outcome

The user authorized a clean re-audit and rerun of rationale generation, score
prediction, and LLM-as-judge evaluation after pausing Solar augmentation and
removing its partial Docker download.  The fixed physical GPU scope was 0--3.
The repository `.venv-standard` and existing local model snapshots were used;
no package installation, new environment, paid API request, or new dataset
download occurred.

Execution-base Git SHA: `6a1f889`.  The experiment was executed from an
intentionally dirty working tree; the implementation and aggregate record are
committed after the terminal audit as
`91e6fdf45520537f6631252bc03dde4332f530e5`.  Restricted inputs, generated
rationales, predictions, checkpoints, judge records, and logs remain ignored.

All declared arms completed.  The best frozen 400-row validation score result
was **Qwen3-Embedding-8B with matched score-blind rationales**, at three-axis
macro continuous RMSE `0.5836757263` and macro Spearman `0.6193649427`.
Score-conditioned rationales made score prediction worse for both Qwen and
KURE.  The exact Q4 judge was saturated: every arm received macro
`4.9646`--`4.9796`, and `97.02%`--`98.38%` of its 4,800 cells were 5.  It is
therefore not a useful selector among these arms.

## Canonical inputs and prompt contracts

- `evaluation.txt` SHA-256:
  `1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145`
- `llm_as_judge.txt` SHA-256:
  `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`
- restricted MAL train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`
- restricted MAL validation SHA-256:
  `0805445029328848164cf15f34b90b88fb5f7896d7b73f24f1717b733a9a00a4`

`src/mal2026/evaluation_prompt_matrix.py` retains the exact
`evaluation.txt` rubric prefix and adds only task-specific input/output
clauses.  Four disjoint contracts are hash-bound:

1. `evaluation_txt_rationale_score_blind_v2`: prompt + essay -> three
   rationales, without a score payload;
2. `evaluation_txt_rationale_score_conditioned_v2`: actual emitted integer
   predictions + prompt + essay -> three rationales;
3. `evaluation_txt_score_only_v1`: prompt + essay -> three scores;
4. `evaluation_txt_score_rationale_aware_v1`: prompt + essay + three
   rationales -> three scores, with the essay declared authoritative.

No path reads, trains, predicts, or evaluates `score.average`.  The reported
macro RMSE is the arithmetic mean of the three independently evaluated axis
RMSE values, not an average-score target.  Score-conditioned rationale
generation is bound to actual Qwen direct predictions, never human or
reference scores.

The exact-judge request uses the byte-exact contents of `llm_as_judge.txt` as
the system message.  Its user message contains `[prompt_text]`,
`[essay_text]`, and the candidate's three emitted integer predictions plus
three rationales.  The judge receives no human/reference score.  The supplied
judge prompt explicitly says not to re-grade predicted-score correctness; it
evaluates domain match, score/rationale consistency, specificity, and
groundedness.  Consequently its result is a rationale-quality proxy, not a
score-RMSE estimate.

## Audit findings and repairs

1. The previous final rationale bundle was score-conditioned and did not use
   reference scores, but its prompt was a compact reconstructed rubric rather
   than a minimal `evaluation.txt` derivation.
2. The retained 6,000 OpenAI SFT completion targets were generated earlier
   with a semantically close public-spec prompt, not byte-identical
   `evaluation.txt`.  They were reused to avoid an unapproved external cost,
   while the SFT input side was rebuilt with the new prompt contracts.  This
   limitation is not represented as exact-prompt API regeneration.
3. Prior rationale-aware Qwen/KURE score inputs used a manually derived
   instruction.  The replacement runs use the exact-rubric router and start
   from completed AI-Hub full-parameter warmstarts, then fit a fresh MAL LoRA.
4. The judge launcher previously omitted `judge_prompt_sha256` from its newly
   written server attestation.  It now binds the exact prompt file hash, model
   SHA, llama.cpp revision, endpoints, and physical GPU scope.
5. A full-population token audit caught Qwen rationale-aware truncation before
   any optimizer update: at the old 2,304-token limit, score-blind data had 52
   train and 8 validation overflows, while score-conditioned data had 56 and
   20.  The failed preflight is preserved.  Qwen was raised to 2,560 tokens;
   observed maxima were 2,465/2,416 for blind and 2,514/2,493 for conditioned.
   KURE maxima remained below its fixed 2,048-token limit.

## Docker and augmentation boundary

The incomplete `upstage/vllm-solar-open2:latest` pull was canceled, the image
was removed, and unreferenced partial layers were pruned.  Root free space
recovered from approximately 16 GiB to 69 GiB.  Solar augmentation remains
paused.  Hugging Face, vLLM, and model caches are located under `/dataset`.
Docker image layers do not follow the shell working directory and the
daemon's root-backed `DockerRootDir` was deliberately not reconfigured or
restarted, so no further Docker pull was attempted.

## Rationale SFT and generation

Both SFT arms use `skt/A.X-4.0-Light` revision
`ba21c20ea1b31ded1ec3e2fb432335077dc4be98`, seed `2026072904`, 6,000 retained
completion targets, two epochs, LoRA rank 32, BF16 compute, DDP4, and no token
truncation.  The conditioned SFT inputs use each retained API candidate's own
emitted integer predictions; full MAL inference instead uses the predeclared
Qwen direct predictions described below.

| SFT arm | Train loss | Runtime (s) | Max tokens | Global steps |
|---|---:|---:|---:|---:|
| score-blind V2 | 1.5649732427 | 1273.3910 | 1644 | 188 |
| score-conditioned V2 | 1.5546426925 | 1356.3371 | 1728 | 188 |

The full vLLM generator used one tensor-parallel replica across physical GPUs
0--3 (`tensor_parallel_size=4`).  Both accepted handoffs contain exactly
2,000 train and 400 validation bundles, three rationales per bundle, zero
missing rows, zero reference scores, and zero final schema/quality failures.

| Rationale arm | Canonical run | Train retries / axis fallback | Val retries / axis fallback | Train SHA-256 | Validation SHA-256 |
|---|---|---:|---:|---|---|
| score-blind | `...score-blind-20260729-004` | 6 / 1 | 1 / 0 | `d4a2be9a...8e55` | `7cc0ba94...1377e` |
| score-conditioned | `...score-conditioned-20260729-002` | 14 / 0 | 7 / 0 | `d0633e6a...70da` | `5c3e730a...beac1` |

The conditioned handoff is cryptographically bound to the predeclared Qwen
direct score result SHA-256
`88fe8821ac87f0b741ba5d43bd7a106e701367282cf5d9870692e254e926924e`.

### Preserved generation negatives

- V1 run 001 placed four expression rationales at the 512-character schema
  ceiling and ended them mid-sentence.
- Raising only the cap did not solve the behavior: V1 run 002 still produced
  three cut-off rationales at 1,024 characters.
- The V2 contract therefore requires 60--420 Korean characters, 1--4 complete
  sentences, no repetition, and terminal punctuation.
- V2 blind run 001 produced 1,996/2,000 train and 398/400 validation valid
  bundles.  Runs 002 and 003 each isolated one persistent train failure after
  bounded retries.  Run 004 recovered only the failing expression axis and
  passed the full population.
- Conditioned run 001 produced 1,999/2,000 train and 400/400 validation valid
  bundles.  A versioned bounded axis-retry policy was declared before fresh
  run 002, which passed all rows.

## Score encoder protocol and results

Models:

- `Qwen/Qwen3-Embedding-8B`, revision
  `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`;
- `nlpai-lab/KURE-v1`, revision
  `d14c8a9423946e268a0c9952fecf3a7aabd73bd9`.

Each arm selects an epoch only on a deterministic 1,600/400 split of MAL
train, reloads the same AI-Hub full-parameter warmstart, refits on all 2,000
MAL train rows for the selected number of epochs, and evaluates the frozen
400-row validation once.  Training uses DDP4 on physical GPUs 0--3.  The
validation metrics below are continuous predictions before integer emission.

| Model / input | Selected epoch | Content RMSE | Organization RMSE | Expression RMSE | Macro RMSE | Macro Spearman | Shuffled-rationale RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen direct | 3 | 0.528946 | 0.719687 | 0.534393 | 0.594342 | 0.588939 | n/a |
| Qwen + score-blind rationale | 4 | **0.516236** | **0.706146** | **0.528646** | **0.583676** | **0.619365** | 0.680332 |
| Qwen + score-conditioned rationale | 3 | 0.535829 | 0.725798 | 0.532903 | 0.598177 | 0.578480 | 0.792773 |
| KURE direct | 5 | 0.585031 | 0.784016 | 0.556521 | 0.641856 | 0.479788 | n/a |
| KURE + score-blind rationale | 7 | 0.581860 | 0.778602 | 0.553991 | 0.638151 | 0.519231 | 0.623559 |
| KURE + score-conditioned rationale | 4 | 0.600922 | 0.804811 | 0.572961 | 0.659565 | 0.456694 | 0.644499 |

Qwen score-blind rationale-aware improves macro RMSE by `0.010666` (1.79%)
over Qwen direct.  Its matched rationale also beats the single shuffled
diagnostic by `0.096657` RMSE, evidence that this Qwen arm uses essay-matched
rationales.  KURE blind improves only `0.003705`; its one shuffled diagnostic
is actually lower than the matched result, so no beneficial rationale use can
be claimed for KURE.  Conditioned rationales worsen Qwen by `0.003835` and
KURE by `0.017709` versus their direct baselines, consistent with score
anchoring/circularity rather than new essay evidence.  None approaches the
requested `0.421300` RMSE.

## Exact `llm_as_judge.txt` Q4 evaluation

Runtime provenance:

- model: Qwen3.6-35B-A3B Q4_K_M GGUF;
- model SHA-256:
  `b46fedd33e0bfb0cae308aa3c158d0a4b2c4a1d2185a1ed6f093cdaf39064772`;
- llama.cpp revision/tag:
  `571d0d540df04f25298d0e159e520d9fc62ed121` / `b10068`;
- four independent replicas on physical GPUs 0--3, four slots each,
  `max_inflight=16`, temperature 0, seed 42;
- 400 validation records and 4,800 valid judge cells per arm, with zero
  transport/schema failures and no human/reference scores.

| Participant arm | Judge macro | Worst cell | Score 1--2 rate | Score 5 rate |
|---|---:|---:|---:|---:|
| Qwen direct + blind rationale | 4.971458 | 4.9175 | 0.1250% | 97.5833% |
| Qwen direct + conditioned rationale | **4.979583** | 4.9225 | 0.1250% | **98.3750%** |
| Qwen rationale-aware blind + blind rationale | 4.971875 | 4.9175 | 0.1042% | 97.5833% |
| Qwen rationale-aware conditioned + conditioned rationale | 4.975625 | **4.9425** | 0.1250% | 97.9583% |
| KURE direct + blind rationale | 4.969375 | 4.9100 | 0.1250% | 97.4583% |
| KURE rationale-aware blind + blind rationale | 4.964583 | 4.8375 | 0.1875% | 97.0208% |
| KURE rationale-aware conditioned + conditioned rationale | 4.972083 | 4.8750 | 0.1667% | 97.7083% |

Paired 10,000-replicate record bootstrap comparisons (seed `20260730`) all
include zero:

| Left minus right | Mean delta | 95% bootstrap CI |
|---|---:|---:|
| Qwen direct conditioned minus Qwen direct blind | +0.008125 | [-0.002083, +0.018333] |
| Qwen rationale-aware blind minus Qwen direct blind | +0.000417 | [-0.008333, +0.009375] |
| Qwen rationale-aware conditioned minus Qwen direct conditioned | -0.003958 | [-0.013750, +0.006042] |
| KURE rationale-aware blind minus KURE direct blind | -0.004792 | [-0.015417, +0.006667] |

The reproducible aggregate-only analysis is written to ignored output
`outputs/evaluation-prompt-exact-judge-matrix-v1/aggregate_analysis.json`
(SHA-256
`913e9eea14f7c83ebd553c8305c043d2b9a02d6d32853a9bd4494278d330d0fa`).
The exact judge's high means are not evidence of score accuracy: the prompt
assumes each predicted score and asks whether its rationale is plausible.  Its
saturation and the null paired intervals make it unsuitable as the sole
reward or selection signal for these already polished rationales.

## Reproduction commands

Representative full commands (the paired arm changes only the named config or
prompt kind):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:. \
  .venv-standard/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_evaluation_prompt_rationale_sft.py \
  --config configs/evaluation_prompt_rationale_sft.score_blind.v2.json

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:. \
  .venv-standard/bin/python scripts/run_evaluation_prompt_rationale_generation.py \
  --prompt-kind evaluation_txt_rationale_score_blind_v2

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src:. \
  .venv-standard/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/run_evaluation_prompt_score_encoder.py \
  --config configs/evaluation_prompt_score_encoder.qwen3_embedding_8b.rationale_aware.score_blind.v1.json

scripts/run_official_q4_judge.sh full RUN_ID validation \
  data/processed/restricted/evaluation_prompt_participants_v1/ARM/participant.validation.jsonl \
  400 llm_as_judge.txt
```

For the conditioned generation command, the Qwen direct train/validation
score JSONL paths and Qwen direct `result.json` are supplied via
`--score-train`, `--score-validation`, and `--score-source-result`.

## Verification and terminal decision

The terminal check ran:

```bash
PYTHONPATH=src:. .venv-standard/bin/python -m unittest -v \
  tests.test_evaluation_prompt_matrix \
  tests.test_evaluation_prompt_rationale_sft \
  tests.test_evaluation_prompt_score_encoder \
  tests.test_official_rl_orchestrator
```

Result: 18/18 tests passed.  Relevant Python files passed `py_compile`, and
`bash -n scripts/run_official_q4_judge.sh` passed.  GPUs 0--3 were idle after
completion.

The deployable result of this matrix is the Qwen score-blind
rationale-aware encoder, but it should be described as a validation result,
not a final test estimate.  Further tuning on the same frozen validation would
invalidate that interpretation.  The score-conditioned path should not be
used as the default, and exact-judge reward optimization should not continue
without a separately authorized anti-saturation/calibration protocol.
