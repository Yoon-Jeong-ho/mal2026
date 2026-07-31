# Solar train-only scoring prompt search — 2026-08-01-001

- **Status:** completed negative experiment; target RMSE `< 0.4` was not
  reached and no Solar prompt-derived score was promoted to validation.
- **Runs:** `solar-prompt-search-v1-20260801-001` through
  `solar-prompt-search-v8-20260801-008`.
- **Question:** can prompt design, local anchors, evidence decomposition,
  relative comparison, or train-only calibration make
  `Solar-Open2-250B-Nota-INT4` a sufficiently accurate 1--5 writing scorer?
- **Privacy:** individual essays, identifiers, anchors, responses, extracted
  evidence, and predictions remain under ignored restricted/output roots.
  This record contains aggregate evidence only.

## Fixed protocol and runtime

All prompt discovery used a deterministic 160-row subset of the canonical
2,000-row train file. A disjoint 400-row train confirmation subset was reserved
and only the frozen v8 selection reached it. The remaining 1,440 train rows
formed the retrieval/anchor pool where a round required demonstrations.
Validation records read by v1--v8: **zero**.

- Canonical train SHA-256:
  `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737`.
- Split seed: `2026080105`; round seeds: `2026080105`--`2026080112` as
  fixed in the round configs.
- Frozen five-fold OOF R0/base artifact SHA-256:
  `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16`.
- Launch Git SHA: `a96f029b1ea640878607d4eb7bd817e6099334c1`;
  the pre-existing dirty worktree was preserved.
- Existing environment: `.venv-standard`; no package or environment was
  installed or created.
- Model: `nota-ai/Solar-Open2-250B-Nota-INT4` from the official image
  `upstage/vllm-solar-open2@sha256:b34b7dd42d986bafb8dcfb285758de2fbfd55e7cf28a1b5ba6b858e6b2b7c8c3`.
- Persistent endpoint/container:
  `http://127.0.0.1:19430`,
  `mal2026-solar-prompt-search-19430`.
- GPU scope: physical GPUs 0--3 only, tensor parallel 4 plus expert
  parallelism, CUDA graphs enabled (`enforce_eager=False`), prefix caching,
  chunked prefill, 12,288-token context, 32,768 batched tokens, 64 sequences,
  and 0.9 GPU-memory utilization. Full short-output batches reached
  99--100% utilization on all four GPUs.

Every round passed a local schema/context audit and a real endpoint smoke gate
before its discovery run. Row-level files were written fresh rather than
overwritten. Runtime events and negative integration results are preserved in
`outputs/solar-prompt-search-v*/<run-id>/ledger.jsonl`.

## Prompt families and discovery results

The common reference is the frozen R0 five-fold OOF base on the same 160 rows:
macro RMSE **0.581802**, macro Spearman **0.643144**. The table reports the
best predeclared candidate in each round.

| Round | Best candidate / method | RMSE | Spearman | Result versus base |
|---|---|---:|---:|---|
| v1 | axis threshold expected score | 0.697980 | 0.489298 | worse |
| v2 | 8-neighbor, axis-specific continuous score | 0.651048 | 0.524883 | worse |
| v3 | same-topic seven-band score grid | 0.850926 | 0.523202 | much worse |
| v4 | eight-neighbor OOF residual, final score | 0.623989 | 0.611509 | worse |
| v5 | base-centered ternary correction, step 0.25 | 0.590730 | n/a | worse |
| v6 | evidence-first then base ternary correction | 0.660494 | 0.628729 | worse |
| v7 | atomic-rubric features + OOF ridge, alpha 100 | 0.583128 | 0.636327 | slightly worse |
| v8 | anonymous direct pairwise brackets, 0.25 blend | **0.577599** | **0.643604** | `0.004202` better, not significant |

Important failure diagnostics:

1. **Mean regression and organization bottleneck.** Earlier absolute-score
   prompts systematically overpredicted low bands and underpredicted high
   bands. The best v2 per-axis RMSEs were about 0.558 content, 0.769
   organization, and 0.551 expression. Across 15 prompt variants, error
   correlation averaged 0.794 and reached 0.846 for organization.
2. **Prompt diversity did not create a deployable selector.** A pointwise
   oracle that sees gold could reach RMSE 0.201, but five-fold learned linear,
   convex, and ridge combinations remained around 0.63--0.65. The diversity
   was therefore not accompanied by a generalizable selection signal.
3. **Numeric residual output collapsed.** v4 delta prompts shifted scores
   downward by roughly 0.7--1.1 points and produced macro RMSE 1.10--1.22.
   Asking Solar to rewrite the final base score was less catastrophic but still
   worse than retaining the encoder base.
4. **Evidence-first prompting amplified weakness bias.** v6 direct continuous
   and integer scores were 0.958 and 1.008 RMSE. Extracting evidence before a
   categorical base correction still yielded 0.660 RMSE.
5. **Relative judgments were order-consistent but not correct.** v5 swapped
   ternary judgments agreed 79.4% of the time, but direction balanced accuracy
   was only 0.420. v8 removed every visible numeric/base/low/high hint and made
   two direct anonymous comparisons per anchor; discovery pairwise balanced
   accuracy was 0.406 with 77.1% order consistency. Thus consistency was not a
   correctness proxy.
6. **The only apparent gain did not confirm.** v8's discovery RMSE delta
   (candidate minus base) was `-0.004202`; a 10,000-sample paired essay
   bootstrap interval was `[-0.017735, 0.009447]`. The frozen 0.25 blend was
   then evaluated on the 400-row train confirmation subset. Base RMSE was
   **0.582565**, while the candidate was **0.585268** (worse by `0.002703`).
   Confirmation pairwise balanced accuracy was 0.391 and macro Spearman fell
   from 0.640933 to 0.625389. No validation run followed.

## Literature-directed changes

The later rounds were selected after prompt-only performance plateaued.

- Axis specialization and evidence/quote extraction followed the experimental
  direction in [Lee et al., *Unleashing LLMs' Proficiency in Zero-shot Essay
  Scoring*](https://aclanthology.org/2024.findings-emnlp.10/). It was tested in
  v6 and did not transfer to this Solar checkpoint and score contract.
- Direct comparison and order swapping followed the bias controls documented
  in [MT-Bench/Chatbot Arena](https://arxiv.org/abs/2306.05685) and the
  [systematic position-bias study](https://arxiv.org/abs/2406.07791). v5 and
  v8 explicitly recorded swapped-order consistency; both failed the
  predeclared direction-accuracy gate.
- The avoidance of balanced score-label grids and the use of train-only
  calibration were motivated by contextual label bias in
  [Zhao et al., *Calibrate Before Use*](https://proceedings.mlr.press/v139/zhao21c.html).
  The v3 score grid independently worsened Solar, while v7's leakage-free
  atomic-feature calibration did not beat the base.
- Independent atomic rubric questions in v7 followed the feature-decomposition
  direction of [LLM-Rubric](https://aclanthology.org/2024.acl-long.745/), but
  every OOF ridge alpha was worse than the frozen encoder score.

These sources motivate the tested controls; they do not claim that the methods
must improve Korean writing scores or guarantee an RMSE threshold.

## Decision

Stop the Solar **prompt-only numeric scoring** search for this checkpoint and
do not use its absolute scores as synthetic labels. Eight rounds, a disjoint
train confirmation, and literature-backed evidence/pairwise/atomic variants
all reject the hypothesis that prompt changes can reduce RMSE from about 0.58
to below 0.4. Further wording changes would mainly increase multiple-testing
and validation-overfitting risk.

Solar remains potentially useful for rationale generation or qualitative
auditing, but score learning should remain with the trained encoder. Any next
RMSE experiment should add genuinely new supervised signal (especially
adjudicated organization and tail examples) or change the score model/training
objective, not reinterpret Solar's internally consistent but inaccurate
comparisons as labels.

The persistent Solar server was deliberately left ready after the run, as
requested; its idle 79.7--79.8 GiB allocation per GPU is model residency, not
an active job.

## Reproduction and aggregate artifacts

Round configs and SHA-256 values:

```text
v1 991d2a5ec5686a59afbce5eb69ff28e0d25063252c89eb9ddf6a5536ce6115ff
v2 e44635075a9f52e4a3fa584f3564ba6ed8dcc60a435a574bdb50d854e0b613eb
v3 302871809d4dd4d377890c219729e5ea53895c979fe7a637af900e9875601db2
v4 725786d15f00283f954191ae541c28f738cd41f9463c0123abdf88cd2d17b68b
v5 3f688e22501769929632b09fe41de9e965a2304f10ad817d70c3ab6c26059efd
v6 e59ba609823b435f54bdd74f05cf716bca90c03e03d0e06a5d9cd5f470c4f9f5
v7 c36ac8ed6ae8df63742f634702eb6ca339c1d0d1abe216b4ae04ca0c30133d8b
v8 fe31c43e6e89dbe06b547db69b103e9058475d1a104ca33ec31e3449acd4e5c8
```

The exact stage pattern was:

```bash
PYTHONPATH=src .venv-standard/bin/python scripts/run_solar_prompt_search_vN.py \
  --config configs/solar_prompt_search.vN.json --stage prepare
PYTHONPATH=src .venv-standard/bin/python scripts/run_solar_prompt_search_vN.py \
  --config configs/solar_prompt_search.vN.json --stage preflight
```

v1--v6 then used their declared `run`/`aggregate-discovery` stages. v7--v8
used `run-features --split discovery` followed by `aggregate-discovery`; v8
alone passed its frozen selection gate and used `run-features --split
confirmation` followed by `aggregate-confirmation`. Candidate names and exact
fixed arguments are enumerated in the corresponding configs and preserved
runtime logs.

Public aggregate SHA-256 values, v1 through v8 discovery, are respectively:

```text
768aa77467153a8eb9a6664a7c7f70f1f87363978cfd02a28cbb5b8d7229875e
1dc8289b540aed9fef3ad5a28dac403152adcf263c7817794db2575a79389c9c
397838bb2accdd81f1c05a1084a4f54eea2b27dd54a88d2bd55bb6624b0029d5
5dd884901bd0e2637505049787d0b34669f979532bd2e48d91bdc62400d270dd
a48a03bd115ca97119bb450537fb6d7748cc7d5283580b9de2cc0396ccdf6613
130bf659de1cca50638fa09006e8a4530d1868df9bddcbe9e976a586df1a6232
5f562905c415000777ba70b23bbc6d3194cb0bb318229ffd3efdc1022eb8aabc
79ed9dee09487715fdec0b52a79eb510b3355f90c1e04607afd5136290198664
```

The v8 train-confirmation aggregate is
`outputs/analysis/solar-prompt-search-v8-20260801-008/confirmation/aggregate.json`
(SHA-256 `f11989f8cb8fb0fd7a5230e7ebdc7c08a8f4ea69a2d4315795ccb7ac7141237f`).
