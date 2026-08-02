# Ordinal tail program — 2026-08-03 run 001

## Status and authority

- **Run ID:** `ordinal-tail-program-v1-20260803-001`
- **Parent Git SHA:** `8b6cdd06d5a7692540a631d72e9ec4d85088a96b`
- **Authorization:** the user approved the end-to-end ordinal-tail plan and asked
  that RMSE be driven as close to `0.4` as the evidence permits.
- **GPU scope:** GPUs 0–3 only; GPU 0 is the smoke-test device.
- **Selection population:** train-only nested OOF (five fixed outer folds, four
  inner folds). The repeatedly exposed validation split is descriptive only.
- **Targets:** `content`, `organization`, and `expression` are learned and
  evaluated independently. The supplied `average` field is forbidden as a
  feature, target, calibration input, or selection metric.

The `0.4` macro RMSE is a stretch target rather than an expected result. It
would require a 29.7% RMSE reduction (about a 50.5% MSE reduction) from the
exact R0 OOF baseline. Each negative result is retained instead of changing a
protocol after observing its validation performance.

## Immutable inputs

| Input | SHA-256 |
|---|---|
| `eval/train.jsonl` | `b24d2f1fcab24536774606f0f6b198aec647561e72f36cfbfdf7968d5b245737` |
| frozen R0 embedding rows | `949451b690ea12df126bc6aa9b1cf7f2f016e3c60f69d67b1d460719b6404f16` |
| exact R0 OOF predictions | `823b0c0b2114d9442e1d4e65ff7fd310e529b82d1283c78b2466b7a0141eec04` |

The frozen-feature screen uses the existing 4,096-dimensional, checksum-bound
R0 Qwen embedding artifact. This is a recorded deviation from the initial
draft's proposed new KURE feature cache: it provides the strongest exact OOF
baseline comparison without generating another representation. KURE remains
the backbone for the promoted end-to-end candidates.

## Stage 0 diagnostic

Command:

```bash
PYTHONPATH=src .venv-standard/bin/python \
  scripts/run_ordinal_tail_diagnostics.py \
  --config configs/ordinal_tail_program.v1.json
```

Aggregate output is stored in the ignored run directory. No essay, prompt,
identifier, embedding, or row prediction is copied into this record.

| Metric | Exact R0 OOF |
|---|---:|
| macro RMSE | 0.568780 |
| equal-group RMSE | 0.691549 |
| low-tail `{1,2}` RMSE | 0.923335 |
| score-5 RMSE | 0.884190 |
| gold-3/4 balanced accuracy | 0.643313 |
| macro Spearman | 0.600288 |

Axis RMSE is `0.509024` content, `0.688392` organization, and `0.508925`
expression. Rounded R0 predictions contain no score-5 predictions in any axis
and only one score-1 prediction overall. Raw labels are fractional in 1,812
content, 1,453 organization, and 1,455 expression rows, so a raw-score MSE
auxiliary is retained rather than replacing the target with integer classes.

## Preregistered stages

1. **Fixed-feature screen.** Compare natural CE, RPS, CORAL, CORN, SLACE at
   alpha 0.5/1/2, effective-number CE at beta 0.99/0.999, and sqrt-tempered
   sampling. Weighted/sampled posteriors are corrected back to the natural
   train-fold prior for the primary RMSE. No correction methods are stacked.
2. **End-to-end KURE/cRT.** Train only the best two distinct loss families from
   the nested screen, preserving fold isolation and the public/AI-Hub
   provenance boundary.
3. **Prompt-reference NPCR.** Compare prompt- and trait-local score-difference
   learning with adjacent and skip-gap pairs; all references are fit-fold
   local.
4. **Calibration/limited ensemble.** Fit axis-wise calibration and any ensemble
   weights only from OOF predictions. Keep an unchanged R0 candidate.
5. **Final refit and descriptive validation.** Freeze the protocol, refit the
   selected candidates, and read validation only after all selection is over.

The promotion gate requires at least `0.005` macro RMSE improvement, no more
than `0.01` RMSE degradation on any axis, no low-tail or score-5 degradation,
no more than `0.01` loss in gold-3/4 balanced accuracy, no more than `0.005`
loss in Spearman, and a positive paired-bootstrap improvement lower bound.

## Environment note

The existing `.venv-standard` environment does not contain `pytest`. Package
installation is prohibited, so test modules are executed with Python's
standard `unittest` runner. This environment limitation and the failed pytest
probe are preserved in the append-only ignored run ledger.

## Stage 1 frozen-feature result

The five outer folds and the 10,000-resample paired bootstrap completed under
Git SHA `c79542369b98fd9795697aa414eafe7410499068`. The exploratory candidate
was selected independently inside each outer-train population; all five folds
selected `coral-natural`.

| Metric | exploratory CORAL | exact R0 | improvement (positive is better) |
|---|---:|---:|---:|
| macro RMSE | 0.631782 | 0.568780 | -0.063002 |
| equal-group RMSE | 0.792635 | 0.691549 | -0.101086 |
| low-tail `{1,2}` RMSE | 1.252784 | 0.923335 | -0.329449 |
| score-5 RMSE | 0.927654 | 0.884190 | -0.043464 |
| gold-3/4 balanced accuracy | 0.620455 | 0.643313 | -0.022857 |
| Spearman | 0.485949 | 0.600288 | -0.114339 |

The paired macro-RMSE improvement interval was `[-0.073804, -0.052106]`;
therefore every point gate and the bootstrap gate failed. The checksum-bound
protected output remains exact R0. This is retained as a negative result: a
small ordinal head over the frozen public embedding does not replace the
fine-tuned R0 scorer.

The preregistered distinct-family ranking still advances only the two cheapest
representation hypotheses to the end-to-end test:

1. `coral-natural` — mean outer/inner OOF RMSE `0.645302`;
2. `rps-natural` — mean outer/inner OOF RMSE `0.664481`.

Stage 2 therefore uses the validation-free config
`configs/kure_ordinal_oof.v1.json`: pinned KURE-v1 plus the checksum-bound
AI-Hub backbone, independent per-axis LoRA, six phase-one epochs, and a
three-epoch natural-prior cRT head. Both phases preserve the declared
`ordinal loss + 0.25 raw-score expected-score MSE` objective. A GPU0 plumbing
smoke precedes the exact five-fold GPU0–3 OOF run; no canonical validation
path exists in the Stage 2 config.

## Stage 4 — prompt-reference NPCR (preregistered)

Stage 4 is bound by `configs/prompt_reference_npcr.v1.json` and uses only the
checksum-bound canonical train rows, exact R0 OOF fold/prediction lineage, and
the frozen 4,096-dimensional public embedding artifact. R0 continuous scores
are verified against the artifact per ID/fold/axis but are never a feature of
the NPCR utility network; they are retained only as the protected identity
fallback and aggregate comparator. No validation path/string is allowed in the
configuration, and `average` is forbidden.

For every axis and fit fold, pairs are same-`prompt_num` only, exclude self and
same-document comparisons, and are selected deterministically from adjacent
gap 1 plus skip gaps. The scalar utility loss is Huber on raw-axis-score
difference `f(query)-f(reference)`. A held row is restored only from fit-fold,
same-prompt anchors via the clipped median of `anchor_score + f(query) -
f(anchor)`; held gold never selects an anchor.

The fixed matrix has two candidates: `adjacent-skip2` (gaps 1/2, two references
per gap) and `adjacent-allskip` (gaps 1/2/3/4, one reference per gap), both
with eight anchors. For each of the five outer folds, candidates are evaluated
on four disjoint 1,200→400 inner fits and selected **only against each other**
by the fixed lexicographic tuple: raw macro RMSE, equal-group RMSE, low-tail
RMSE, high-tail RMSE, then candidate ID. Exact R0 predictions never enter
inner selection or an outer refit; every outer fold always refits the selected
NPCR candidate and emits one honest NPCR prediction.

Only after all five outer predictions are assembled is the complete nested
NPCR OOF compared with protected exact R0. Promotion then requires macro
raw-RMSE improvement of at least 0.005, each axis worsening by at most 0.01
RMSE, 3/4 balanced accuracy falling by at most 0.01, nonzero global low
`{1,2}` and high `{5}` support, low/high tail non-inferiority, and a source-ID
clustered, three-axis-preserving 1,000-resample paired-bootstrap macro-RMSE
lower bound above zero. Score 1 is descriptive only. Failure changes only the
global recommendation to the exact-R0 identity; it does not replace any outer
NPCR prediction.

`scripts/run_prompt_reference_npcr_gpu0_3.sh smoke` performs the one-epoch,
one-axis GPU0 plumbing smoke and records config/runner/module/launcher/test
hashes; its eight-row result is explicitly non-selectable. `full` refuses to
run unless that attestation still matches, then schedules folds 0/4 on GPU0
serially and folds 1/2/3 on GPUs 1/2/3. Both modes refuse occupied GPUs and
never overwrite outputs. No Stage 4 GPU launch has occurred while Stage 3 owns
GPUs 0--3.

## Stage 5 pre-result preregistration — 2026-08-03T03:57:01+09:00

At registration time, the observed count of Stage 3 outer output JSON files was
exactly `0`. The non-runnable manifest
`configs/conservative_oof_combiner.prereg.v1.json` therefore fixes the Stage 5
scientific choices before any Stage 3 outer result exists: the source method is
`coral-natural`, calibration is identity with no learned calibration or weight
selection, and the only candidate is `0.8 * exact_R0 + 0.2 * coral-natural`.
The manifest also fixes every promotion threshold, including Spearman
non-inferiority with at most `0.005` loss, and deterministic Stage 3/4 artifact
path templates.

After upstream completion, a runnable runtime config may populate only pending
artifact hashes, their already-fixed deterministic paths, and observed
completion metadata. It may not change source identity, calibration, weights,
gate thresholds, or any other scientific value in the preregistration.

## Stage 6 submission decision-list preregistration — 2026-08-03

Before any Stage 3 outer result is read, `configs/stage6_submission_prereg.v1.json`
fixes at most three hidden submission slots. H0 is always the deployable historical
full-train R0 epoch-1--4 **axis-wise** continuous prediction ensemble. H1 is
Stage 3 `coral-natural` only when the fixed common gate passes; otherwise it is
Stage 4 NPCR only when Stage 4's already-fixed gate passes. H2 is only the
already-preregistered Stage 5 identity `0.8 * H0 + 0.2 * coral-natural` blend
when its own fixed gate passes. RPS is descriptive only and is never refit or
submitted through this decision list.

The Stage 3 gate is fixed as: at least 0.005 macro-RMSE improvement; at most
0.01 per-axis RMSE worsening and 0.01 gold-3/4 balanced-accuracy loss; at most
0.005 macro-Spearman loss; non-inferior low `{1,2}` and high `{5}` tails with
nonzero support in every axis; finite metrics across all five outer folds; and
a 10,000-resample source-ID-clustered, three-axis-preserving paired-bootstrap
lower bound strictly above zero for `R0 RMSE - candidate RMSE`.

A final candidate is a frozen procedure, not an outer-fold checkpoint. After
its decision-list condition passes it may be refit only on all 2,000 canonical
training rows using the same model family, hyperparameters, seed policy, and
axis-wise raw-score objective. Stage 4's full-refit candidate is the mode of its
five already-nested outer selected candidate IDs, with a lexicographic tie-break;
this does not open a new matrix search. No `average` target, feature, loss, or
selection metric is permitted.

Validation remains a one-time descriptive aggregate-only audit after hashes for
all present candidate artifacts and their submission order have been frozen in
the run ledger. It cannot select, rank, remove, reorder, tune, or re-evaluate a
candidate. If all Stage 3/4/5 gates fail, H0 alone is retained and empty slots
are not filled with variants.

### Stage 6 implementation-freeze addendum — pre-commit

The manifest now fixes full-refit seeds rather than merely a seed policy. Stage 3
`coral-natural` uses base `2026080302` and
`int.from_bytes(sha256(f'{base}\0full-refit\0coral-natural\0{axis}\0{phase}'.encode()).digest()[:4], 'big') % (2**31 - 1)`:
phase-1/cRT seeds are content `592530948`/`782205628`, organization
`677979439`/`152263886`, and expression `286654097`/`1508715968`. Stage 4
uses its first-15-hex derivation with `outer=full-refit` and `phase=outer-refit`.
For `adjacent-allskip`, content/organization/expression seeds are `704285545`,
`1314313536`, and `224474707`; for `adjacent-skip2` they are `1185757628`,
`1203094570`, and `811459031`.

H2 may exist when the already-preregistered Stage 5 gate passes, independently
of the standalone Stage 3 H1 gate. Exactly one Stage 3 `coral-natural`
full-refit artifact is created if either gate passes; H2 always reuses that
artifact by hash, and H1 reuses the same hash when its Stage 3 gate also passes.
If H1 falls back to Stage 4 NPCR, H2 still uses that sole coral artifact. A
second coral refit is forbidden. For an H1 Stage 4 NPCR full refit, the five
nested outer selected candidate IDs are reduced by a fixed modal rule with
lexicographic tie-break and no new candidate search.

Deployment is frozen as follows. Stage 3 scores the official prompt plus essay.
Stage 4 uses the exact H0 score-blind `rank2_ax4_random1` rationale generator
(seed 42, temperature 0, top-p 1, 512 tokens), exact legacy `_score_input`,
and public Qwen3-Embedding-8B at revision
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, with 2,048-token truncation
and last-nonpadding float32 L2 pooling. It persists the all-2,000 restricted
same-prompt labeled anchor library and the exact bijective canonical
prompt-text↔`prompt_num` map (9 unique prompt texts and 9 unique prompt numbers).
Submission supplies prompt text, so `prompt_num` is inferred only by an exact
prompt-text match. An unknown/mismatched prompt text, unavailable inferred
`prompt_num`, or fewer than one eligible anchor uses the query's H0 continuous
prediction unchanged. H1/H2 reuse the H0 final
DPO rationale component, conditioned on their own emitted integer scores; the
blind/final adapter hashes are bound in the manifest.

The historical H0 metric is grandfathered as its sole already-recorded
contextual evaluation and is never rerun. The one-time descriptive validation
audit applies only to newly frozen H1/H2 artifacts after their hashes and order
are ledgered; it cannot affect any gate, slot, ranking, or retuning decision.
