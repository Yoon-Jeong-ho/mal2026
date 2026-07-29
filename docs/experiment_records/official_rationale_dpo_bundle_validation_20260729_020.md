# Bundle-DPO frozen-validation comparison — run 020

## Contract and provenance

- Evaluation contract: one bundled participant JSON containing content,
  organization, and expression scores/rationales. Axis-triplet outputs are not
  generated, trained, or used for selection.
- Split: frozen validation, 400 records. Validation was not used for DPO
  preferences, reward, or training.
- Authorized hardware: physical GPUs 0–3 only. DPO rationale generation used
  vLLM tensor parallel 4; exact-Q4 judging used four independent one-GPU
  llama.cpp servers with four slots each.
- Exact judge prompt SHA-256:
  `91cd2f94fa78cc1a07d1bb63a1c5faf07fa25d77d5c60bf6952081c8f047cb6d`.
- Score-file SHA-256:
  `ee1ea09fd25028483482465569a61d8b60edf03015675f03c1fbf6f6956c8a4e`.
- Baseline participant SHA-256:
  `7b3e02b7e87eb7c24f9e10f2e716fa204a2086603f805c5bc28efc697f5e3a64`.
- DPO training completion SHA-256:
  `6fbbcc5f1efc4402452916afffa9331f1d9575bee830c8bd612af2bed949df96`.
- Seed: generation/judge 42; temperature 0.

```bash
PYTHONPATH=src:. .venv-standard/bin/python \
  scripts/run_official_bundle_dpo_validation.py \
  --run-id official-rationale-dpo-bundle-validation-exact-judge-20260729-020
```

The first runner invocation completed TP4 generation and composition, then
exited before Q4 launch. The terminal exception text was unavailable, so its
exact cause is uncertain. All completed artifacts and checksums were
preserved; the same run resumed with `--resume-after-generation` and completed
the judge stage. This is integration-recovery evidence, not a model result.

## Frozen-validation result

Both arms passed 400/400 strict judge parses (4,800 cells each).

| Arm | Macro (/5) | Mean total (/60) | Perfect 60/60 |
|---|---:|---:|---:|
| SFT baseline | 4.9839583 | 59.8075 | 89.25% |
| Bundle DPO | **4.9877083** | **59.8525** | **93.00%** |

Paired totals: DPO won 37, tied 340, and lost 23 examples. Mean DPO-minus-SFT
delta was `+0.045/60` (median 0); the two-sided exact sign-test p-value was
`0.092461`. Axis deltas on the five-point scale were content `+0.01125`,
organization `+0.00375`, and expression `-0.00375`.

The DPO arm is retained as the numerically highest rationale model, but the
effect is small and not robust at a conventional 0.05 sign-test threshold.
With 85% paired ties and 93% perfect totals, the exact judge is ceiling
saturated. Additional judge-optimized GRPO is therefore not run: its reward
would be nearly constant, while axis splitting would optimize a contract the
deployed evaluator does not use. This is a saturation stop, not a claim that
rationale quality is solved.

Aggregate comparison:

`outputs/official-rationale-rl-v1/evaluation/official-rationale-dpo-bundle-validation-exact-judge-20260729-020/aggregate_bundle_dpo_validation_comparison.json`

All row-level generations, participants, and judge records remain in ignored
restricted paths. This record contains no essay, rationale, identifier, or
individual prediction.
