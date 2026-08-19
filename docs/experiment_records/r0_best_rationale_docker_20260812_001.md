# R0 ensemble plus best rationale Docker publication: 20260812-001

**Status:** completed. The selected hybrid was staged, its prompt and API
contracts were tested, the model-bearing image passed a no-argument GPU0 smoke,
and tag `20260812-003` was pushed to Docker Hub.

## Authorization and boundaries

- The user requested selection and publication of the strongest configuration
  supported by the rationale-pipeline final report.
- Default MAL2026 GPU scope 0--3 applied; only physical GPU0 was used. The
  minimum read-only check showed GPU0 idle before launch. No pre-existing
  process was terminated, displaced, or attributed. The owned container was
  removed and GPU0 returned to 0 MiB afterward.
- No training, retuning, new data generation, package installation, or
  environment creation occurred. Existing artifacts and `.venv-standard` were
  used. The Docker repository already existed.
- Git SHA: `9cd82664879ca3b9e2b063b64cecbbd17701c652` with unrelated pre-existing
  working-tree changes left intact.
- Restricted generated rationale rows were read only for an aggregate length
  audit. No row, text, identifier, score, or prediction is recorded here.

## Selection

The latest report's best score encoder was not selected because its integer
macro RMSE, 0.6912499683146406, is worse than the existing R0 ensemble's
0.6158981882311673. The score path therefore remains the historical four-epoch
Qwen3-Embedding-8B uniform prediction ensemble, whose continuous macro RMSE is
0.5582937519204271.

Only the final rationale stage was updated to the final report's best rationale
candidate:

- candidate: `mal-direct-lora-epoch2`;
- adapter model SHA-256:
  `48eff87081928adb08ae483e2ceb40c67615777a3d9a1f1e69f4a0c9382d5dfb`;
- score-blind rationale judge macro: 4.926875;
- judge worst-cell mean: 4.7375;
- source final report SHA-256:
  `a80e134a4266b2db1828f596fc6111aedd2e04fe91dea3ae9f82cc369c2e8b11`;
- completion audit SHA-256:
  `dec604e8926f6874c6781c98ffb0e4df362f8c7313275c775138e70a6ad61a48`.

This combination preserves the independently measured R0 score metric because
the new rationale is generated after score prediction. The exact end-to-end
pairing of R0-emitted integer scores with this rationale candidate was not
re-judged; rationale quality above is the final report's independent selection
evidence, not a new hybrid judge metric.

## Prompt contract

The organizer's combined message is accepted as one `user` role containing the
scoring instructions, `[prompt_text]`, and `[essay_text]`. The serving boundary
extracts only the two marked task fields. The latest rationale adapter then
receives the exact system/user prompt used for its training and evaluation:

```text
Rationale_evaluation_training.txt
sha256:c7d18cdfceb82cba9d355e9f98b0cea7cc60f500bd6e56494ac87be0d3160285
```

A focused parity test compared the serving renderer against
`mal2026.rationale_pipeline_prompts.rationale_messages` using quotes, braces,
newlines, and marker-like text. The message lists were equal byte-for-byte at
the Python string level. JSON string escaping was retained and neither
`reference_scores_integer` nor `predicted_score` appeared in the policy prompt.
The prompt file and expected SHA-256 are also embedded in the image manifest;
startup fails closed on a mismatch.

The three relevant test modules passed 24/24 tests. The latest 400-record
rationale output aggregate showed per-axis maximum lengths of 320, 278, and
277 characters, zero rationales above the serving 384-character limit, a
maximum combined tokenizer length of 327, and zero outputs above the internal
512-token budget.

## Build and integration recovery

The new image uses the verified R0 v2 image as an immutable base and overlays
only serving code, the latest prompt, adapter, and manifest:

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --file deployment/Dockerfile.r0-best-rationale \
  --build-arg 'BASE_IMAGE=docker.io/yoonjh1/mal2026-r0-ensemble@sha256:02e651ef1eed001da2232e5270db35117c34938713741eb82df2fca95615a05a' \
  --build-arg 'RUNTIME_OVERLAY=runtime_overlay_r0_best_rationale' \
  -t docker.io/yoonjh1/mal2026-r0-ensemble:20260812-003 \
  deployment
```

The first launch returned health 503 but the earlier server did not expose its
initialization error. Error traceback logging was added. The second launch then
identified `SubmissionContractError: runtime pipeline differs`: the new hybrid
manifest kind had not been added to the top-level R0 loader dispatch. The
dispatch was repaired and covered by a focused unit test. A separate stdin
diagnostic attempt failed in vLLM multiprocessing because `<stdin>` is not an
importable spawn path; it was not treated as model evidence. All negative logs
remain in the ignored output directory.

## GPU/API smoke

The final image started with no command after its image reference, using GPU0
only and the inherited entrypoint
`["python3","-m","mal2026_submission.server"]`.

- health-ready time: 124 seconds;
- `/v1/models`: `mal2026-r0-ensemble-best-rationale-v1`;
- one synthetic, combined-user task request: HTTP 200 in 31.66465 seconds;
- emitted integer scores: content 3, organization 3, expression 4;
- rationale character lengths: 183, 175, 162;
- response began with `{`, contained no fence, survived the organizer's exact
  naïve first-JSON extraction logic, and contained no rationale brace;
- peak observed GPU memory: 52,099 MiB on an H100 80 GB;
- the owned container was removed and GPU0 returned to 0 MiB.

The prior R0 bundle's conservative L40S memory proxy measured 36,677 MiB under
a slightly smaller vLLM reservation than production would obtain on a 46,068
MiB L40S. The model bases and LoRA tensor shape/size are unchanged here, but an
actual L40S kernel/latency run remains unverified.

## Published image

```text
docker.io/yoonjh1/mal2026-r0-ensemble:20260812-003
docker://yoonjh1/mal2026-r0-ensemble@sha256:fb226eb5263021b6c35190ada65920c8101b7dedc45a836eb476f8fc10907f28
```

- OCI index digest:
  `sha256:fb226eb5263021b6c35190ada65920c8101b7dedc45a836eb476f8fc10907f28`;
- Linux/amd64 manifest:
  `sha256:70a052a02e79b0006d68a111777c6402f59e00940588f0dcf939ed9b3789b974`;
- config:
  `sha256:f89bc62ab8af279d915d32aa7d815a97cb014d87f8baa3572a931875ddf314b0`;
- compressed layer bytes: 33,664,030,086;
- compressed layers plus config: 33,664,071,335 bytes;
- logical local image size: 33,664,079,738 bytes.

The push completed successfully. An anonymous registry token and manifest GET
returned HTTP 200 at verification time, proving that the repository was public
at that moment. Per-blob HEAD sweeps were deliberately not repeated because
they previously inflated Docker Hub access counters; the authenticated remote
manifest, successful push digest, public anonymous manifest, and pre-push local
GPU smoke bind the published artifact sufficiently.

Ignored evidence is under
`outputs/r0-best-rationale-docker/r0-best-rationale-docker-20260812-001/`.
