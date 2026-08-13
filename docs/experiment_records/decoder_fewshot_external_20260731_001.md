# External decoder few-shot validation — 2026-07-31-001

- **Status:** completed.
- **Run ID:** `decoder-fewshot-external-v1-20260731-001`.
- **Question:** under the already-fixed validation-only five-shot scoring
  protocol, do Solar-Open2 INT4 or the available Terra/Luna API models improve
  decoder scoring over the five local decoder baselines?
- **Privacy:** individual validation prompts, writings, mappings, responses,
  scores, and identifiers remain only under ignored restricted/output roots.
  This record contains aggregate evidence only.

## Authorization and fixed protocol

The user explicitly requested the Solar quantized model and Terra/Luna API
experiments. This authorized the official Solar Docker pull and external API
transfer for this named validation task. Only GPUs 0--3 were inspected or
used. No package or environment was installed.

The scientific protocol is unchanged from
`decoder-fewshot-validation-v1-20260731-001`:

- user-supplied `evaluation.txt` verbatim, SHA-256
  `1950b3f837bf390032cd6eb03214e718f972531d442060e3c611bef1da7e1145`;
- 400 validation writings and two prompt conditions per model (800 requests);
- `balanced5`: one train-only demonstration of every score 1--5 per axis;
- `central5`: three score-3 and two score-4 train-only demonstrations per axis;
- five fixed rotations, exactly 80 validation writings per rotation;
- exact JSON with integer scores and Korean rationales for `content`,
  `organization`, and `expression`; no `average` field;
- no validation score is included in a request or used to select/order shots.

External config SHA-256:
`e24d56472f660ecab5deadfb206d2d2d877556339cc004998f2296393f840d56`.
The base shot manifest SHA-256 is
`a9431b7b432526bf30659dd33d812f35c967286e0a107903a63b93eb70bfe111`.
Launch Git SHA was `a96f029b1ea640878607d4eb7bd817e6099334c1` with a dirty
worktree, which was preserved.

## Runtime and external transfer

### Solar-Open2

- Model: `nota-ai/Solar-Open2-250B-Nota-INT4` (local W4A16/INT4 weights).
- Official image:
  `upstage/vllm-solar-open2@sha256:b34b7dd42d986bafb8dcfb285758de2fbfd55e7cf28a1b5ba6b858e6b2b7c8c3`;
  inspected image ID is the same digest and content size is 30,264,870,473
  bytes.
- Official image's vLLM 0.22.0, tensor parallel 4 plus expert parallel on GPUs
  0--3, CUDA graphs enabled (`enforce_eager=False`), 90% memory target,
  64 sequences, 32,768 batched tokens, prefix caching, and chunked prefill.
- All 800 prompts passed a tokenizer/context audit (4,568--4,871 prompt
  tokens; 12,288 context). Two real requests passed before the full run.
- Server initialization took 266.55 seconds after process initialization. It
  served 802 successful HTTP requests (2 smoke plus 800 full), with no retry or
  parse failure. Full-run usage was 3,761,692 prompt and 206,490 completion
  tokens. GPUs were verified idle after shutdown.

### Terra/Luna Responses Batch API

- `store=false`, reasoning effort `none`, strict JSON schema, 1,800 maximum
  output tokens. These fixed API models do not accept a caller-selected
  temperature, so provider-default sampling is an explicit reproducibility
  limitation.
- Repaired request sizes were 23,079,106 bytes for Terra and 23,078,306 bytes
  for Luna (46,157,412 bytes total). Request SHA-256 values were
  `b1fbfbe92715a7f5fc5555c743ba375c678b01236afbb9a4ecb4467563cfa486`
  and
  `572e20b1480295ee4f2fc42770c11b82b7875f89c491cede5746632a0060d631`,
  respectively.
- Luna batch `batch_6a6cad4261548190b0192593b2b9f160` completed 800/800
  with zero failures. Usage: 5,383,790 input, 323,202 output, and 5,706,992
  total tokens.
- Terra batch `batch_6a6cad3c6644819098156c7e928da831` completed 800/800
  with zero failures. Usage: 5,383,790 input, 323,976 output, and 5,707,766
  total tokens.
- The API does not return billable currency amounts, and no public price was
  assumed for these model aliases. Token and byte counts above are therefore
  the exact available cost/transfer evidence; dollar cost is not yet
  independently determined.

## Results

`raw RMSE` compares integer decoder predictions with continuous validation
scores. `int RMSE` compares them with half-up integer gold. Spearman is against
continuous gold. Each completed arm has 400 writings and 100% parse coverage.

| Model | Shots | Raw RMSE | Spearman | Int RMSE | Score 3 | Score 4 |
|---|---|---:|---:|---:|---:|---:|
| Solar-Open2-250B INT4 | balanced | 0.7619 | 0.4402 | 0.7948 | 33.92% | 59.25% |
| Solar-Open2-250B INT4 | central | **0.7227** | 0.4424 | **0.7773** | 47.83% | 50.00% |
| GPT-5.6 Luna | balanced | 0.8094 | **0.4643** | 0.8486 | 36.75% | 50.67% |
| GPT-5.6 Luna | central | 0.7932 | 0.4465 | 0.8495 | 45.92% | 44.50% |
| GPT-5.6 Terra | balanced | 0.7633 | 0.4957 | 0.8040 | 36.33% | 50.83% |
| GPT-5.6 Terra | central | 0.7551 | **0.4971** | 0.7964 | 39.67% | 50.42% |

Solar central is already the strongest individual decoder arm in this project:
it improves raw RMSE by 0.0988 over the previous individual best, Qwen3.5-9B
central (0.8215), and by 0.1332 over the constant-3 baseline (0.8559). It does
not improve rank correlation over the previous best Qwen3.5 balanced arm
(0.4958).

The improvement remains dominated by the central distribution rather than
reliable tail scoring. Across all three axes, Solar central exactly predicts
only 9/116 score-2 labels (7.76%), 0/9 score-1 labels, and 0/105 score-5
labels. Solar balanced improves those to 23/116 (19.83%), 2/9 (22.22%), and
11/105 (10.48%), respectively, but worsens global RMSE. Terra balanced is the
strongest API tail arm, but still reaches only 36/116 score-2 labels (31.03%),
2/9 score-1 labels (22.22%), and 12/105 score-5 labels (11.43%). Thus diverse
shots move the output distribution, but still do not calibrate the tails.

A post-hoc, label-free equal mean of Solar and Luna central predictions has
raw RMSE 0.7077 and Spearman 0.4897. This ensemble was not a predeclared
primary arm and is diagnostic only, not a validation-selected promotion.

## Negative results and integration recovery

- The first two API smoke attempts failed with HTTP 400 before any batch upload:
  Responses API assistant-role demonstrations require `output_text`, not
  `input_text`. The invalid prepared files and logs were preserved. Only the
  wire representation of assistant messages was repaired; prompts, shots,
  order, validation inputs, and scoring contract did not change. Both repeated
  two-request smokes passed.
- Luna's first finalized public aggregate mislabeled request SHA-256 as config
  SHA-256. The original is preserved as
  `aggregate.before_metadata_repair.json`; the current aggregate separates
  `config_sha256` and `request_sha256`. Metrics and restricted predictions did
  not change.

## Current interpretation

Solar-Open2 INT4 is materially better than every individually tested local or
API few-shot decoder on validation RMSE, while Terra central provides the best
individual rank correlation in this external comparison. A simple Solar/Luna
mean is better in RMSE again. However, all three external models remain
concentrated on scores 3 and 4 and are weak on true scores 1, 2, and 5. They
therefore remain unsuitable as a standalone replacement for the trained
encoder scorer. Their most defensible near-term use is as a diverse auxiliary
signal or disagreement feature; using them as synthetic score labels would
require a separately fixed, tail-aware filter/calibration protocol.

Public aggregates are under
`outputs/analysis/decoder-fewshot-external-v1-20260731-001/`; restricted
requests and predictions are under the ignored
`data/processed/restricted/decoder_fewshot_external_v1/` run directory.
The completed aggregate SHA-256 is
`c8dc18b84db1ad503575b185f5035291dad8c7f6efb24b8e4fdef89630e77bec`.
