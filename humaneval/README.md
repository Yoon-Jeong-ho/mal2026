# Human score and rationale validation app

This local application supports a fixed, source-blind review of 20 writings
from the restricted `eval/train.jsonl` and `eval/validation.jsonl` inputs.
Writing text, prompts, and rationales are read at runtime and are never copied
into tracked files or the response database.

## Folder layout

Everything needed to maintain the interface is isolated here:

```text
humaneval/
├── README.md          # operations, privacy, and study contract
├── run.py             # preflight, launch, and export CLI
├── core.py            # selection, input validation, and SQLite persistence
├── server.py          # dependency-free local HTTP server
├── web/               # browser UI
├── tests/             # synthetic tests only
└── records/           # aggregate, non-sensitive protocol records
```

Restricted writings, rationales, and response databases are intentionally not
inside this tracked directory.

## Frozen default design

- Reviewers: 명훈, 찬희, 정호, 지민.
- All reviewers receive the same 20 writings in the same order.
- Hidden selection bands: independent half-up rounding of the source content,
  organization, and expression scores.
- Each axis has the same 20-essay distribution: score 1 = 4, score 2 = 4,
  score 3 = 2, score 4 = 2, score 5 = 8. Across all 60 axis judgments this is
  score 1 = 12, score 2 = 12, score 3 = 6, score 4 = 6, score 5 = 24.
- The one exact `[유의 사항]` suffix shared by all prompts is removed from
  each topic prompt and displayed once as common writing guidance.
- Each reviewer first gives independent integer scores from 1 through 5 for
  content, organization, and expression, with a separate optional reason for
  each axis.
- Only after those scores are saved does the app show rationale A and then
  rationale B. The API/model identity is blinded in the browser and retained
  only in the ignored result database.
- Each rationale receives three independent judgments, one each for content,
  organization, and expression. Every axis uses `appropriate`, `partial`, or
  `inappropriate`, plus a separate optional reason.

The default API source is candidate 1 from the completed local Terra batch.
The default learned source is the completed score-blind evaluation-prompt v2
rationale run. Alternate completed rationale JSONL files can be supplied with
the command-line options below without changing application code.

The two raw sources have different output contracts. All 2,400 API candidate-1
rows have `diagnosis` and `next_step` for every axis, while all 2,400 learned v2
rows are plain rationales whose frozen prompt forbids an improvement suggestion.
To avoid exposing that structural source cue, the app displays only API
`diagnosis` and deliberately omits `next_step`; it never fabricates a learned
suggestion.

The rationale-review help panel is bound to `llm_as_judge.txt` and adapts the
parts that apply to this human task: domain match, specificity, groundedness,
and strict treatment of generic, mixed-domain, or invented evidence. It does
not ask reviewers to reproduce the file's four separate 1--5 judge scores.

## Preflight and launch

Use the repository Python environment as-is; the app has no web-framework
dependency.

```bash
python humaneval/run.py --dry-run
python humaneval/run.py
```

The default listener is `127.0.0.1:8765`. For a remote reviewer, prefer an SSH
port forward rather than a public bind because the pages contain restricted
writing text:

```bash
ssh -L 8765:127.0.0.1:8765 USER@SERVER
```

Then open `http://127.0.0.1:8765`. Progress resumes from the first incomplete
phase after selecting the same reviewer name, including after a server restart.

To use a later selected model rationale run:

```bash
python humaneval/run.py \
  --model-rationales /ignored/path/rationales.train.jsonl \
  --model-rationales /ignored/path/rationales.validation.jsonl
```

Accepted rationale rows use `source_id` plus either `rationale`, `rationales`,
or `participant_output`. Each bundle must contain content, organization, and
expression. API-style `diagnosis` and `next_step` objects and plain rationale
strings are both supported.

## Results and export

The default result store is the ignored file
`outputs/humaneval/responses.sqlite3`. It contains reviewer responses,
timestamps, blind-source mappings, hidden selection bands, and opaque source
IDs, but no essay, prompt, or rationale text.

Export the saved response rows to another ignored path with:

```bash
python humaneval/run.py \
  --export-jsonl outputs/humaneval/responses.jsonl
```

The study fingerprint covers item order and hashes of all displayed inputs. A
restart with different inputs fails closed instead of silently mixing studies
in one database.

## Network and Mac deployment boundary

The reviewer name selector is not authentication. Keep the default loopback
listener unless access is protected by SSH or another approved private network.
Do not expose the app through public router forwarding.

A Git checkout on a Mac contains only code and aggregate documentation, not the
restricted inputs. Moving even the selected 20 writings and rationales to a Mac
requires explicit data-transfer authorization and an approved secure transfer
method. Prefer a minimal study bundle over copying complete evaluation files;
never add that bundle or Mac response databases to Git.
