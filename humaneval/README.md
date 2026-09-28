# Authenticated human score and rationale validation app

This application supports a fixed, source-blind review of 20 writings
from the restricted `eval/train.jsonl` and `eval/validation.jsonl` inputs.
Writing text, prompts, and rationales are never copied into tracked files.
The Python app reads them at runtime and excludes them from its response SQLite
database. The separate Cloudflare implementation stores the selected study
content in its access-controlled D1 database; see [its runbook](cloudflare/README.md). Remote access uses one shared
entrance password; reviewers select their own name after entering the site.

## Folder layout

Everything needed to maintain the interface is isolated here:

```text
humaneval/
├── README.md          # operations, privacy, and study contract
├── auth.py            # password hashing and ignored auth configuration
├── run.py             # preflight, launch, and export CLI
├── core.py            # selection, input validation, and SQLite persistence
├── server.py          # authenticated dependency-free HTTP server
├── deploy/            # tunnel and service examples; contains no secrets
├── cloudflare/        # alternative Worker/D1 backend and export utilities
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
```

Before the first launch, create the ignored authentication file interactively:

```bash
python humaneval/run.py --generate-auth-config
```

The command prompts, without terminal echo, for one shared entrance password
of at least 6 characters, entered twice.

It writes only salted scrypt hashes to the ignored, owner-readable file
`outputs/humaneval/auth.json`. The plaintext password is not written to disk or
passed as a command-line argument. Give each reviewer the public URL and the
shared password, and ask each person to select their own name.

For a loopback-only browser smoke test:

```bash
python humaneval/run.py --insecure-local-http
```

Open `http://127.0.0.1:8765`. Never attach an Internet tunnel while using this
explicit insecure mode. For the remote HTTPS deployment:

```bash
python humaneval/run.py \
  --public-origin https://human-eval.example.com
```

Replace the example origin with the exact tunnel hostname. The listener stays
on `127.0.0.1:8765`; the application refuses a non-loopback bind. See
`humaneval/deploy/README.md` for the Cloudflare Tunnel and service setup.

To rotate the shared password, stop the app and explicitly replace the
authentication hash file:

```bash
python humaneval/run.py \
  --generate-auth-config \
  --replace-auth-config
```

Restarting after rotation invalidates all in-memory sessions but does not alter
saved evaluation responses. Progress resumes from the first incomplete phase
after the same reviewer authenticates again.

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

The shared password gates every endpoint that returns study content or saves a
response. Sessions expire after 12 hours, mutation requests
require a per-session CSRF token, and five failed logins from one client block
new attempts for 15 minutes. The public login page and reviewer names are not
secret. A shared password cannot identify who disclosed it, so rotate it
immediately if it is forwarded beyond the four reviewers.

Keep the application listener on loopback and use an outbound HTTPS tunnel; do
not expose port 8765 or configure public router forwarding. The public hostname
must use HTTPS because the production session cookie is `Secure`, `HttpOnly`,
and `SameSite=Strict`.

A Git checkout on a Mac contains only code and aggregate documentation, not the
restricted inputs. Moving even the selected 20 writings and rationales to a Mac
requires explicit data-transfer authorization and an approved secure transfer
method. Prefer a minimal study bundle over copying complete evaluation files;
never add that bundle or Mac response databases to Git.

Authentication limits who can load the data but does not prevent an authorized
reviewer from taking a screenshot or copying visible text. Because reviewers
share one password, it also does not prevent someone from choosing another
reviewer's name. Reviewer instructions
and data-handling authorization remain part of the study's operational boundary.
