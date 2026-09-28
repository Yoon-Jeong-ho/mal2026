# Cloudflare human-validation source

This is an alternative backend for the same frozen 20-item, score-then-A-then-B
study. It shares `../web/` with the Python server. It does not proxy the Python
app: the Worker uses D1 for study content, responses and sessions.

The Python server uses a local scrypt password file. The Worker instead uses
server-only `PASSWORD_PEPPER` and `PASSWORD_DIGEST` secrets (HMAC-SHA256), a
`TURNSTILE_SECRET`, the public Turnstile site key and hostname, and the three
rate-limit bindings in `wrangler.jsonc`. These are distinct authentication
configurations. Shared reviewer names are not individual identity verification.

## Preserved deployment state

`wrangler.jsonc` records the existing database identity and public site key;
these are not credentials. Its `workers_dev` and `preview_urls` flags are both
false. Checking in this source does not enable a public endpoint, deploy the
Worker, modify D1, change DNS, or rotate a password. Reopening collection or
moving study content needs the existing project's authorization first.

For an authorized deployment, use the pinned `package.json` and
`pnpm-lock.yaml` with an already approved environment. Bind the intended D1
database, approved HTTPS hostname/route, static assets, all rate limiters and
Turnstile settings. `schema.sql` defines the D1 tables. `export_seed.py` builds
an owner-readable seed under ignored `outputs/` from the approved inputs.
That seed contains restricted writing and rationale text and must stay out of
Git. Applying it to D1 is a data transfer, not a source-only operation.
`configure_password.py` prompts privately and writes the two password secrets
through Wrangler; it must only be run when credential setup/rotation is intended.

## Saved-response export

The historical export script is now versioned here instead of existing only
under ignored outputs. It issues SELECT queries to the configured remote D1
binding, validates completed response rows, and writes an owner-readable
JSONL, aggregate summary, manifest and checksums to a new output directory:

```bash
node humaneval/cloudflare/export_cloudflare_results.mjs \
  outputs/humaneval/NEW_EXPORT_DIRECTORY \
  humaneval/cloudflare /absolute/path/to/pnpm
```

Run from the repository root and use an ignored output directory. The pnpm
executable must resolve the already configured Wrangler environment. Exported
responses include reviewer names and free-text reasons; do not commit them.
The exporter excludes essay/rationale text and session/password material. It
rejects partial rows and existing output directories. The result is a
point-in-time snapshot, not proof that collection stopped or every reviewer
finished. It does not stop the Worker. Keep existing snapshots unchanged.

## Offline validation

```bash
node --test humaneval/cloudflare/tests/*.test.mjs
node --check humaneval/cloudflare/src/index.js
node --check humaneval/cloudflare/export_cloudflare_results.mjs
```

These tests use synthetic values and mock bindings. They do not access the
production Worker, D1, Turnstile or real responses. They do not establish a
successful live deployment. A Wrangler dry run needs its separately installed
pinned dependencies; do not infer it from the dependency-free tests.
