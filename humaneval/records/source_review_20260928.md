# Human-validation source preservation — 2026-09-28

Previously local authentication, UI and Cloudflare sources were integrated on
main `3ec85129af69e3e0dc98416f40c45ce4f74afe27`. The eight modified tracked
files were merged against their original base; there were no text conflicts.
Upstream training, deployment and policy changes were preserved. Frozen study
selection, scoring and rationale order were not changed.

The historical D1 result exporter, previously under ignored outputs, is now
versioned at `humaneval/cloudflare/export_cloudflare_results.mjs` with identical
source bytes. The Cloudflare runbook documents its point-in-time export
semantics and distinguishes Worker/D1 storage and authentication from the
Python/SQLite backend. No restricted input, response, seed SQL, credential,
checkpoint, binary dependency or live operational log is included.

Validation: 11 existing Python tests passed (eight in the ordinary sandbox and
three localhost HTTP tests on a permitted socket retry); eight Node tests
passed, including four added Worker endpoint rejection/configuration tests.
JavaScript syntax checks, Python AST parsing and the D1 schema in an in-memory
SQLite database passed. An offline exporter fixture with one synthetic
completed response passed summary, SHA256 and overwrite-rejection checks.
Tests do not contact production D1 or Turnstile. A Wrangler build/deploy and a
live browser acceptance test were not run. Existing deployment enablement flags
and credentials were unchanged; publication of source is not a deployment.

Backup verification: five local input/rationale files and all four files of
the 2026-08-09 Cloudflare response snapshot match server22 byte for byte. The
snapshot contains 40 completed response rows and explicitly does not prove
that collection stopped. Local and server Python SQLite copies both contain
zero responses and 20 study items, and pass integrity checks on disposable
copies, but their file and logical hashes differ; they are not interchangeable
backups. Local operational logs, auth state, generated seed and Mac tunnel
binaries were retained separately. This audit does not authorize deleting the
whole checkout or claim every ignored file was transferred.
