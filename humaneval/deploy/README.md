# Remote deployment runbook

This runbook publishes the authenticated application through a stable HTTPS
hostname while keeping the Python origin on `127.0.0.1:8765`. It does not
authorize moving restricted writing data to a new machine. Prefer running on
the already approved, always-on host that contains the inputs.

## Prerequisites

- Python 3.12 from the repository's declared environment.
- An always-on host with the approved restricted inputs and an ignored
  `outputs/humaneval/` state directory.
- A domain active on Cloudflare and `cloudflared` installed from the official
  Cloudflare package.
- Explicit approval before transferring any selected writing or rationale to a
  different host.

Cloudflare Tunnel makes outbound-only connections; do not open TCP 8765 on the
host firewall or router. The login page is publicly reachable, but no study
content is returned until the shared password succeeds.

## 1. Create authentication hashes

Run this directly in a private terminal on the deployment host:

```bash
python humaneval/run.py --generate-auth-config
```

The password prompts do not echo. The resulting ignored file is
`outputs/humaneval/auth.json`, mode `0600`. Do not paste secrets into a shell
command, service unit, GitHub issue, or tracked file.

## 2. Validate the frozen study

```bash
python humaneval/run.py --dry-run
PYTHONPATH=. python -m unittest discover -v humaneval/tests
```

Record the printed study fingerprint and deployed Git commit before collecting
responses.

## 3. Create a named tunnel and DNS route

Follow Cloudflare's current named-tunnel documentation. The equivalent CLI
flow is:

```bash
cloudflared tunnel login
cloudflared tunnel create mal2026-humaneval
cloudflared tunnel route dns mal2026-humaneval human-eval.example.com
```

Copy `cloudflared.example.yml` to the host's protected Cloudflare configuration
directory. Replace the hostname, tunnel UUID, and credentials path there. The
credentials JSON is a secret and must never be copied into this repository.

Start the application first:

```bash
python humaneval/run.py \
  --public-origin https://human-eval.example.com
```

Then start the named tunnel with its protected configuration:

```bash
cloudflared tunnel --config /etc/cloudflared/config.yml run mal2026-humaneval
```

Use the operating system's service manager for both long-running processes.
`humaneval.service.example` is a hardened systemd starting point; replace every
placeholder before installing it. Cloudflare also documents installing
`cloudflared` as a system service.

### Mac mini sleep and automatic startup

The display may sleep, but system sleep pauses both the Python server and the
tunnel. The tracked `com.mal2026.humaneval.plist.example` launch-agent template
wraps the server in `/usr/bin/caffeinate -s`, which prevents idle system sleep
while the service is healthy and the Mac mini is on AC power. It also restarts
the server after a crash and starts it again at user login.

Copy the template outside the repository, replace every placeholder, and load
it only after the authentication file and real input paths are ready:

```bash
cp humaneval/deploy/com.mal2026.humaneval.plist.example \
  "$HOME/Library/LaunchAgents/com.mal2026.humaneval.plist"
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.mal2026.humaneval.plist"
launchctl kickstart -k "gui/$(id -u)/com.mal2026.humaneval"
```

`cloudflared service install` separately registers the tunnel with launchd.
Do not use Quick Tunnels for the actual study because their hostname is
temporary. A named tunnel and stable HTTPS hostname are required.

## 4. External smoke test

Use an authorized test account and synthetic or approved pilot inputs.

1. Open the HTTPS URL on a mobile hotspot, not the server's local network.
2. Confirm an incorrect password returns a generic denial.
3. Confirm the correct common password and selected name open that reviewer's
   progress.
4. Save one response, reload, log in again, and confirm progress resumes.
5. Confirm `http://SERVER_IP:8765` is unreachable from another device.
6. Confirm `curl https://HOST/healthz` returns only `{"status":"ok"}`.
7. Confirm no essay, prompt, rationale, or password appears in service
   logs, the auth JSON, or the response database.

## Operations

- Back up SQLite with its online backup API or the SQLite `.backup` command;
  do not copy a live WAL database as unrelated files.
- Keep `auth.json`, `responses.sqlite3`, backups, tunnel credentials, and any
  minimal study bundle owner-readable only.
- Rotating credentials requires stopping the app, running
  `--generate-auth-config --replace-auth-config`, and restarting it. Responses
  are unaffected; all sessions are invalidated by the restart.
- At study completion, export under ignored `outputs/`, record a checksum,
  stop the two services, remove the DNS route/tunnel, and follow the approved
  retention schedule for restricted inputs and authentication hashes.

Official references:

- <https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/>
- <https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/>
