#!/usr/bin/env python3
"""Run the local MAL2026 human score/rationale validation web application."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from humaneval.auth import (  # noqa: E402
    AuthConfigError,
    load_auth_config,
    prompt_and_write_auth_config,
)
from humaneval.core import ResponseStore, axis_band_counts, build_study  # noqa: E402
from humaneval.server import HumanValidationServer  # noqa: E402


DEFAULT_API = ROOT / "data/processed/restricted/openai_rationale_batches/openai-rationale-terra-full-20260719-001/candidates.jsonl"
DEFAULT_MODEL_ROOT = ROOT / "data/processed/restricted/evaluation_prompt_rationale_v2/evaluation-prompt-rationale-generation-v2-score-blind-20260729-004"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "eval/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "eval/validation.jsonl")
    parser.add_argument("--rubric", type=Path, default=ROOT / "evaluation.txt")
    parser.add_argument("--judge-guide", type=Path, default=ROOT / "llm_as_judge.txt")
    parser.add_argument("--api-rationales", type=Path, action="append", default=None)
    parser.add_argument("--model-rationales", type=Path, action="append", default=None)
    parser.add_argument("--api-candidate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--database", type=Path, default=ROOT / "outputs/humaneval/responses.sqlite3")
    parser.add_argument(
        "--auth-config",
        type=Path,
        default=ROOT / "outputs/humaneval/auth.json",
        help="ignored authentication hash file",
    )
    parser.add_argument(
        "--generate-auth-config",
        action="store_true",
        help="prompt for the shared password, then exit",
    )
    parser.add_argument(
        "--replace-auth-config",
        action="store_true",
        help="allow --generate-auth-config to replace an existing hash file",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--public-origin",
        help="exact public HTTPS origin, for example https://human-eval.example.com",
    )
    parser.add_argument(
        "--insecure-local-http",
        action="store_true",
        help="local browser smoke test only; never combine with an Internet tunnel",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print aggregate selection evidence only")
    parser.add_argument("--export-jsonl", type=Path, help="export saved responses and exit")
    return parser.parse_args()


def require_ignored_result_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    ignored_root = (ROOT / "outputs").resolve()
    if not resolved.is_relative_to(ignored_root):
        raise SystemExit(f"result artifacts must stay under ignored outputs/: {resolved}")
    return resolved


def normalized_public_origin(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit(
            "--public-origin must be one exact HTTPS origin without a path, "
            "for example https://human-eval.example.com"
        )
    return f"https://{parsed.netloc.lower()}"


def main() -> None:
    args = parse_args()
    if args.replace_auth_config and not args.generate_auth_config:
        raise SystemExit("--replace-auth-config is valid only with --generate-auth-config")
    if args.generate_auth_config:
        auth_path = require_ignored_result_path(args.auth_config)
        try:
            written = prompt_and_write_auth_config(
                auth_path, replace=args.replace_auth_config
            )
        except AuthConfigError as exc:
            raise SystemExit(f"authentication configuration failed: {exc}") from exc
        print(f"authentication hashes saved with owner-only permissions: {written}")
        print("No plaintext password was written to disk.")
        return

    api_paths = args.api_rationales or [DEFAULT_API]
    model_paths = args.model_rationales or [
        DEFAULT_MODEL_ROOT / "rationales.train.jsonl",
        DEFAULT_MODEL_ROOT / "rationales.validation.jsonl",
    ]
    study = build_study(
        split_paths={"train": args.train, "validation": args.validation},
        rubric_path=args.rubric,
        judge_guide_path=args.judge_guide,
        api_rationale_paths=api_paths,
        model_rationale_paths=model_paths,
        seed=args.seed,
        api_candidate=args.api_candidate,
    )
    counts = axis_band_counts(study.items)
    print(f"study fingerprint: {study.fingerprint}")
    print(f"selected items: {len(study.items)}; hidden per-axis target-band counts: {counts}")
    print(f"common prompt notices: 1; reviewer names: 4")
    if args.dry_run:
        print("preflight passed")
        return

    database = require_ignored_result_path(args.database)
    store = ResponseStore(database, study)
    if args.export_jsonl is not None:
        output = require_ignored_result_path(args.export_jsonl)
        count = store.export_jsonl(output)
        print(f"exported {count} response rows to {output}")
        return

    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        raise SystemExit(
            "the human-validation origin must stay on loopback; point the "
            "authenticated tunnel to 127.0.0.1 instead of binding publicly"
        )
    auth_path = require_ignored_result_path(args.auth_config)
    try:
        auth_config = load_auth_config(auth_path)
    except AuthConfigError as exc:
        raise SystemExit(f"authentication configuration failed: {exc}") from exc

    if args.public_origin and args.insecure_local_http:
        raise SystemExit("choose either --public-origin or --insecure-local-http, not both")
    if args.public_origin:
        allowed_origins = {normalized_public_origin(args.public_origin)}
        secure_cookie = True
    elif args.insecure_local_http:
        display_host = "127.0.0.1" if args.host == "localhost" else args.host
        if display_host == "::1":
            display_host = "[::1]"
        allowed_origins = {f"http://{display_host}:{args.port}"}
        secure_cookie = False
        print("WARNING: insecure local HTTP mode is for loopback smoke tests only.")
    else:
        raise SystemExit(
            "launch requires --public-origin https://...; for a loopback-only "
            "browser smoke test, use --insecure-local-http"
        )

    static_root = ROOT / "humaneval/web"
    server = HumanValidationServer(
        (args.host, args.port),
        store,
        static_root,
        auth_config=auth_config,
        allowed_origins=allowed_origins,
        secure_cookie=secure_cookie,
    )
    host, port = server.server_address[:2]
    print(f"response database: {database}")
    print(f"listening on http://{host}:{port}")
    print(f"accepted browser origin: {next(iter(allowed_origins))}")
    print("Keep this listener on loopback and route remote HTTPS traffic through the tunnel.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
