#!/usr/bin/env python3
"""Run the installed vLLM CLI while exposing a descriptive process title."""
from __future__ import annotations

import sys

from setproctitle import setproctitle
from vllm.entrypoints.cli.main import main


def entrypoint() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: run_named_vllm.py PROCESS_TITLE VLLM_COMMAND [ARGS ...]")
    process_title = sys.argv[1].strip()
    if not process_title:
        raise SystemExit("process title must be nonblank")
    setproctitle(process_title[:255])
    sys.argv = ["vllm", *sys.argv[2:]]
    main()


if __name__ == "__main__":
    entrypoint()
