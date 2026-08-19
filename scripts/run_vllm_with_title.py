#!/usr/bin/env python3
"""Run the repository's vLLM CLI with an explicit project process title."""
from __future__ import annotations

import os

os.environ.setdefault("SPT_NOENV", "1")
import setproctitle
from vllm.entrypoints.cli.main import main


if __name__ == "__main__":
    setproctitle.setproctitle("(D)_vllm")
    raise SystemExit(main())
