from __future__ import annotations

import argparse
import atexit
import os
import sys
from pathlib import Path
from typing import IO, TextIO

import uvicorn


class _TeeTextIO(TextIO):
    """Write to multiple text streams (console + log file)."""

    __slots__ = ("_streams",)

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTG backend API server.")
    parser.add_argument(
        "--profiling-enabled",
        action="store_true",
        help="Print per-request profiling events and timings to stdout.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Append stdout and stderr (uvicorn, profiling, etc.) to this file.",
    )
    args = parser.parse_args()
    if args.profiling_enabled:
        os.environ["PROFILING_ENABLED"] = "1"
    if args.log_file is not None:
        log_path = args.log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp: IO[str] = log_path.open("a", encoding="utf-8")
        atexit.register(log_fp.close)
        sys.stdout = _TeeTextIO(sys.__stdout__, log_fp)
        sys.stderr = _TeeTextIO(sys.__stderr__, log_fp)
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("backend.main:app", host=host, port=port)
