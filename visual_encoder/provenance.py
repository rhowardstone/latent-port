"""Self-describing run metadata (external review A11).

Every experiment JSON should embed enough to reproduce it: the full CLI args
(so e.g. warmstart_eps is never silently lost), the git SHA, package versions,
and a timestamp. This is the systemic fix for the class of provenance gaps that
produced the withdrawn 77.7% result.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "") if sha else "unknown"
    except Exception:
        return "unknown"


def _versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "numpy", "datasets"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "unavailable"
    return versions


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def provenance(args) -> dict:
    """Full self-describing metadata block to embed in a result JSON."""
    return {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "cli_args": {k: _jsonable(v) for k, v in vars(args).items()},
        "versions": _versions(),
    }
