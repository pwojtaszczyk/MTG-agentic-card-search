from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATABASE = _REPO_ROOT / "database" / "all-cards.db"


def get_database_path() -> Path:
    """Resolve the SQLite database path from ``DATABASE_PATH`` or the package default."""
    return Path(os.getenv("DATABASE_PATH", str(DEFAULT_DATABASE)))
