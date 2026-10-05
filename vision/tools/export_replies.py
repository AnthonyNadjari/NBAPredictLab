#!/usr/bin/env python3
"""Copy the replies ledger (server state dir) to docs/vision/replies/ for the panel and the review."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replies_ledger  # noqa: E402
from config import REPO_ROOT  # noqa: E402

if __name__ == "__main__":
    if replies_ledger.LEDGER_DIR.exists():
        out = replies_ledger.export(REPO_ROOT / "docs" / "vision" / "replies")
        print(f"exported {len(out)} month file(s)")
