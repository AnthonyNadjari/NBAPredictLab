#!/usr/bin/env python3
"""Print the pending request from the control panel, once: "<kind> <max_replies>".

The panel writes docs/vision/request.json ({"id", "kind": session|dry|login, "max_replies"});
ids already handled are remembered in the server's state dir so each request runs once.
Prints "none" when there is nothing to do.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LOCAL_STATE_DIR, REPO_ROOT  # noqa: E402

REQUEST = REPO_ROOT / "docs" / "vision" / "request.json"
DONE = LOCAL_STATE_DIR / "handled_requests.txt"


def main() -> int:
    try:
        req = json.loads(REQUEST.read_text(encoding="utf-8"))
    except Exception:
        print("none")
        return 0
    rid, kind = str(req.get("id") or ""), req.get("kind")
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    if not rid or rid in done or kind not in ("session", "dry", "login"):
        print("none")
        return 0
    DONE.write_text("\n".join(sorted(done | {rid})[-200:]))
    max_replies = req.get("max_replies")
    print(f"{kind} {int(max_replies) if isinstance(max_replies, int) and 0 < max_replies <= 100 else ''}".strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
