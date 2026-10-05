#!/usr/bin/env python3
"""Thread publishing queue, read by the server.

The control panel's "Publier" button runs publish_thread.yml, which appends
{"id", "game_id", "texts", "dry", "requested_at"} to docs/vision/publish_queue.json.

  publish_queue.py next TEXTS_OUT   -> prints "<request id> <game id> <dry>" for the oldest
                                       request not handled yet (and writes its edited texts,
                                       if any, to TEXTS_OUT), or "none". Each request is marked
                                       handled BEFORE posting: a crash never posts twice.
  publish_queue.py log ID GAME RESULT_JSON
                                    -> appends the outcome to docs/vision/publish_log.json,
                                       which the control panel shows.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LOCAL_STATE_DIR, REPO_ROOT  # noqa: E402

QUEUE = REPO_ROOT / "docs" / "vision" / "publish_queue.json"
LOG = REPO_ROOT / "docs" / "vision" / "publish_log.json"
DONE = LOCAL_STATE_DIR / "handled_publish.txt"
MAX_AGE = timedelta(hours=12)        # a forgotten request must not post a stale thread
GAME_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _fresh(req: dict, now: datetime) -> bool:
    try:
        t = datetime.fromisoformat(str(req.get("requested_at")).replace("Z", "+00:00"))
    except ValueError:
        return False
    return now - t <= MAX_AGE


def next_request(texts_out: Path, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    for req in _read_json(QUEUE, {}).get("requests", []):
        rid, game = str(req.get("id") or ""), str(req.get("game_id") or "")
        if not rid or rid in done or not GAME_ID_RE.match(game) or not _fresh(req, now):
            continue
        DONE.parent.mkdir(parents=True, exist_ok=True)
        DONE.write_text("\n".join(sorted(done | {rid})[-500:]))
        texts = req.get("texts")
        ok_texts = isinstance(texts, list) and all(isinstance(t, str) for t in texts)
        texts_out.write_text(json.dumps(texts, ensure_ascii=False) if ok_texts else "", encoding="utf-8")
        return f"{rid} {game} {'true' if req.get('dry') is True else 'false'}"
    return "none"


def log_result(rid: str, game: str, result_path: Path, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    res = _read_json(result_path, {}) or {}
    posted, dry = int(res.get("posted") or 0), bool(res.get("dry_run"))
    # a real thread is "ok" once its opener is live (a stopped thread keeps its error message)
    ok = posted > 0 and (not dry or not res.get("error"))
    entry = {
        "id": rid, "game_id": game, "at": now.isoformat(timespec="seconds"),
        "dry_run": dry, "ok": ok,
        "posted": posted, "total": res.get("total"),
        "url": f"https://x.com/NBAPredictLab/status/{res['first_id']}"
               if res.get("first_id") and not res.get("dry_run") else None,
        "error": res.get("error") or (None if posted else "nothing was posted (see the server log)"),
    }
    log = [e for e in _read_json(LOG, []) if isinstance(e, dict) and e.get("id") != rid]
    LOG.write_text(json.dumps((log + [entry])[-40:], indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "next":
        print(next_request(Path(argv[2])))
        return 0
    if len(argv) == 5 and argv[1] == "log":
        print(json.dumps(log_result(argv[2], argv[3], Path(argv[4]))))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
