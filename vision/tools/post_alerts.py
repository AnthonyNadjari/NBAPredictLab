#!/usr/bin/env python3
"""Post injury alert drafts (vision/alerts.py) through the logged-in browser, from the publish tick.

On only when docs/autopilot.json has "injury_alerts": true. Each draft is posted at most once,
only if measured less than 45 min ago, at most 6 a day. Same request guard as threads
(vision/publisher.py). Prints "nothing" (no browser started) when there is nothing to post.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TAPE = Path(os.getenv("NBA_TAPE_DIR", "/opt/nba-vision/tape")) / "alerts"
REPO = Path(__file__).resolve().parents[2]
MAX_AGE = timedelta(minutes=45)
MAX_PER_DAY = 6


def due(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    try:
        if not json.loads((REPO / "docs" / "autopilot.json").read_text(encoding="utf-8")).get("injury_alerts"):
            return []
    except Exception:
        return []
    drafts = TAPE / "drafts.jsonl"
    if not drafts.exists():
        return []
    posted_file = TAPE / "posted.jsonl"
    posted = [json.loads(l) for l in posted_file.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if posted_file.exists() else []
    done = {p["tag"] for p in posted}
    today = sum(1 for p in posted if p.get("at", "")[:10] == now.strftime("%Y-%m-%d"))
    out = []
    for d in (json.loads(l) for l in drafts.read_text(encoding="utf-8").splitlines() if l.strip()):
        t = datetime.fromisoformat(d["measured_at"])
        if d["tag"] in done or now - t > MAX_AGE:
            continue
        if today + len(out) >= MAX_PER_DAY:
            break
        out.append(d)
    return out


def main() -> int:
    todo = due()
    if not todo or "--check" in sys.argv:
        print("nothing" if not todo else f"{len(todo)} to post")
        return 0
    from auth import launch_and_auth
    from publisher import post_thread
    res = launch_and_auth()
    if res[0] is None:
        print(f"alerts: not logged in ({res[-1]})")
        return 1
    pw, _, context, _page = res
    try:
        for d in todo:
            ids, err = post_thread(context, [{"text": d["text"], "image": None}])
            rec = {"tag": d["tag"], "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "tweet_id": ids[0] if ids else None, "error": err}
            with open(TAPE / "posted.jsonl", "a", encoding="utf-8") as fh:   # once, even on failure
                fh.write(json.dumps(rec) + "\n")
            print("alert posted:" if ids else "alert failed:", d["player"], ids or err, flush=True)
    finally:
        try:
            context.close()
            pw.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
