#!/usr/bin/env python3
"""Summary of the server's tape for the control panel's "Veille" tab -> docs/vision/tape.json.

Injury-report changes (last 3 days), injury alert drafts, tipster picks (last 2 days) with the most
backed picks. Stdlib only. Prints "changed" when the file content changed (the caller commits then).

    python vision/tools/export_tape.py
"""
import csv
import gzip
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

TAPE = Path(os.getenv("NBA_TAPE_DIR", "/opt/nba-vision/tape"))
OUT = Path(__file__).resolve().parents[2] / "docs" / "vision" / "tape.json"


def _days(n):
    now = datetime.now(timezone.utc)
    return [(now - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(n)]


def _csv(day, name):
    for f in (TAPE / day / name, TAPE / day / (name + ".gz")):
        if f.exists():
            op = gzip.open if f.suffix == ".gz" else open
            with op(f, "rt", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
    return []


def main():
    injuries = []
    for day in _days(3):
        injuries += [r for r in _csv(day, "injuries.csv") if r.get("status") != "removed" and r.get("team")]
    injuries.sort(key=lambda r: r["ts"], reverse=True)
    alerts = []
    f = TAPE / "alerts" / "drafts.jsonl"
    if f.exists():
        alerts = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()][-30:][::-1]
    picks = []
    for day in _days(2):
        f = TAPE / "tipsters" / f"{day}.jsonl"
        if f.exists():
            picks += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    picks.sort(key=lambda p: p.get("seen_at") or "", reverse=True)
    top = Counter((p.get("pick") or "").strip().lower() for p in picks if p.get("market") in ("moneyline", "spread", "total"))
    data = {
        "injuries": [{k: r.get(k) for k in ("ts", "team", "player", "status", "comment")} for r in injuries[:150]],
        "alerts": [{k: a.get(k) for k in ("news_at", "player", "team", "opp", "before", "after", "text")} for a in alerts],
        "tipsters": [{k: p.get(k) for k in ("seen_at", "author", "likes", "game", "pick", "market", "line", "odds", "units", "tweet_id")}
                     for p in picks[:150]],
        "tipster_consensus": [{"pick": k, "count": n} for k, n in top.most_common(10) if k and n >= 2],
    }
    new = json.dumps(data, indent=1, ensure_ascii=False)
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if json.loads(old or "{}") != data:
        OUT.write_text(new, encoding="utf-8")
        print("changed")
    else:
        print("same")


if __name__ == "__main__":
    main()
