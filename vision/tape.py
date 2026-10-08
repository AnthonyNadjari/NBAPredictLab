#!/usr/bin/env python3
"""
Market + injury tape (server timer nbatape, every 3 min; stdlib only, no browser).

Question it answers later: when an injury is announced, how long until the prices move?
If we see the news before the market prices it, that delay is the only edge left (research H7b:
our model beat the ESPN open only where the open predated news).

Each run appends to TAPE_DIR/<UTC date>/:
  kalshi.csv    every open KXNBAGAME market: ts, event, ticker, team, yes bid/ask, last, volume
  injuries.csv  ESPN injury report, only CHANGES since the previous run (new player, new status,
                new comment): ts, team, player, status, short comment
Old days are gzipped after 2 days. Analyse with research code later (nothing is pushed to git).
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

TAPE_DIR = Path(os.getenv("NBA_TAPE_DIR", "/opt/nba-vision/tape"))
KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXNBAGAME&status=open&limit=200"
INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
UA = {"User-Agent": "NBAPredictLab-tape/1.0 (+https://github.com/AnthonyNadjari/NBAPredictLab)"}


def _get(url: str) -> dict | None:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"tape: {url.split('?')[0]} failed ({e})", flush=True)
            time.sleep(2 + 3 * attempt)
    return None


def _append(path: Path, header: list[str], rows: list[list]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def kalshi_rows(ts: str) -> list[list]:
    rows, cursor = [], ""
    for _ in range(10):
        js = _get(KALSHI + (f"&cursor={cursor}" if cursor else ""))
        if not js:
            break
        for m in js.get("markets", []):
            rows.append([ts, m.get("event_ticker"), m.get("ticker"), m.get("yes_sub_title") or m.get("subtitle"),
                         m.get("yes_bid_dollars"), m.get("yes_ask_dollars"), m.get("last_price_dollars"),
                         m.get("volume"), m.get("expected_expiration_time")])
        cursor = js.get("cursor") or ""
        if not cursor:
            break
    return rows


def injury_changes(ts: str, state_file: Path) -> list[list]:
    js = _get(INJURIES)
    if not js:
        return []
    try:
        seen = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        seen = {}
    now, rows = {}, []
    for team in js.get("injuries", []):
        tname = team.get("displayName")
        for inj in team.get("injuries", []):
            ath = inj.get("athlete") or {}
            pid = str(ath.get("id") or inj.get("id"))
            status = (inj.get("status") or (inj.get("type") or {}).get("description") or "").strip()
            comment = (inj.get("shortComment") or "").strip()
            key = f"{status}|{comment[:120]}"
            now[pid] = key
            if seen.get(pid) != key:
                rows.append([ts, tname, ath.get("displayName"), status, comment[:300]])
    for pid in set(seen) - set(now):                  # removed from the report = back / cleared
        rows.append([ts, "", pid, "removed", ""])
    state_file.write_text(json.dumps(now), encoding="utf-8")
    return rows


def compress_old_days(keep_days: int = 2) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    for day in TAPE_DIR.glob("20??-??-??"):
        if day.name < cutoff:
            for f in day.glob("*.csv"):
                with open(f, "rb") as src, gzip.open(f.with_suffix(".csv.gz"), "wb") as dst:
                    shutil.copyfileobj(src, dst)
                f.unlink()


def main() -> int:
    now = datetime.now(timezone.utc)
    ts = now.isoformat(timespec="seconds")
    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    day = TAPE_DIR / now.strftime("%Y-%m-%d")
    k = kalshi_rows(ts)
    _append(day / "kalshi.csv", ["ts", "event", "ticker", "team", "yes_bid", "yes_ask", "last", "volume", "expected_end"], k)
    i = injury_changes(ts, TAPE_DIR / "injuries_state.json")
    _append(day / "injuries.csv", ["ts", "team", "player", "status", "comment"], i)
    print(f"tape {ts}: {len(k)} kalshi markets, {len(i)} injury changes", flush=True)
    try:                                   # injury alert drafts (vision/alerts.py), never fatal
        import alerts
        alerts.main()
    except Exception as e:
        print(f"tape: alerts failed ({e})", flush=True)
    compress_old_days()
    return 0


if __name__ == "__main__":
    sys.exit(main())
