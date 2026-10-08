#!/usr/bin/env python3
"""
Injury alerts: "Embiid OUT -> Sixers' chances go from 62% to 47%" (DRAFT mode: nothing posted).

Runs right after vision/tape.py (same timer, every 3 min), stdlib only:
1. new "Out" lines in today's tape/injuries.csv for a KEY player (>= 26 min a game over his last
   10, from data/player_games.csv, same ESPN ids) whose team has an open Kalshi game market
   within 48 h -> a pending alert with the market price just before the news;
2. 12+ minutes later, the price after -> a draft tweet in tape/alerts/drafts.jsonl.
The drafts show how fast and how much the market moves; once they read well, the same text can
go through the publish queue.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.engine.teams import TEAMS  # noqa: E402  (pure Python, no dependency)

TAPE_DIR = Path(os.getenv("NBA_TAPE_DIR", "/opt/nba-vision/tape"))
OUT_DIR = TAPE_DIR / "alerts"
STATE = OUT_DIR / "state.json"
KEY_MINUTES = 26.0
WAIT_AFTER = timedelta(minutes=12)
KALSHI_ALIASES = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "PHO": "PHX", "BRK": "BKN",
                  "WSH": "WAS", "UTAH": "UTA", "NOR": "NOP"}
NAME_TO_CODE = {v.lower(): k for k, v in TEAMS.items()}
NICK = {k: v.split()[-1] if not v.endswith("76ers") else "76ers" for k, v in TEAMS.items()}
NICK["POR"] = "Trail Blazers"


def _ts(s):
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _rows(name: str, days: int = 2) -> list[dict]:
    out, now = [], datetime.now(timezone.utc)
    for d in range(days):
        day = TAPE_DIR / (now - timedelta(days=d)).strftime("%Y-%m-%d")
        for f in (day / name, day / (name + ".gz")):
            if f.exists():
                opener = gzip.open if f.suffix == ".gz" else open
                with opener(f, "rt", encoding="utf-8") as fh:
                    out += list(csv.DictReader(fh))
    return out


def key_players() -> dict[str, dict]:
    """ESPN player id -> {name, team, min, pts} for players averaging KEY_MINUTES+ over their last 10."""
    games: dict[str, list] = {}
    try:
        with open(REPO / "data" / "player_games.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("season_type") == "Regular Season" or r.get("season_type") == "Playoffs":
                    games.setdefault(r["player_id"], []).append(r)
    except OSError:
        return {}
    out = {}
    for pid, rows in games.items():
        rows.sort(key=lambda r: r["game_date"])
        last = rows[-10:]
        mins = [float(r["min"] or 0) for r in last]
        if len(last) >= 5 and sum(mins) / len(mins) >= KEY_MINUTES:
            out[pid] = {"name": last[-1]["player"], "team": last[-1]["team"], "min": round(sum(mins) / len(mins), 1),
                        "pts": round(sum(float(r["pts"] or 0) for r in last) / len(last), 1)}
    return out


def market(code: str, at: datetime, kalshi: list[dict]) -> dict | None:
    """Our team's Kalshi game market (next 48 h): {event, opp, price} with the mid at or before `at`."""
    best = None
    for r in kalshi:
        t = _ts(r["ts"])
        if t is None or t > at:
            continue
        suffix = r["ticker"].rsplit("-", 1)[-1]
        if KALSHI_ALIASES.get(suffix, suffix) != code:
            continue
        end = _ts(r.get("expected_end"))
        if end is None or not (at - timedelta(hours=6) < end < at + timedelta(hours=48)):
            continue
        try:
            mid = (float(r["yes_bid"]) + float(r["yes_ask"])) / 2
        except (TypeError, ValueError):
            continue
        if best is None or t >= best["t"]:
            best = {"t": t, "event": r["event"], "price": mid}
    if not best:
        return None
    pair = best["event"].rsplit("-", 1)[-1][7:]                  # e.g. 26OCT12SASUTA -> SASUTA
    opp = pair.replace(code, "", 1) if code in pair else pair
    best["opp"] = KALSHI_ALIASES.get(opp, opp)
    return best


TAGS = {"PHI": "Sixers", "GSW": "DubNation", "POR": "RipCity", "OKC": "ThunderUp", "LAL": "LakeShow",
        "BOS": "DifferentHere", "NYK": "Knicks", "MIA": "HEATCulture"}
MONTHS = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}


def _game_day(event: str) -> str:
    """'tonight' / 'tomorrow' / 'Oct 21' from a Kalshi event ticker (KXNBAGAME-26OCT21PHINYK)."""
    try:
        code = event.rsplit("-", 1)[-1][:7]                     # 26OCT21
        d = datetime(2000 + int(code[:2]), MONTHS[code[2:5]], int(code[5:7])).date()
        today = (datetime.now(timezone.utc) - timedelta(hours=5)).date()   # US Eastern, roughly
        return "tonight" if d == today else "tomorrow" if d == today + timedelta(days=1) else f"on {d:%b} {d.day}"
    except Exception:
        return "next game"


def alert_text(a: dict, after: float, mins: int) -> str:
    """The post: emoji-led lines, numbers first, one team hashtag + #NBA (under 280 chars)."""
    team, opp = NICK[a["team"]], NICK.get(a["opp"], a["opp"])
    before_p, after_p = round(100 * a["before"]), round(100 * after)
    move = after_p - before_p
    arrow = "📉" if move < 0 else "📈"
    big = "🔥 " if abs(move) >= 8 else ""
    tag = TAGS.get(a["team"], team.replace(" ", ""))
    return (f"🚨 INJURY ALERT\n\n"
            f"❌ {a['player']} is OUT {_game_day(a.get('event', ''))} vs the {opp}\n\n"
            f"📊 {a['pts']} PTS · {a['min']} MIN per game (last 10)\n\n"
            f"{arrow} {team} win chance: {before_p}% → {after_p}% {big}\n"
            f"⏱️ market moved {abs(move)} pts in {mins} min\n\n"
            f"#{tag} #NBA")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        state = {"last_ts": None, "pending": [], "done": []}
    kalshi, injuries = _rows("kalshi.csv"), _rows("injuries.csv")
    now = datetime.now(timezone.utc)
    last = _ts(state.get("last_ts")) or now - timedelta(minutes=10)
    keys = None
    for r in injuries:
        t = _ts(r["ts"])
        if not t or t <= last or (r.get("status") or "").lower() != "out":
            continue
        keys = keys if keys is not None else key_players()
        p = next((v | {"id": k} for k, v in keys.items() if v["name"] == r["player"]), None)
        code = NAME_TO_CODE.get((r.get("team") or "").lower())
        if not p or not code:
            continue
        tag = f"{p['id']}-{t:%Y-%m-%d}"
        if tag in state["done"]:
            continue
        m = market(code, t, kalshi)
        if not m:
            continue
        state["pending"].append({"tag": tag, "player": p["name"], "min": p["min"], "pts": p["pts"], "team": code,
                                 "opp": m["opp"], "event": m["event"], "before": round(m["price"], 3),
                                 "news_at": t.isoformat(timespec="seconds"), "comment": (r.get("comment") or "")[:200]})
        state["done"].append(tag)
    keep = []
    for a in state["pending"]:
        t0 = _ts(a["news_at"])
        if now - t0 < WAIT_AFTER:
            keep.append(a)
            continue
        m = market(a["team"], now, kalshi)
        if not m:
            continue
        after = round(m["price"], 3)
        mins = int((now - t0).total_seconds() // 60)
        text = alert_text(a, after, mins)
        with open(OUT_DIR / "drafts.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**a, "after": after, "measured_at": now.isoformat(timespec="seconds"),
                                 "text": text}, ensure_ascii=False) + "\n")
        print("alert draft:", text.replace("\n", " "), flush=True)
    state["pending"] = keep
    state["done"] = state["done"][-2000:]
    state["last_ts"] = max((r["ts"] for r in injuries), default=state.get("last_ts"))
    STATE.write_text(json.dumps(state), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
