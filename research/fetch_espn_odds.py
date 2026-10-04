"""Download ESPN events + closing odds for past seasons (for backtesting)."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}&limit=100"
ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds"
SEASONS = {  # first day, last day
    "2021-22": (date(2021, 10, 19), date(2022, 6, 17)),
    "2022-23": (date(2022, 10, 18), date(2023, 6, 13)),
    "2023-24": (date(2023, 10, 24), date(2024, 6, 18)),
    "2024-25": (date(2024, 10, 22), date(2025, 6, 23)),
    "2025-26": (date(2025, 10, 21), date(2026, 6, 25)),
}


def get(url, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def day_events(d):
    js = get(SB.format(d.strftime("%Y%m%d"))) or {}
    rows = []
    for e in js.get("events", []):
        c = e["competitions"][0]
        if (e.get("season") or {}).get("type") not in (2, 3, 5):  # regular, post, play-in
            continue
        t = {x["homeAway"]: x for x in c["competitors"]}
        rows.append({
            "event_id": e["id"], "date_utc": e["date"], "season_type": e["season"]["type"],
            "home": t["home"]["team"]["abbreviation"], "away": t["away"]["team"]["abbreviation"],
            "home_score": t["home"].get("score"), "away_score": t["away"].get("score"),
            "completed": c["status"]["type"].get("completed"), "neutral": c.get("neutralSite"),
        })
    return rows


def ml(side):
    side = side or {}
    out = {"ml": side.get("moneyLine")}
    for k in ("open", "close"):
        v = (side.get(k) or {}).get("moneyLine") or {}
        out[f"ml_{k}"] = v.get("american") if isinstance(v, dict) else None
    return out


def event_odds(eid):
    js = get(ODDS.format(eid)) or {}
    items = js.get("items", [])
    if not items:
        return {"event_id": eid}
    # prefer consensus-ish books in this order
    pref = ["ESPN BET", "DraftKings", "Caesars", "BetMGM", "FanDuel", "Betfair"]
    items.sort(key=lambda it: next((i for i, p in enumerate(pref) if p.lower() in (it.get("provider", {}).get("name", "").lower())), 99))
    it = items[0]
    h, a = ml(it.get("homeTeamOdds")), ml(it.get("awayTeamOdds"))
    return {"event_id": eid, "book": it.get("provider", {}).get("name"), "spread": it.get("spread"),
            "total": it.get("overUnder"), "home_ml": h["ml"], "away_ml": a["ml"],
            "home_ml_open": h["ml_open"], "away_ml_open": a["ml_open"],
            "home_ml_close": h["ml_close"], "away_ml_close": a["ml_close"], "n_books": len(items),
            "raw": json.dumps([{"p": x.get("provider", {}).get("name"), "s": x.get("spread"),
                                 "h": (x.get("homeTeamOdds") or {}).get("moneyLine"),
                                 "a": (x.get("awayTeamOdds") or {}).get("moneyLine")} for x in items])}


for season, (start, end) in SEASONS.items():
    f = OUT / f"espn_{season}.csv"
    if f.exists():
        continue
    days = [start + timedelta(i) for i in range((end - start).days + 1)]
    with ThreadPoolExecutor(8) as ex:
        ev = [r for rows in ex.map(day_events, days) for r in rows]
    ev = pd.DataFrame(ev).drop_duplicates("event_id")
    with ThreadPoolExecutor(8) as ex:
        od = pd.DataFrame(list(ex.map(event_odds, ev.event_id)))
    df = ev.merge(od, on="event_id", how="left")
    df["season"] = season
    df.to_csv(f, index=False)
    print(season, len(df), "with ml:", df.home_ml.notna().sum(), flush=True)
