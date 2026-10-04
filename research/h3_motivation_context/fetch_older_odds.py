"""Download ESPN events + pre-game moneylines for 2018-19 .. 2020-21 (not covered by research/data/espn_*.csv).

Why: (1) more training seasons for sparse end-of-season context features, (2) an independent,
never-looked-at sample of playoff games to re-test the 'playoff favourites underperform' hint that
was found on 2022-23..2025-26.

ESPN keeps, for completed games, the last pre-game line of each provider ("current"); for these
seasons there is no explicit open/close split. We store every provider's moneyline in `raw` and choose
a book later (h3data.py). Polite: <= ~4 requests/second, everything cached (per-event JSONL cache so an
interrupted run resumes; per-season CSV when done).

    python research/h3_motivation_context/fetch_older_odds.py
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/data/h3_motivation_context"
OUT.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}&limit=100"
ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds"
SEASONS = {
    "2018-19": (date(2018, 10, 16), date(2019, 6, 14)),
    "2019-20": (date(2019, 10, 22), date(2020, 10, 12)),
    "2020-21": (date(2020, 12, 22), date(2021, 7, 21)),
}
_lock = threading.Lock()
_last = [0.0]
MIN_GAP = 0.26  # seconds between any two requests (global) -> < 4 req/s


def get(url, tries=4):
    for i in range(tries):
        with _lock:
            wait = _last[0] + MIN_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = S.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2.0 * (i + 1))
    return None


def day_events(d):
    js = get(SB.format(d.strftime("%Y%m%d"))) or {}
    rows = []
    for e in js.get("events", []):
        c = e["competitions"][0]
        if (e.get("season") or {}).get("type") not in (2, 3, 5):
            continue
        t = {x["homeAway"]: x for x in c["competitors"]}
        rows.append({
            "event_id": e["id"], "date_utc": e["date"], "season_type": e["season"]["type"],
            "home": t["home"]["team"]["abbreviation"], "away": t["away"]["team"]["abbreviation"],
            "home_score": t["home"].get("score"), "away_score": t["away"].get("score"),
            "completed": c["status"]["type"].get("completed"), "neutral": c.get("neutralSite"),
        })
    return rows


def event_odds(eid):
    js = get(ODDS.format(eid)) or {}
    items = js.get("items", [])
    return {"event_id": eid, "n_books": len(items),
            "raw": json.dumps([{"p": x.get("provider", {}).get("name"), "s": x.get("spread"),
                                "t": x.get("overUnder"),
                                "h": (x.get("homeTeamOdds") or {}).get("moneyLine"),
                                "a": (x.get("awayTeamOdds") or {}).get("moneyLine")} for x in items])}


def main():
    for season, (start, end) in SEASONS.items():
        f = OUT / f"espn_{season}.csv"
        if f.exists():
            print(season, "cached", flush=True)
            continue
        ev_cache = OUT / f"_events_{season}.csv"
        if ev_cache.exists():
            ev = pd.read_csv(ev_cache, dtype={"event_id": str})
        else:
            days = [start + timedelta(i) for i in range((end - start).days + 1)]
            with ThreadPoolExecutor(4) as ex:
                ev = [r for rows in ex.map(day_events, days) for r in rows]
            ev = pd.DataFrame(ev).drop_duplicates("event_id")
            ev.to_csv(ev_cache, index=False)
        print(season, "events", len(ev), flush=True)
        jl = OUT / f"_odds_{season}.jsonl"
        done = {}
        if jl.exists():
            for line in jl.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:  # truncated last line of an interrupted run
                    continue
                done[str(r["event_id"])] = r
        todo = [e for e in ev.event_id.astype(str) if e not in done]
        if jl.exists() and jl.stat().st_size and not jl.read_bytes().endswith(b"\n"):
            with open(jl, "a") as fh:
                fh.write("\n")
        with open(jl, "a") as fh, ThreadPoolExecutor(6) as ex:
            for i, r in enumerate(ex.map(event_odds, todo)):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[str(r["event_id"])] = r
                if i % 200 == 0:
                    print(season, f"odds {i}/{len(todo)}", flush=True)
        od = pd.DataFrame(list(done.values()))
        od["event_id"] = od.event_id.astype(str)
        df = ev.assign(event_id=ev.event_id.astype(str)).merge(od, on="event_id", how="left")
        df["season"] = season
        df.to_csv(f, index=False)
        print(season, len(df), "with books:", (df.n_books > 0).sum(), flush=True)


if __name__ == "__main__":
    main()
