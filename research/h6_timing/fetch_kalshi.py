"""Download Kalshi NBA game-winner markets and their pre-game price path (cached).

Why Kalshi: ESPN's odds API keeps no intraday history for past NBA games (the
`odds/{provider}/history/0/movement` endpoint returns count=0 for every past event, all
seasons, all providers; see probe_espn.py). Kalshi's public, unauthenticated market-data
API keeps hourly and 1-minute candlesticks (best bid / best ask / last trade) for settled
markets. Series KXNBAGAME covers the 2024-25 play-in + playoffs and all of 2025-26.

Output (research/data/h6_timing/kalshi/):
  markets.json              every KXNBAGAME market (trimmed)
  candles/<ticker>_h.json   hourly candles, tip-60h .. tip
  candles/<ticker>_m.json   1-minute candles, tip-3h .. tip
Tip-off = ESPN scheduled start (research/data/espn_<season>.csv, date_utc).
Rate: 2 requests/second overall (Kalshi returns 429 above that on /historical). Reruns skip
cached files and retry anything that failed.
"""
import glob
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.engine.teams import from_espn  # noqa: E402

OUT = ROOT / "research/data/h6_timing/kalshi"
CANDLES = OUT / "candles"
CANDLES.mkdir(parents=True, exist_ok=True)
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNBAGAME"
RATE = 2.0  # historical endpoints answer 429 above ~2 requests/second
HOURLY_BACK_H = 60
MINUTE_BACK_H = 3

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (research; NBAPredictLab)"
_lock = threading.Lock()
_next = [0.0]
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _throttle():
    with _lock:
        now = time.time()
        wait = _next[0] - now
        _next[0] = max(now, _next[0]) + 1.0 / RATE
    if wait > 0:
        time.sleep(wait)


def get(url, params=None, tries=6):
    for i in range(tries):
        _throttle()
        try:
            r = S.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(10 * (i + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed: {url} {params}")


def list_markets():
    f = OUT / "markets.json"
    if f.exists():
        return json.loads(f.read_text())
    keep = ("ticker", "event_ticker", "yes_sub_title", "open_time", "close_time", "occurrence_datetime",
            "result", "volume_fp", "status", "title")
    rows = []
    for url, extra in ((f"{API}/historical/markets", {}), (f"{API}/markets", {"status": "settled"})):
        cur = ""
        while True:
            js = get(url, {"series_ticker": SERIES, "limit": 1000, "cursor": cur, **extra})
            rows += [{k: m.get(k) for k in keep} for m in js.get("markets", [])]
            cur = js.get("cursor")
            if not cur:
                break
    rows = list({r["ticker"]: r for r in rows}.values())
    f.write_text(json.dumps(rows))
    return rows


def espn_games():
    o = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "research/data/espn_*.csv"))])
    o["home"], o["away"] = o.home.map(from_espn), o.away.map(from_espn)
    o["tip"] = pd.to_datetime(o.date_utc, utc=True)
    o["game_date"] = o.tip.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return o


def match(markets):
    """One row per Kalshi event matched to an ESPN game (ET date + team pair)."""
    ev = {}
    for m in markets:
        parts = m["ticker"].split("-")
        if len(parts) != 3:
            continue
        code, team = parts[1], parts[2]
        ev.setdefault(m["event_ticker"], {"code": code, "markets": {}})["markets"][team] = m["ticker"]
    o = espn_games()
    idx = {(r.game_date, frozenset((r.home, r.away))): r for r in o.itertuples()}
    rows = []
    for et, e in ev.items():
        c = e["code"]
        try:
            d = datetime(2000 + int(c[:2]), MONTHS[c[2:5]], int(c[5:7]))
        except (KeyError, ValueError):
            continue
        teams = frozenset(e["markets"])
        if len(teams) != 2:
            continue
        g = None
        for shift in (0, -1, 1):
            key = ((d + pd.Timedelta(days=shift)).strftime("%Y-%m-%d"), teams)
            if key in idx:
                g = idx[key]
                break
        if g is None:
            continue
        rows.append({"event_ticker": et, "event_id": g.event_id, "season": g.season, "game_date": g.game_date,
                     "tip": g.tip.isoformat(), "home": g.home, "away": g.away,
                     "home_ticker": e["markets"][g.home], "away_ticker": e["markets"][g.away]})
    return pd.DataFrame(rows)


def _trim(c):
    return [{"t": x["end_period_ts"], "b": x["yes_bid"]["close"], "a": x["yes_ask"]["close"],
             "p": (x.get("price") or {}).get("close"), "v": x.get("volume")} for x in c]


def candles(ticker, tip_ts, kind):
    f = CANDLES / f"{ticker}_{kind}.json"
    if f.exists():
        return
    back, period = (HOURLY_BACK_H, 60) if kind == "h" else (MINUTE_BACK_H, 1)
    params = {"start_ts": tip_ts - back * 3600, "end_ts": tip_ts, "period_interval": period}
    js = get(f"{API}/historical/markets/{ticker}/candlesticks", params)
    if js is None:  # market not yet moved to the historical partition
        js = get(f"{API}/series/{SERIES}/markets/{ticker}/candlesticks", params) or {}
    f.write_text(json.dumps(_trim(js.get("candlesticks", []))))


def main():
    markets = list_markets()
    games = match(markets)
    games.to_csv(OUT / "events_matched.csv", index=False)
    print(f"{len(markets)} markets, {len(games)} events matched to ESPN games", flush=True)
    jobs = []
    for g in games.itertuples():
        tip_ts = int(pd.Timestamp(g.tip).timestamp())
        for tk in (g.home_ticker, g.away_ticker):
            for kind in ("h", "m"):
                if not (CANDLES / f"{tk}_{kind}.json").exists():
                    jobs.append((tk, tip_ts, kind))
    print(f"{len(jobs)} candle requests to make", flush=True)
    done = [0]

    failed = []

    def run(j):
        try:
            candles(*j)
        except RuntimeError:
            failed.append(j)
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(run, jobs))
    print(f"done ({len(failed)} failed requests; rerun to retry them)", flush=True)


if __name__ == "__main__":
    main()
