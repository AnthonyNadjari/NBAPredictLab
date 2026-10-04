"""Re-download the full per-book ESPN odds for every event (trimmed), cached on disk.

The original research/fetch_espn_odds.py only kept per-book spread + moneyline. The same
endpoint also exposes, per book, the spread juice (spreadOdds), the total, and for some
books/seasons the open/close moneyline and spread. We keep those fields here.

Output: research/data/h4_market_microstructure/books/<event_id>.json (one per event; reruns
skip cached files). Rate: ~4 requests/second overall (ESPN politeness limit is ~5/s).
"""
import glob
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/data/h4_market_microstructure/books"
OUT.mkdir(parents=True, exist_ok=True)
ODDS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}"
        "/competitions/{0}/odds?limit=50")
RATE = 4.5  # requests per second, global (ESPN politeness limit is ~5/s)

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"
_lock = threading.Lock()
_next = [0.0]


def _throttle():
    with _lock:
        now = time.time()
        wait = _next[0] - now
        _next[0] = max(now, _next[0]) + 1.0 / RATE
    if wait > 0:
        time.sleep(wait)


def get(url, tries=4):
    for i in range(tries):
        _throttle()
        try:
            r = S.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
        except Exception:
            pass
        time.sleep(2.0 * (i + 1))
    return None


def _am(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def side(s):
    s = s or {}
    out = {"ml": s.get("moneyLine"), "so": s.get("spreadOdds")}
    for k in ("open", "close"):
        out[f"ml_{k}"] = _am(s, k, "moneyLine", "american")
        out[f"sp_{k}"] = _am(s, k, "pointSpread", "american")
        out[f"so_{k}"] = _am(s, k, "spread", "american")
    return out


def trim(it):
    return {"pid": _am(it, "provider", "id"), "p": _am(it, "provider", "name"),
            "s": it.get("spread"), "t": it.get("overUnder"), "oo": it.get("overOdds"),
            "uo": it.get("underOdds"), "t_open": _am(it, "open", "total", "american"),
            "t_close": _am(it, "close", "total", "american"),
            "h": side(it.get("homeTeamOdds")), "a": side(it.get("awayTeamOdds"))}


def fetch(eid):
    f = OUT / f"{eid}.json"
    if f.exists():
        return 0
    js = get(ODDS.format(eid))
    if js is None:
        return -1
    items = [trim(x) for x in js.get("items", [])]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(items))
    tmp.replace(f)  # atomic: an interrupted run never leaves a half-written cache file
    return 1


def main():
    ev = pd.concat([pd.read_csv(f, usecols=["event_id"]) for f in
                    sorted(glob.glob(str(ROOT / "research/data/espn_*.csv")))])
    ids = [str(x) for x in ev.event_id.unique()]
    todo = [i for i in ids if not (OUT / f"{i}.json").exists()]
    print(f"{len(ids)} events, {len(todo)} to fetch", flush=True)
    done = fail = 0
    with ThreadPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(fetch, todo)):
            done += r == 1
            fail += r == -1
            if k % 250 == 0:
                print(f"{k}/{len(todo)} ok={done} fail={fail}", flush=True)
    print(f"finished ok={done} fail={fail}", flush=True)
    return fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
