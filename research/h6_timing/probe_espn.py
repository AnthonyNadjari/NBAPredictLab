"""Does ESPN keep intraday odds history / as-of-game injury reports for past NBA games?

Probes, on a stratified sample of past events (all 5 seasons):
  1. core odds API, per provider: odds/{pid}/history/{0,1}/movement?limit=100  -> count of points
  2. odds/{pid}/history and odds/{pid}/history/0 (other plausible paths)       -> HTTP status
  3. site summary?event=ID: `injuries` (statuses + update timestamps) and `pickcenter`
Results cached to research/data/h6_timing/espn_probe.json (reruns are free).
Rate: ~3 requests/second.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/data/h6_timing/espn_probe.json"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{0}/competitions/{0}/odds"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (research; NBAPredictLab)"
PER_SEASON = 8
SKIP_PROVIDERS = ("accuscore", "consensus", "teamrankings", "numberfire", "betegy")


def get(url):
    time.sleep(0.34)
    try:
        r = S.get(url, timeout=20)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None)
    except Exception as e:  # network error: record it, do not crash the probe
        return str(e), None


def probe_event(eid, game_date):
    code, js = get(CORE.format(eid) + "?limit=50")
    res = {"event_id": int(eid), "odds_status": code, "providers": []}
    for it in (js or {}).get("items", []):
        pid, name = it["provider"]["id"], it["provider"]["name"]
        if any(s in name.lower() for s in SKIP_PROVIDERS):
            continue
        p = {"pid": pid, "name": name}
        for h in (0, 1):
            c, mv = get(f"{CORE.format(eid)}/{pid}/history/{h}/movement?limit=100")
            p[f"movement_{h}"] = (mv or {}).get("count") if c == 200 else c
        p["history_status"] = get(f"{CORE.format(eid)}/{pid}/history")[0]
        res["providers"].append(p)
    c, sm = get(SUMMARY.format(eid))
    inj = (sm or {}).get("injuries") or []
    dates = [i.get("date") for t in inj for i in t.get("injuries", []) if i.get("date")]
    res["injuries_n"] = len(dates)
    res["injury_dates_before_game"] = sum(d[:10] <= game_date for d in dates)
    res["injury_dates_max"] = max(dates) if dates else None
    res["pickcenter_n"] = len((sm or {}).get("pickcenter") or [])
    return res


def main():
    if OUT.exists():
        rows = json.loads(OUT.read_text())
    else:
        rows = []
        for f in sorted((ROOT / "research/data").glob("espn_20*.csv")):
            o = pd.read_csv(f)
            o = o[o.completed == True]  # noqa: E712
            for r in o.sample(PER_SEASON, random_state=6).itertuples():
                gd = pd.Timestamp(r.date_utc).tz_convert("America/New_York").strftime("%Y-%m-%d")
                x = probe_event(r.event_id, gd)
                x["season"] = r.season
                x["game_date"] = gd
                rows.append(x)
                print(r.season, r.event_id, [(p["name"], p["movement_0"]) for p in x["providers"]], flush=True)
        OUT.write_text(json.dumps(rows, indent=1))
    n_prov = sum(len(r["providers"]) for r in rows)
    n_mv = sum(1 for r in rows for p in r["providers"]
               if isinstance(p["movement_0"], int) and p["movement_0"] > 0
               or isinstance(p["movement_1"], int) and p["movement_1"] > 0)
    inj_any = sum(r["injuries_n"] > 0 for r in rows)
    inj_asof = sum(r["injury_dates_before_game"] > 0 for r in rows)
    summary = {
        "events_probed": len(rows),
        "provider_feeds_probed": n_prov,
        "feeds_with_movement_points": n_mv,
        "events_with_injuries_listed": inj_any,
        "events_with_any_injury_note_dated_on_or_before_game": inj_asof,
        "events_with_pickcenter": sum(r["pickcenter_n"] > 0 for r in rows),
        "latest_injury_note_dates": sorted({(r["injury_dates_max"] or "")[:7] for r in rows}),
    }
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
