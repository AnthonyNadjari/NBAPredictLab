#!/usr/bin/env python3
"""
One original post a day before the season (until opening night): a card + a short text, queued for
the server like any thread. Three formats in rotation, every number from our data:

  0  power rankings: our preseason Elo (last season's ratings, regressed to the mean like our model)
  1  best records last season: the official ESPN standings
  2  numbers of last season: longest win streak, best home and road records, biggest win...

    python scripts/daily_post.py [--date YYYY-MM-DD] [--dry]

Adds a special to docs/pending_games.json and a request to docs/vision/publish_queue.json, then
commits and pushes (unless --dry). Runs in the daily workflow after the 21:00 UTC refresh.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine import history  # noqa: E402
from src.engine.features import ELO_CARRY, ELO_HOME, ELO_K  # noqa: E402
from src.engine.teams import TEAMS  # noqa: E402

OPENING_NIGHT = date(2026, 10, 20)
LAST_SEASON = "2025-26"
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
NICK = {k: ("76ers" if v.endswith("76ers") else "Trail Blazers" if k == "POR" else v.split()[-1]) for k, v in TEAMS.items()}
CTA = "🔔 Follow: a probability on every game from Oct 20, graded after the buzzer."


CUP_FINALS = {"2025-12-16"}   # NBA Cup final: listed as regular season by ESPN, but not in the standings


def _regular(h):
    r = h[(h.season == LAST_SEASON) & (h.season_type == "Regular Season") & ~h.game_date.isin(CUP_FINALS)].copy()
    return r[r.home.isin(TEAMS) & r.away.isin(TEAMS)].sort_values("game_date")


def power_rankings(h):
    g = h[h.home.isin(TEAMS) & h.away.isin(TEAMS) & h.home_pts.notna()].sort_values("game_date")
    elo, season_of = {}, {}
    for row in g.itertuples(index=False):
        for t in (row.home, row.away):
            if t not in elo:
                elo[t] = 1500.0
            elif season_of[t] != row.season:
                elo[t] = 1500.0 + ELO_CARRY * (elo[t] - 1500.0)
            season_of[t] = row.season
        margin = row.home_pts - row.away_pts
        diff = elo[row.home] + ELO_HOME - elo[row.away]
        p = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        winner_diff = diff if margin > 0 else -diff
        mult = np.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        d = ELO_K * mult * (float(margin > 0) - p)
        elo[row.home] += d
        elo[row.away] -= d
    pre = {t: 1500.0 + ELO_CARRY * (v - 1500.0) for t, v in elo.items()}     # new season regression
    top = sorted(pre.items(), key=lambda kv: -kv[1])[:10]
    rows = [{"team": t, "value": f"{v:.0f}"} for t, v in top]
    t1, t2, t3 = (NICK[t] for t, _ in top[:3])
    text = (f"📊 Our 2026-27 preseason power rankings\n\n🥇 {t1}\n🥈 {t2}\n🥉 {t3}\n\n"
            f"Team strength from every game of last season, pulled back toward average for the summer.\n\n{CTA}")
    card = {"type": "ranking", "kicker": "2026-27 preseason", "eyebrow": "Power rankings · our model",
            "title_html": "Who's <em>strongest</em>", "rows": rows, "foot": "Elo rating, before opening night"}
    return text, card


def best_records(_h):
    js = requests.get("https://site.api.espn.com/apis/v2/sports/basketball/nba/standings",
                      params={"season": 2026, "seasontype": 2}, timeout=20).json()
    teams = []
    for conf in js.get("children", []):
        for e in conf["standings"]["entries"]:
            st = {x["name"]: x.get("value") for x in e["stats"]}
            code = ESPN_TO_NBA.get(e["team"]["abbreviation"], e["team"]["abbreviation"])
            teams.append((code, int(st["wins"]), int(st["losses"]), conf.get("abbreviation", "")))
    top = sorted(teams, key=lambda t: (-t[1], t[2]))[:10]
    rows = [{"team": c, "value": f"{w}-{l}", "note": conf} for c, w, l, conf in top]
    (a, wa, la, _), (b, wb, lb, _), (c, wc, lc, _) = top[:3]
    text = (f"🏆 Last season's best records\n\n1️⃣ {NICK[a]} {wa}-{la}\n2️⃣ {NICK[b]} {wb}-{lb}\n3️⃣ {NICK[c]} {wc}-{lc}\n\n"
            f"Who holds up in 2026-27? {CTA}")
    card = {"type": "ranking", "kicker": "2025-26 regular season", "eyebrow": "Official standings",
            "title_html": "The <em>best</em> records", "rows": rows, "foot": "Regular season, NBA standings"}
    return text, card


def season_numbers(h):
    r = _regular(h)
    rows = []
    # longest win streak
    best = (None, 0)
    for t in TEAMS:
        g = r[(r.home == t) | (r.away == t)]
        won = ((g.home == t) & (g.home_pts > g.away_pts)) | ((g.away == t) & (g.away_pts > g.home_pts))
        run = m = 0
        for w in won:
            run = run + 1 if w else 0
            m = max(m, run)
        if m > best[1]:
            best = (t, m)
    rows.append({"team": best[0], "value": f"{best[1]} W", "name": NICK[best[0]], "note": "longest win streak"})
    home = r.assign(w=r.home_pts > r.away_pts).groupby("home").w.agg(["sum", "count"])
    t = home["sum"].idxmax()
    rows.append({"team": t, "value": f"{int(home.loc[t, 'sum'])}-{int(home.loc[t, 'count'] - home.loc[t, 'sum'])}",
                 "name": NICK[t], "note": "best home record"})
    road = r.assign(w=r.away_pts > r.home_pts).groupby("away").w.agg(["sum", "count"])
    t = road["sum"].idxmax()
    rows.append({"team": t, "value": f"{int(road.loc[t, 'sum'])}-{int(road.loc[t, 'count'] - road.loc[t, 'sum'])}",
                 "name": NICK[t], "note": "best road record"})
    r = r.assign(m=(r.home_pts - r.away_pts).abs(), win=np.where(r.home_pts > r.away_pts, r.home, r.away),
                 tot=r.home_pts + r.away_pts)
    big = r.loc[r.m.idxmax()]
    rows.append({"team": big.win, "value": f"+{int(big.m)}", "name": NICK[big.win], "note": "biggest win"})
    pts = pd.concat([r[["home", "home_pts"]].set_axis(["t", "p"], axis=1),
                     r[["away", "away_pts"]].set_axis(["t", "p"], axis=1)], ignore_index=True)
    top_pts = pts.loc[pts.p.idxmax()]
    rows.append({"team": top_pts.t, "value": f"{int(top_pts.p)}", "name": NICK[top_pts.t], "note": "most points in a game"})
    diff = pd.concat([r.assign(t=r.home, d=r.home_pts - r.away_pts)[["t", "d"]],
                      r.assign(t=r.away, d=r.away_pts - r.home_pts)[["t", "d"]]]).groupby("t").d.mean()
    t = diff.idxmax()
    rows.append({"team": t, "value": f"+{diff[t]:.1f}", "name": NICK[t], "note": "best point differential per game"})
    text = (f"🔢 2025-26 in numbers\n\n🔥 {rows[0]['name']}: {best[1]} straight wins\n🏠 {rows[1]['name']}: {rows[1]['value']} at home\n"
            f"✈️ {rows[2]['name']}: {rows[2]['value']} on the road\n💥 {rows[3]['name']} won by {int(big.m)}\n\n{CTA}")
    card = {"type": "ranking", "kicker": "2025-26 regular season", "eyebrow": "Season in numbers",
            "title_html": "Last season, <em>in numbers</em>", "rows": rows, "numbered": False, "foot": "Regular season, all 1,230 games"}
    return text, card


FORMATS = [power_rankings, best_records, season_numbers]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    day = date.fromisoformat(a.date) if a.date else datetime.now(timezone.utc).date()
    if day >= OPENING_NIGHT:
        print("Season started: threads take over, no daily post")
        return 0
    fmt = FORMATS[day.toordinal() % len(FORMATS)]
    text, card = fmt(history.load())
    sid = f"daily-{day.isoformat()}"
    print(f"{sid} ({fmt.__name__}, {len(text)} chars)\n{text}\n{json.dumps(card)[:300]}")
    if a.dry:
        return 0
    pend_p, q_p = ROOT / "docs" / "pending_games.json", ROOT / "docs" / "vision" / "publish_queue.json"
    for attempt in range(3):
        pend = json.loads(pend_p.read_text(encoding="utf-8"))
        specials = [s for s in pend.get("specials", []) if s.get("id") != sid]
        if any(s.get("id") == sid and s.get("published") for s in pend.get("specials", [])):
            print("already published")
            return 0
        pend["specials"] = specials + [{"id": sid, "type": "announcement", "title": fmt.__name__.replace("_", " ").title(),
                                        "matchup": "Post du jour", "date": day.isoformat(), "published": False,
                                        "thread": [{"text": text, "card": card}]}]
        pend_p.write_text(json.dumps(pend, indent=2, ensure_ascii=False), encoding="utf-8")
        q = json.loads(q_p.read_text(encoding="utf-8")) if q_p.exists() else {"requests": []}
        rid = f"daily-{day.isoformat()}"
        if not any(r.get("id") == rid for r in q["requests"]):
            q["requests"] = (q["requests"] + [{"id": rid, "game_id": sid, "texts": None, "dry": False,
                                               "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                               "source": "daily_post"}])[-20:]
        q_p.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
        git = lambda *x: subprocess.run(["git", *x], cwd=ROOT).returncode
        git("add", "docs/pending_games.json", "docs/vision/publish_queue.json")
        git("-c", "user.name=github-actions[bot]", "-c", "user.email=github-actions[bot]@users.noreply.github.com",
            "commit", "-q", "-m", f"Daily post queued: {sid}")
        if git("push", "-q", "origin", "HEAD:main") == 0:
            return 0
        git("fetch", "-q", "origin", "main")
        git("reset", "-q", "--hard", "origin/main")
    return 1


if __name__ == "__main__":
    sys.exit(main())
