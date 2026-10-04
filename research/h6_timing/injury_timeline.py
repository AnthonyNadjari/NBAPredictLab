"""When does injury-status information arrive on game day? (NBA official injury report archive)

Since 2025-26 the NBA publishes its official injury report every 15 minutes and keeps every
version at ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<hh>_<mm><AM|PM>.pdf.
For a sample of 2025-26 regular-season game days (every 4th day) we download the report
at the bot's run time (09:00 UTC) and at ~20 later ET clock times, parse player statuses,
and measure, per game, how much status information is still to change after each time:

  pending_minutes(slot) = sum over players whose status at `slot` differs from their final
                          pre-tip status, weighted by the player's average minutes in his
                          team's games BEFORE that date (2025-26 player logs, leak-free).

`final` = the last report published before the game's scheduled tip. Two-way / G League
assignments are dropped (they are not news). A player absent from a report counts as
"Available". Cache: research/data/h6_timing/injury_pdfs/ (PDFs) and injury_rows.csv (parsed).
Rate: 2 requests/second.
"""
import io
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RES, ROOT, espn_games  # noqa: E402

sys.path.insert(0, str(ROOT))

PDFS = DATA / "injury_pdfs"
PDFS.mkdir(parents=True, exist_ok=True)
ROWS = DATA / "injury_rows.csv"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{hh:02d}_{mm:02d}{ap}.pdf"
URL_HOURLY = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{hh:02d}{ap}.pdf"  # before ~2025-12-22
ET, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")
ET_SLOTS = ["08:00", "10:00", "11:00", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30", "15:00", "16:00",
            "17:00", "17:30", "18:00", "18:30", "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00"]
STATUSES = {"Out", "Questionable", "Doubtful", "Probable", "Available"}
DAY_STEP = 4
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0 (research; NBAPredictLab)"


def slots_for(day):
    """(label, ET datetime) for one game date: bot run (09:00 UTC) + fixed ET clock times."""
    bot = datetime.fromisoformat(f"{day}T09:00").replace(tzinfo=UTC).astimezone(ET)
    out = [("bot_09utc", bot.replace(minute=bot.minute - bot.minute % 15))]
    for s in ET_SLOTS:
        out.append((s, datetime.fromisoformat(f"{day}T{s}").replace(tzinfo=ET)))
    return out


def pdf_path(t):
    return PDFS / f"{t:%Y-%m-%d_%H%M}.pdf"


def download(t):
    f = pdf_path(t)
    if f.exists() or f.with_suffix(".missing").exists():
        return f if f.exists() else None
    hh, ap, d = t.hour % 12 or 12, "AM" if t.hour < 12 else "PM", t.strftime("%Y-%m-%d")
    urls = [URL.format(d=d, hh=hh, mm=t.minute, ap=ap)]
    if t.minute == 0:
        urls.append(URL_HOURLY.format(d=d, hh=hh, ap=ap))
    for url in urls:
        for i in range(3):
            time.sleep(0.5)
            try:
                r = S.get(url, timeout=30)
            except requests.RequestException:
                time.sleep(2 * (i + 1))
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                f.write_bytes(r.content)
                return f
            break  # 403/404: this naming does not exist for that slot
    f.with_suffix(".missing").write_text("403")
    return None


def parse(f):
    """Rows (game_date, matchup, team, player, status, reason) from one report PDF (x-position columns)."""
    from pypdf import PdfReader
    rows = []
    cur = {"date": None, "matchup": None, "team": None}
    for page in PdfReader(str(f)).pages:
        chunks = []
        page.extract_text(visitor_text=lambda t, cm, tm, fd, fs: chunks.append((tm[5], tm[4], t.strip()))
                          if t.strip() else None)
        hdr = {t: x for y, x, t in chunks if t in ("Team", "Player", "Current", "Reason", "Matchup")}
        xt, xp, xs, xr, xm = (hdr.get("Team", 264), hdr.get("Player", 425), hdr.get("Current", 585),
                              hdr.get("Reason", 666), hdr.get("Matchup", 200))
        lines = {}
        for y, x, t in chunks:
            lines.setdefault(round(y), []).append((x, t))
        for y in sorted(lines):
            toks = sorted(lines[y])
            col = lambda lo, hi: " ".join(t for x, t in toks if lo - 3 <= x < hi - 3)  # noqa: E731
            date = col(0, xm - 80)
            if re.match(r"\d{2}/\d{2}/\d{4}", date):
                cur["date"] = datetime.strptime(date[:10], "%m/%d/%Y").strftime("%Y-%m-%d")
            m = col(xm, xt)
            if re.match(r"^[A-Z]{2,4}@[A-Z]{2,4}$", m):
                cur["matchup"] = m
            team = col(xt, xp)
            if team and team not in ("Team",):
                cur["team"] = team
            name, status, reason = col(xp, xs), col(xs, xr), col(xr, 2000)
            if "NOT YET SUBMITTED" in " ".join(t for x, t in toks) and cur["matchup"]:
                rows.append({"game_date": cur["date"], "matchup": cur["matchup"], "team": cur["team"],
                             "player": "", "status": "NOT_SUBMITTED", "reason": "NOT YET SUBMITTED"})
                continue
            if status in STATUSES and name and cur["matchup"]:
                rows.append({"game_date": cur["date"], "matchup": cur["matchup"], "team": cur["team"],
                             "player": name, "status": status, "reason": reason})
    return rows


def norm_name(n):
    """'Pippen Jr., Scotty' -> 'scotty pippen jr' ; strips accents and punctuation."""
    if "," in n:
        last, first = n.split(",", 1)
        n = f"{first} {last}"
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z ]", "", n.lower().replace("-", " ")).split())


def build_rows():
    if ROWS.exists():
        return pd.read_csv(ROWS)
    g = espn_games()
    g = g[(g.season == "2025-26") & (g.season_type == 2)]
    days = sorted(g.game_date.unique())[::DAY_STEP]
    out = []
    for i, day in enumerate(days):
        for label, t in slots_for(day):
            f = download(t)
            if f is None:
                continue
            for r in parse(f):
                r.update({"day": day, "slot": label, "slot_et": t.isoformat()})
                out.append(r)
        print(f"  injury reports: {i + 1}/{len(days)} days", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(ROWS, index=False)
    return df


def player_minutes():
    """key -> (sorted game dates, cumulative minutes) from 2025-26 player logs, for leak-free averages."""
    base = ROOT / "research/data/h1_player_availability"
    frames = [pd.read_csv(base / f"player_logs_2025-26_{k}.csv", usecols=["PLAYER_NAME", "GAME_DATE", "MIN"])
              for k in ("RegularSeason", "PlayIn", "Playoffs") if (base / f"player_logs_2025-26_{k}.csv").exists()]
    p = pd.concat(frames, ignore_index=True)
    p["key"] = p.PLAYER_NAME.map(norm_name)
    p = p.sort_values("GAME_DATE")
    return {k: (g.GAME_DATE.to_numpy(), np.cumsum(g.MIN.fillna(0).to_numpy())) for k, g in p.groupby("key")}


def avg_minutes(pm, key, day):
    if key not in pm:
        return 0.0
    dates, cum = pm[key]
    i = np.searchsorted(dates, day)  # games strictly before `day`
    return float(cum[i - 1] / i) if i > 0 else 0.0


PLAY, OUT = {"Available", "Probable"}, {"Out", "Doubtful"}


def cat(status):
    return "play" if status in PLAY else "out" if status in OUT else "q" if status == "Questionable" else "unk"


def _order(u):
    h, m = int(u[:2]), int(u[3:])
    return (h + (24 if h < 8 else 0)) * 60 + m


def analyze():
    """Per game and report time: how much 'who plays' uncertainty is still unresolved.

    Players counted ('uncertain' set): listed Questionable / Doubtful / Probable at some point on game day,
    or whose play/sit category changes during the day (late scratch, late activation). Long-term Outs that
    never change are excluded (not news). A team marked NOT YET SUBMITTED is 'unknown' at that time.
    pending(t) = sum of minutes of uncertain players whose category at t differs from the final pre-tip
    category, or is still Questionable / unknown at t."""
    from src.engine.teams import to_code
    rows = build_rows()
    rows = rows[~rows.reason.fillna("").str.contains("G League|Two-Way", regex=True)].copy()
    rows["key"] = rows.player.fillna("").map(norm_name)

    def code(t):
        try:
            return to_code(str(t).strip())
        except KeyError:
            return None

    rows["team_code"] = rows.team.map(code)
    rows["slot_et"] = pd.to_datetime(rows.slot_et, utc=True)
    g = espn_games()
    g = g[(g.season == "2025-26") & (g.season_type == 2) & g.game_date.isin(rows.day.unique())]
    pm = player_minutes()
    recs, subm = [], []
    grouped = {k: v for k, v in rows.groupby(["day", "game_date"])}
    for gm in g.itertuples():
        day_rows = grouped.get((gm.game_date, gm.game_date))
        if day_rows is None:
            continue
        r = day_rows[day_rows.team_code.isin([gm.home, gm.away])]
        slots = sorted(t for t in day_rows.slot_et.unique() if t < gm.tip)
        if len(slots) < 2 or r.empty:
            continue
        status = {}  # (slot, team) -> {player key: status}, or None when unknown
        for t in slots:
            rt = r[r.slot_et == t]
            for team in (gm.home, gm.away):
                x = rt[rt.team_code == team]
                unknown = rt.empty or (x.status == "NOT_SUBMITTED").any()
                status[(t, team)] = None if unknown else dict(zip(x.key, x.status))
        final_t = slots[-1]
        if status[(final_t, gm.home)] is None or status[(final_t, gm.away)] is None:
            continue
        bot_slots = set(day_rows.loc[day_rows.slot == "bot_09utc", "slot_et"])
        for t in slots:
            for team in (gm.home, gm.away):
                subm.append({"event_id": gm.event_id, "slot_utc": t, "submitted": status[(t, team)] is not None})
        uncertain = []
        for team in (gm.home, gm.away):
            for k in set(r[(r.team_code == team) & (r.key != "")].key):
                known = [status[(t, team)].get(k, "Available") for t in slots if status[(t, team)] is not None]
                if any(x in ("Questionable", "Doubtful", "Probable") for x in known) or len({cat(x) for x in known}) > 1:
                    fin = cat(status[(final_t, team)].get(k, "Available"))
                    uncertain.append((team, k, fin, avg_minutes(pm, k, gm.game_date)))
        for t in slots:
            pend_m, pend_n = 0.0, 0
            for team, k, fin, w in uncertain:
                st = status[(t, team)]
                c = "unk" if st is None else cat(st.get(k, "Available"))
                if c in ("unk", "q") or c != fin:
                    pend_m += w
                    pend_n += 1
            recs.append({"event_id": gm.event_id, "game_date": gm.game_date, "tip": gm.tip, "slot_utc": t,
                         "is_bot_slot": t in bot_slots, "pending_minutes": pend_m, "pending_players": pend_n,
                         "uncertain_players": len(uncertain), "uncertain_minutes": sum(u[3] for u in uncertain),
                         "final_gtd_players": sum(u[2] == "q" for u in uncertain)})
    d = pd.DataFrame(recs)
    sb = pd.DataFrame(subm)
    d.to_csv(DATA / "injury_timeline_by_game.csv", index=False)  # large: kept in the gitignored cache
    d["utc"] = d.slot_utc.dt.strftime("%H:%M")
    sb["utc"] = sb.slot_utc.dt.strftime("%H:%M")
    ev_ids = set(d[d.tip.dt.hour.isin([23, 0, 1, 2, 3])].event_id)
    ev = d[d.event_id.isin(ev_ids)]
    bot = ev[ev.is_bot_slot].drop_duplicates("event_id").set_index("event_id")
    out = []
    for u, x in ev.groupby("utc"):
        x = x[x.event_id.isin(bot.index)]
        if len(x) < 20:
            continue
        b = bot.loc[x.event_id]
        sbx = sb[(sb.utc == u) & sb.event_id.isin(set(x.event_id))]
        out.append({"utc": u, "n_games": len(x), "teams_not_submitted": 1 - sbx.submitted.mean(),
                    "pending_players": x.pending_players.mean(), "pending_minutes": x.pending_minutes.mean(),
                    "games_with_pending": (x.pending_players > 0).mean(),
                    "share_of_09utc_pending": x.pending_minutes.sum() / max(b.pending_minutes.sum(), 1e-9)})
    t = pd.DataFrame(out)
    t = t.iloc[sorted(range(len(t)), key=lambda i: _order(t.utc.iloc[i]))].reset_index(drop=True)
    print(f"\n=== Injury-report timeline (2025-26 regular season, every {DAY_STEP}th game day) ===")
    print(f"games: {d.event_id.nunique()} (evening tips: {len(ev_ids)}); report versions parsed: "
          f"{rows[['day', 'slot']].drop_duplicates().shape[0]}")
    print(f"per evening game: {bot.uncertain_players.mean():.2f} players with game-day status uncertainty "
          f"({bot.uncertain_minutes.mean():.0f} player-minutes); still Questionable in the last pre-tip report: "
          f"{bot.final_gtd_players.mean():.2f}")
    print(f"at the bot's 09:00 UTC run: {(bot.pending_players > 0).mean():.1%} of evening games have >= 1 unresolved "
          f"status; {bot.pending_players.mean():.2f} players / {bot.pending_minutes.mean():.0f} player-minutes per "
          f"game still to resolve")
    print("evening games (tip 23:00-03:59 UTC): status uncertainty still unresolved, by UTC time of the report")
    print(t.round(3).to_string(index=False))
    t.to_csv(RES / "injury_timeline_by_utc.csv", index=False)
    return d, t


if __name__ == "__main__":
    analyze()
