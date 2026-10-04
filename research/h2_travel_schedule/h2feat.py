"""Travel / schedule-fatigue features, computed strictly from the schedule before each game.

Input: research/data/team_logs.csv (nba_api team game logs; one row per team-game) plus the
three NBA Cup finals that the logs omit. Only dates and venues of the team's *previous* games in
the same season, and the venue/date of the current game, are used -- never box scores or results.

Per team-game output columns:
  venue, at_home, travel_km (prev venue -> this venue), tz_shift (h, + = eastward),
  rest_days, b2b, b2b_travel (2nd night of B2B after >50 km travel), three_in_four, four_in_six,
  games_last7 (games in the 7 days before), travel_7d (km travelled over days d-6..d),
  road_trip_len (consecutive non-home games incl. this one), home_stand_len,
  days_since_home (days since last game in own arena, capped 30), return_from_trip (home game
  right after a road trip: its length), visiting_altitude (playing >1000 m from a low home),
  after_altitude (previous game was at altitude, team based low), venue_elev.

Limitations: a team's actual itinerary is unknown, so travel is "previous venue -> this venue"
(a team that flies home between two road games is credited with the direct leg).
"""
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from arenas import CUP_FINALS, NEUTRAL, NEUTRAL_GAMES, home_arena

ROOT = Path(__file__).resolve().parents[2]
FIRST_SEASON = "2021-22"


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


_tz_cache = {}


def utc_offset_h(tz, date_str):
    """UTC offset (hours) of a time zone at local noon on a date (handles DST and Arizona)."""
    k = (tz, date_str)
    if k not in _tz_cache:
        ts = pd.Timestamp(date_str + " 12:00").to_pydatetime().replace(tzinfo=ZoneInfo(tz))
        _tz_cache[k] = ts.utcoffset().total_seconds() / 3600
    return _tz_cache[k]


def _parse_home(matchup):
    # "A vs. B" -> A home ; "A @ B" -> B home. (Neutral-site games print the same string for both.)
    if " vs. " in matchup:
        a, b = matchup.split(" vs. ")
        return a.strip(), b.strip()
    a, b = matchup.split(" @ ")
    return b.strip(), a.strip()


def schedule():
    """One row per NBA game (seasons >= 2021-22) with date, season, type, home, away, venue tuple."""
    t = pd.read_csv(ROOT / "research/data/team_logs.csv", dtype={"GAME_ID": str})
    t = t[t.SEASON >= FIRST_SEASON]
    g = t.drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "MATCHUP"]].copy()
    hv = g.MATCHUP.map(_parse_home)
    g["home"], g["away"] = hv.str[0], hv.str[1]
    g = g.rename(columns={"GAME_ID": "nba_game_id", "GAME_DATE": "date", "SEASON": "season",
                          "SEASON_TYPE": "season_type"}).drop(columns="MATCHUP")
    cup = pd.DataFrame([{"nba_game_id": f"cupfinal_{s}", "date": dt, "season": s, "season_type": "CupFinal",
                         "home": h, "away": a} for dt, s, h, a in CUP_FINALS])
    g = pd.concat([g, cup], ignore_index=True)
    # sanity: each team appears at most once per date
    long = pd.concat([g[["date", "home"]].rename(columns={"home": "team"}),
                      g[["date", "away"]].rename(columns={"away": "team"})])
    assert not long.duplicated().any(), "team plays twice on one date?"

    def venue(r):
        key = (r.date, frozenset({r.home, r.away}))
        if key in NEUTRAL_GAMES:
            return NEUTRAL_GAMES[key], NEUTRAL[NEUTRAL_GAMES[key]]
        return r.home, home_arena(r.home, r.season)

    v = g.apply(venue, axis=1)
    g["venue_code"], g["venue"] = v.str[0], v.str[1]
    g["neutral"] = g.venue_code.isin(NEUTRAL.keys())
    used = {k for k in NEUTRAL_GAMES if k[0] in set(g.date)}
    found = {(r.date, frozenset({r.home, r.away})) for r in g.itertuples() if r.neutral}
    missing = used - found
    assert not missing, f"neutral-site games not found in schedule: {missing}"
    return g.sort_values(["date", "nba_game_id"]).reset_index(drop=True)


def team_features(g):
    """Per (date, team) schedule features, using only games strictly before `date` + this venue."""
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = g[["nba_game_id", "date", "season", "venue_code", "venue", "neutral"]].copy()
        x["team"], x["opp"] = g[side], g[opp]
        x["nominal_home"] = side == "home"
        rows.append(x)
    tg = pd.concat(rows, ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)
    out = []
    for (team, season), s in tg.groupby(["team", "season"], sort=False):
        s = s.sort_values("date")
        own = home_arena(team, season)
        own_elev = own[3]
        dates = pd.to_datetime(s.date).to_numpy()
        prev_venue, prev_date, prev_at_home = own, None, True
        road, stand, last_home = 0, 0, None
        first = pd.Timestamp(dates[0])
        travel_hist = []  # (date, km)
        prev_venue_elev = own_elev
        for i, r in enumerate(s.itertuples()):
            d = pd.Timestamp(dates[i])
            at_home = (not r.neutral) and r.nominal_home
            km = haversine_km(prev_venue, r.venue)
            off_now = utc_offset_h(r.venue[2], r.date)
            off_prev = utc_offset_h(prev_venue[2], (prev_date or (d - pd.Timedelta(days=1))).strftime("%Y-%m-%d"))
            rest = (d - prev_date).days if prev_date is not None else 10
            past = dates[:i]
            n_in = lambda lo: int(((past >= np.datetime64(d - pd.Timedelta(days=lo))) & (past < np.datetime64(d))).sum())
            travel_hist.append((d, km))
            travel_7d = sum(k for dd, k in travel_hist if (d - dd).days <= 6)
            return_len = road if (at_home and not prev_at_home) else 0
            road = 0 if at_home else road + 1
            stand = stand + 1 if at_home else 0
            ref_home = last_home if last_home is not None else first - pd.Timedelta(days=1)
            out.append({
                "nba_game_id": r.nba_game_id, "date": r.date, "season": season, "team": team,
                "opp": r.opp, "venue_code": r.venue_code, "at_home": int(at_home),
                "travel_km": km, "tz_shift": off_now - off_prev, "rest_days": min(rest, 10),
                "b2b": int(rest == 1), "b2b_travel": int(rest == 1 and km > 50),
                "three_in_four": int(n_in(3) + 1 >= 3), "four_in_six": int(n_in(5) + 1 >= 4),
                "games_last7": n_in(7), "travel_7d": travel_7d,
                "road_trip_len": road, "home_stand_len": stand,
                "days_since_home": min((d - ref_home).days, 30) if not at_home else 0,
                "return_from_trip": return_len,
                "visiting_altitude": int(r.venue[3] > 1000 and own_elev < 1000 and not at_home),
                "after_altitude": int(prev_venue_elev > 1000 and own_elev < 1000 and prev_date is not None
                                      and rest <= 2 and r.venue[3] < 1000),
                "venue_elev": r.venue[3], "home_tz": own[2],
                "first_game": int(prev_date is None),
            })
            if at_home:
                last_home = d
            prev_venue, prev_date, prev_at_home, prev_venue_elev = r.venue, d, at_home, r.venue[3]
    return pd.DataFrame(out)


def body_clock_hour(tip_utc, tz):
    """Tip-off hour expressed in a team's home time zone (circadian 'body clock')."""
    return tip_utc.tz_convert(tz).hour + tip_utc.tz_convert(tz).minute / 60


def build(cache=True):
    path = ROOT / "research/data/h2_travel_schedule/team_schedule_features.csv"
    if cache and path.exists():
        return pd.read_csv(path, dtype={"nba_game_id": str})
    g = schedule()
    f = team_features(g)
    path.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(path, index=False)
    return f


if __name__ == "__main__":
    f = build(cache=False)
    print(f.shape)
    print(f.describe().T.round(2).to_string())
