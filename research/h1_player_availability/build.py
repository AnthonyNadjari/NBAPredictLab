"""H1 data build: games + odds + leak-free player-availability features.

Output: research/data/h1_player_availability/h1_games.csv (one row per game).

Definitions (all "prior" quantities use games strictly before the game):
  team game index j      chronological games of a team (reg season + play-in + playoffs)
  rotation at j          player who (a) appeared for the team in at least one of its last
                         ROT_WINDOW games of the same season, (b) averaged >= ROT_MIN minutes
                         over his last ROT_WINDOW appearances for the team (same season),
                         (c) whose most recent appearance anywhere was for this team
                         (i.e. not traded away / signed elsewhere since).
  absent_oracle at j     rotation player with no row in game j's box score
                         (ORACLE: in production this needs the inactive list / injury report)
  absent_prev at j       rotation player with no row in the team's previous game
                         (REALISTIC: known before tip-off, "out last game, probably out again")
  player value (prior)   from the player's last VALUE_WINDOW appearances (any team, any
                         season, strictly before the game):
     v_min  = expected minutes (mean of last ROT_WINDOW appearances for this team)
     v_gs   = (shrunk GameScore per minute - replacement rate) * v_min
     v_oo   = shrunk on/off plus-minus per 48 * v_min / 48
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "research" / "data"
CACHE = DATA / "h1_player_availability"
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

ROT_WINDOW = 10      # team games / appearances defining the expected rotation
ROT_MIN = 15.0       # minutes threshold for the rotation
VALUE_WINDOW = 82    # appearances used for the per-minute value estimates
K_GS = 300.0         # minutes of shrinkage toward replacement for GameScore/min
K_OO = 2000.0        # minutes of shrinkage toward 0 for on/off
SEASONS_PLAYERS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


# --------------------------------------------------------------------------- odds / games
def am_to_p(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def espn_odds():
    from src.engine.teams import from_espn
    o = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(DATA / "espn_*.csv")))], ignore_index=True)
    o["home"], o["away"] = o.home.map(from_espn), o.away.map(from_espn)
    o["game_date"] = (pd.to_datetime(o.date_utc, utc=True).dt.tz_convert("America/New_York")
                      .dt.strftime("%Y-%m-%d"))
    # raw American odds incl. vig. 2021-22 has no open/close split: its "current" line
    # for a completed game is the closing line (identical to home_ml_close in 2022-23 100%).
    o["ml_home_close"] = pd.to_numeric(o.home_ml_close, errors="coerce").fillna(pd.to_numeric(o.home_ml, errors="coerce"))
    o["ml_away_close"] = pd.to_numeric(o.away_ml_close, errors="coerce").fillna(pd.to_numeric(o.away_ml, errors="coerce"))
    o["ml_home_open"] = pd.to_numeric(o.home_ml_open, errors="coerce")
    o["ml_away_open"] = pd.to_numeric(o.away_ml_open, errors="coerce")
    o = o.drop_duplicates(["game_date", "home", "away"], keep="last")
    return o[["game_date", "home", "away", "season", "ml_home_close", "ml_away_close", "ml_home_open",
              "ml_away_open"]].rename(columns={"season": "espn_season"})


def games_table():
    """2022-23..2025-26 = the shared upsets dataset (identical evaluation set);
    2021-22 = training-only block rebuilt from team logs + ESPN closing lines."""
    f = DATA / "upsets_dataset.csv"
    if f.exists():
        d = pd.read_csv(f, dtype={"game_id": str})
    else:
        from upsets import load
        d = load()
        d["game_id"] = d.game_id.astype(str)
    d["game_id"] = d.game_id.str.zfill(10)
    keep = ["game_id", "game_date", "season", "season_type", "home", "away", "home_win", "margin",
            "mkt_open", "mkt_close", "spread", "elo_win_prob"]
    d = d[keep].copy()

    t = pd.read_csv(DATA / "team_logs.csv", dtype={"GAME_ID": str})
    t = t[t.SEASON == "2021-22"]
    h = t[t.MATCHUP.str.contains(" vs. ")]
    a = t[t.MATCHUP.str.contains(" @ ")][["GAME_ID", "TEAM_ABBREVIATION", "PTS"]]
    g = h.merge(a, on="GAME_ID", suffixes=("", "_a"))
    g21 = pd.DataFrame({"game_id": g.GAME_ID, "game_date": g.GAME_DATE, "season": g.SEASON,
                        "season_type": g.SEASON_TYPE, "home": g.TEAM_ABBREVIATION, "away": g.TEAM_ABBREVIATION_a,
                        "margin": g.PTS - g.PTS_a})
    g21["home_win"] = (g21.margin > 0).astype(float)
    d = pd.concat([g21, d], ignore_index=True)

    o = espn_odds()
    d = d.merge(o, on=["game_date", "home", "away"], how="left")
    for k in ("close", "open"):
        ph, pa = am_to_p(d[f"ml_home_{k}"]), am_to_p(d[f"ml_away_{k}"])
        d[f"mkt_{k}_raw"] = ph / (ph + pa)
    # keep the shared dataset's de-vigged numbers where present (identical by construction)
    d["mkt_close"] = d.mkt_close.fillna(d.mkt_close_raw)
    d["mkt_open"] = d.mkt_open.fillna(d.mkt_open_raw)
    d = d[d.mkt_close.notna() & d.home_win.notna()].drop(columns=["mkt_close_raw", "mkt_open_raw", "espn_season"])
    return d.sort_values(["game_date", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- player logs
def player_logs():
    files = sorted(CACHE.glob("player_logs_*.csv"))
    if not files:
        raise SystemExit("no player logs cached: run fetch_players.py first")
    p = pd.concat([pd.read_csv(f, dtype={"GAME_ID": str}) for f in files], ignore_index=True)
    p["GAME_ID"] = p.GAME_ID.str.zfill(10)
    p = p.drop_duplicates(["PLAYER_ID", "GAME_ID"])
    p["date"] = pd.to_datetime(p.GAME_DATE)
    p["MIN"] = p.MIN.fillna(0).astype(float)
    # Hollinger Game Score
    p["gmsc"] = (p.PTS + 0.4 * p.FGM - 0.7 * p.FGA - 0.4 * (p.FTA - p.FTM) + 0.7 * p.OREB + 0.3 * p.DREB
                 + p.STL + 0.7 * p.AST + 0.7 * p.BLK - 0.4 * p.PF - p.TOV)
    return p


def team_logs():
    t = pd.read_csv(DATA / "team_logs.csv", dtype={"GAME_ID": str})
    t = t[t.SEASON.isin(SEASONS_PLAYERS)].copy()
    t["GAME_ID"] = t.GAME_ID.str.zfill(10)
    t["date"] = pd.to_datetime(t.GAME_DATE)
    t = t.drop_duplicates(["TEAM_ABBREVIATION", "GAME_ID"])
    t["game_min"] = t.MIN / 5.0
    return t[["TEAM_ABBREVIATION", "GAME_ID", "date", "SEASON", "PLUS_MINUS", "game_min"]].rename(
        columns={"TEAM_ABBREVIATION": "team", "PLUS_MINUS": "team_margin", "SEASON": "season"})


def player_values(p, t, repl_rate):
    """Per player-appearance: value estimates using that appearance and the previous
    VALUE_WINDOW-1 ones. They are later attached to a game via 'last appearance strictly
    before the game date', so they never include the game being predicted."""
    p = p.merge(t[["team", "GAME_ID", "team_margin", "game_min"]],
                left_on=["TEAM_ABBREVIATION", "GAME_ID"], right_on=["team", "GAME_ID"], how="left")
    p["game_min"] = p.game_min.fillna(48.0)
    p["team_margin"] = p.team_margin.fillna(0.0)
    p["off_pm"] = p.team_margin - p.PLUS_MINUS
    p["off_min"] = (p.game_min - p.MIN).clip(lower=0)
    p = p.sort_values(["PLAYER_ID", "date"])
    g = p.groupby("PLAYER_ID", sort=False)
    roll = lambda c: g[c].rolling(VALUE_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
    s_min, s_gs = roll("MIN"), roll("gmsc")
    s_pm, s_off, s_offmin = roll("PLUS_MINUS"), roll("off_pm"), roll("off_min")
    p["gs_rate"] = (s_gs + repl_rate * K_GS) / (s_min + K_GS)
    on48 = 48 * s_pm / s_min.clip(lower=1)
    off48 = 48 * s_off / s_offmin.clip(lower=1)
    p["onoff48"] = (on48 - off48) * s_min / (s_min + K_OO)
    return p[["PLAYER_ID", "date", "gs_rate", "onoff48"]]


def replacement_rate(p):
    """GameScore per minute of fringe players (<15 min/game) in 2020-21 only (pre-test)."""
    q = p[p.SEASON == "2020-21"]
    avg = q.groupby("PLAYER_ID").MIN.mean()
    fringe = q[q.PLAYER_ID.isin(avg[avg < ROT_MIN].index)]
    return float(fringe.gmsc.sum() / fringe.MIN.sum())


def availability(p, t, vals):
    """One row per team-game with missing-value sums (oracle and previous-game proxy)."""
    last_team = p.sort_values("date")[["PLAYER_ID", "date", "TEAM_ABBREVIATION"]]
    out = []
    for team, tg in t.sort_values(["date", "GAME_ID"]).groupby("team"):
        tg = tg.reset_index(drop=True)
        pt = p[p.TEAM_ABBREVIATION == team]
        players = pt.PLAYER_ID.unique()
        pidx = {pid: k for k, pid in enumerate(players)}
        gidx = {gid: k for k, gid in enumerate(tg.GAME_ID)}
        n, m = len(tg), len(players)
        minutes = np.full((n, m), np.nan)  # nan = no box-score row
        rows = pt[pt.GAME_ID.isin(gidx)]
        minutes[rows.GAME_ID.map(gidx).to_numpy(), rows.PLAYER_ID.map(pidx).to_numpy()] = rows.MIN.to_numpy()
        played = ~np.isnan(minutes)
        seasons = tg.season.to_numpy()
        for j in range(n):
            # same-season window of previous team games
            lo = j
            while lo > 0 and seasons[lo - 1] == seasons[j] and j - lo < ROT_WINDOW:
                lo -= 1
            if lo == j:
                continue  # first game of the season: no rotation yet
            recent = played[lo:j].any(axis=0)
            cand = np.where(recent)[0]
            # season start of this team
            s0 = j
            while s0 > 0 and seasons[s0 - 1] == seasons[j]:
                s0 -= 1
            for c in cand:
                mins = minutes[s0:j, c]
                mins = mins[~np.isnan(mins)][-ROT_WINDOW:]
                exp_min = mins.mean()
                if exp_min < ROT_MIN:
                    continue
                out.append((team, tg.GAME_ID[j], tg.date[j], players[c], exp_min,
                            not played[j, c], not played[j - 1, c], j - s0))
    a = pd.DataFrame(out, columns=["team", "GAME_ID", "date", "PLAYER_ID", "exp_min", "absent_oracle",
                                   "absent_prev", "team_game_no"])
    # (c) last appearance anywhere strictly before the game must be for this team
    a = a.sort_values("date")
    lt = last_team.rename(columns={"date": "last_date", "TEAM_ABBREVIATION": "last_team"}).sort_values("last_date")
    a = pd.merge_asof(a, lt, left_on="date", right_on="last_date", by="PLAYER_ID", allow_exact_matches=False,
                      direction="backward")
    a = a[a.last_team == a.team].drop(columns=["last_team", "last_date"])
    # prior value: last appearance strictly before the game date
    v = vals.sort_values("date").rename(columns={"date": "vdate"})
    a = pd.merge_asof(a.sort_values("date"), v, left_on="date", right_on="vdate", by="PLAYER_ID",
                      allow_exact_matches=False, direction="backward")
    a["v_min"] = a.exp_min
    a["v_gs"] = (a.gs_rate - REPL) * a.exp_min
    a["v_oo"] = a.onoff48 * a.exp_min / 48
    return a


REPL = None


def team_features(a):
    a = a.copy()
    # rank within the team's rotation for this game (by prior value, GameScore metric)
    a["rank_gs"] = a.groupby(["team", "GAME_ID"]).v_gs.rank(ascending=False, method="first")
    a["rank_min"] = a.groupby(["team", "GAME_ID"]).v_min.rank(ascending=False, method="first")
    feats = {}
    for kind in ("oracle", "prev"):
        flag = a[f"absent_{kind}"].astype(float)
        for v in ("v_min", "v_gs", "v_oo"):
            a[f"miss_{v[2:]}_{kind}"] = flag * a[v]
        a[f"top2_{kind}"] = flag * (a.rank_gs <= 2)
        a[f"top1_{kind}"] = flag * (a.rank_gs <= 1)
    # decomposition of oracle absences: new (played last game) vs carried over
    a["miss_gs_new"] = (a.absent_oracle & ~a.absent_prev).astype(float) * a.v_gs
    a["miss_gs_return"] = (~a.absent_oracle & a.absent_prev).astype(float) * a.v_gs  # back from absence
    a["top2_new"] = (a.absent_oracle & ~a.absent_prev).astype(float) * (a.rank_gs <= 2)
    a["top2_return"] = (~a.absent_oracle & a.absent_prev).astype(float) * (a.rank_gs <= 2)
    cols = [c for c in a.columns if c.startswith(("miss_", "top2_", "top1_"))]
    agg = a.groupby(["team", "GAME_ID"])[cols].sum()
    agg["n_rot"] = a.groupby(["team", "GAME_ID"]).size()
    agg["rot_gs_total"] = a.groupby(["team", "GAME_ID"]).v_gs.sum()
    return agg.reset_index()


def build(force=False):
    global REPL
    out = CACHE / "h1_games.csv"
    if out.exists() and not force:
        return pd.read_csv(out, dtype={"game_id": str})
    p, t = player_logs(), team_logs()
    REPL = replacement_rate(p)
    print(f"replacement GameScore/min (2020-21 fringe players) = {REPL:.3f}")
    vals = player_values(p, t, REPL)
    a = availability(p, t, vals)
    a.to_csv(CACHE / "rotation_player_games.csv", index=False)
    tf = team_features(a)
    g = games_table()
    for side in ("home", "away"):
        x = tf.rename(columns={c: f"{side}_{c}" for c in tf.columns if c not in ("team", "GAME_ID")})
        g = g.merge(x, left_on=["game_id", side], right_on=["GAME_ID", "team"], how="left").drop(
            columns=["GAME_ID", "team"])
    num = [c for c in g.columns if c.startswith(("home_miss", "away_miss", "home_top", "away_top", "home_n_rot",
                                                  "away_n_rot", "home_rot", "away_rot"))]
    g["has_rot"] = g.home_n_rot.notna() & g.away_n_rot.notna()
    g[num] = g[num].fillna(0.0)
    g.to_csv(out, index=False)
    return g


if __name__ == "__main__":
    g = build(force=True)
    print(g.groupby("season").agg(n=("game_id", "size"), has_rot=("has_rot", "mean"),
                                  open=("mkt_open", lambda s: s.notna().sum()),
                                  ml_close=("ml_home_close", lambda s: s.notna().sum())))
