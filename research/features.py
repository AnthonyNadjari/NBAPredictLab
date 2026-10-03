"""Leak-free pre-game features built from team game logs.

Every feature for a game uses only games played strictly before it.
Input: one row per team per game (nba_api LeagueGameFinder format).
"""
import numpy as np
import pandas as pd

ELO_K = 20.0
ELO_HOME = 70.0          # home advantage in Elo points
ELO_CARRY = 0.75         # share of previous-season Elo kept
EWM_SPAN = 12            # games for "form" averages


def to_games(logs: pd.DataFrame) -> pd.DataFrame:
    """Merge the two team rows of each game into one home/away row."""
    logs = logs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["is_home"] = logs["MATCHUP"].str.contains(" vs. ")
    logs["poss"] = logs["FGA"] - logs["OREB"] + logs["TOV"] + 0.44 * logs["FTA"]
    keep = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "poss", "FG3A", "FGA", "FTA", "TOV", "OREB"]
    h = logs[logs.is_home][keep + ["GAME_DATE", "SEASON", "SEASON_TYPE"]]
    a = logs[~logs.is_home][keep]
    g = h.merge(a, on="GAME_ID", suffixes=("_h", "_a"))
    g = g.rename(columns={"GAME_DATE": "date", "SEASON": "season", "SEASON_TYPE": "season_type",
                          "TEAM_ABBREVIATION_h": "home", "TEAM_ABBREVIATION_a": "away",
                          "TEAM_ID_h": "home_id", "TEAM_ID_a": "away_id"})
    g["home_win"] = (g.PTS_h > g.PTS_a).astype(int)
    g["margin"] = g.PTS_h - g.PTS_a
    return g.sort_values(["date", "GAME_ID"]).reset_index(drop=True)


def add_elo(g: pd.DataFrame, k=ELO_K, home=ELO_HOME, carry=ELO_CARRY) -> pd.DataFrame:
    elo, last_season = {}, {}
    pre_h, pre_a = [], []
    for row in g.itertuples(index=False):
        for t in (row.home_id, row.away_id):
            if t not in elo:
                elo[t] = 1500.0
            elif last_season[t] != row.season:
                elo[t] = 1500.0 + carry * (elo[t] - 1500.0)
            last_season[t] = row.season
        eh, ea = elo[row.home_id], elo[row.away_id]
        pre_h.append(eh)
        pre_a.append(ea)
        diff = eh + home - ea
        p = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        mov = abs(row.margin)
        mult = np.log(mov + 1) * 2.2 / ((diff if row.home_win else -diff) * 0.001 + 2.2)
        delta = k * mult * (row.home_win - p)
        elo[row.home_id] += delta
        elo[row.away_id] -= delta
    g = g.copy()
    g["elo_h"], g["elo_a"] = pre_h, pre_a
    g["elo_diff"] = g.elo_h + home - g.elo_a
    g["elo_prob"] = 1.0 / (1.0 + 10 ** (-g.elo_diff / 400.0))
    return g


def _team_view(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with its own/opponent stats."""
    cols = {}
    for side, opp in (("h", "a"), ("a", "h")):
        d = pd.DataFrame({
            "GAME_ID": g.GAME_ID, "date": g.date, "season": g.season,
            "team": g[f"{'home' if side == 'h' else 'away'}_id"],
            "is_home": side == "h",
            "pts": g[f"PTS_{side}"], "opp_pts": g[f"PTS_{opp}"],
            "poss": (g[f"poss_{side}"] + g[f"poss_{opp}"]) / 2,
            "win": g.home_win if side == "h" else 1 - g.home_win,
        })
        cols[side] = d
    t = pd.concat(cols.values()).sort_values(["team", "date", "GAME_ID"]).reset_index(drop=True)
    t["ortg"] = 100 * t.pts / t.poss
    t["drtg"] = 100 * t.opp_pts / t.poss
    t["net"] = t.ortg - t.drtg
    return t


def add_form(g: pd.DataFrame, span=EWM_SPAN) -> pd.DataFrame:
    t = _team_view(g)
    grp = t.groupby(["team", "season"], group_keys=False)
    # shift(1): only games before this one
    t["net_ewm"] = grp["net"].apply(lambda s: s.shift(1).ewm(span=span, min_periods=1).mean())
    t["net_season"] = grp["net"].apply(lambda s: s.shift(1).expanding().mean())
    t["ortg_ewm"] = grp["ortg"].apply(lambda s: s.shift(1).ewm(span=span, min_periods=1).mean())
    t["drtg_ewm"] = grp["drtg"].apply(lambda s: s.shift(1).ewm(span=span, min_periods=1).mean())
    t["pace_ewm"] = grp["poss"].apply(lambda s: s.shift(1).ewm(span=span, min_periods=1).mean())
    t["wpct"] = grp["win"].apply(lambda s: s.shift(1).expanding().mean())
    t["games_played"] = grp.cumcount()
    prev_date = t.groupby("team")["date"].shift(1)
    t["rest"] = (t.date - prev_date).dt.days.clip(upper=7).fillna(7)
    t["b2b"] = (t.rest == 1).astype(int)
    # games in the last 5 days (fatigue)
    t["g5"] = t.groupby("team")["date"].transform(lambda s: _count_window(s.values, 5))
    feats = ["net_ewm", "net_season", "ortg_ewm", "drtg_ewm", "pace_ewm", "wpct", "games_played", "rest", "b2b", "g5"]
    for side, flag in (("h", True), ("a", False)):
        sub = t[t.is_home == flag][["GAME_ID"] + feats].rename(columns={f: f"{f}_{side}" for f in feats})
        g = g.merge(sub, on="GAME_ID", how="left")
    # early season: shrink season stats toward 0 with games played
    for side in ("h", "a"):
        w = g[f"games_played_{side}"] / (g[f"games_played_{side}"] + 10)
        g[f"net_shr_{side}"] = (w * g[f"net_season_{side}"].fillna(0))
        g[f"net_ewm_{side}"] = g[f"net_ewm_{side}"].fillna(0) * w
    g["net_diff"] = g.net_shr_h - g.net_shr_a
    g["form_diff"] = g.net_ewm_h - g.net_ewm_a
    g["rest_diff"] = g.rest_h - g.rest_a
    g["b2b_diff"] = g.b2b_h - g.b2b_a
    g["g5_diff"] = g.g5_h - g.g5_a
    g["wpct_diff"] = g.wpct_h.fillna(0.5) - g.wpct_a.fillna(0.5)
    return g


def _count_window(dates: np.ndarray, days: int) -> np.ndarray:
    """Number of games a team played in the `days` days before each game."""
    d = dates.astype("datetime64[D]").astype(np.int64)
    out = np.zeros(len(d))
    j = 0
    for i in range(len(d)):
        while d[j] < d[i] - days:
            j += 1
        out[i] = i - j
    return out


def build(logs: pd.DataFrame) -> pd.DataFrame:
    g = to_games(logs)
    g = add_elo(g)
    g = add_form(g)
    g["playoffs"] = (g.season_type != "Regular Season").astype(int)
    return g
