"""Leak-free pre-game features.

Training and live prediction share one code path: upcoming games are
appended to the history without a result, and every feature is computed
from games strictly before the row (shift(1) / Elo state before update).
"""
import numpy as np
import pandas as pd

ELO_K = 20.0
ELO_HOME = 70.0       # home advantage, Elo points
ELO_CARRY = 0.75      # share of last season's Elo kept at a new season
FORM_SPAN = 12        # EWM span (games) for recent form
SHRINK_GAMES = 10     # early-season shrinkage of season averages toward 0

MODEL_FEATURES = ["elo_diff", "net_diff", "form_diff", "rest_diff", "b2b_diff", "g5_diff"]


def _elo(g: pd.DataFrame) -> pd.DataFrame:
    elo, season_of = {}, {}
    pre_h, pre_a = np.empty(len(g)), np.empty(len(g))
    for i, row in enumerate(g.itertuples(index=False)):
        for t in (row.home, row.away):
            if t not in elo:
                elo[t] = 1500.0
            elif season_of[t] != row.season:
                elo[t] = 1500.0 + ELO_CARRY * (elo[t] - 1500.0)
            season_of[t] = row.season
        eh, ea = elo[row.home], elo[row.away]
        pre_h[i], pre_a[i] = eh, ea
        if np.isnan(row.margin):
            continue  # upcoming game: no update
        home_win = row.margin > 0
        diff = eh + ELO_HOME - ea
        p = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        winner_diff = diff if home_win else -diff
        mult = np.log(abs(row.margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = ELO_K * mult * (float(home_win) - p)
        elo[row.home] += delta
        elo[row.away] -= delta
    g = g.copy()
    g["home_elo"], g["away_elo"] = pre_h, pre_a
    g["elo_diff"] = g.home_elo + ELO_HOME - g.away_elo
    g["elo_win_prob"] = 1.0 / (1.0 + 10 ** (-g.elo_diff / 400.0))
    return g


def _team_rows(g: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, from that team's point of view."""
    parts = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = pd.DataFrame({
            "row": g.index, "date": g.date, "season": g.season, "team": g[side],
            "is_home": side == "home", "pts": g[f"{side}_pts"], "opp_pts": g[f"{opp}_pts"],
            "fgm": g[f"{side}_fgm"], "fga": g[f"{side}_fga"], "fg3m": g[f"{side}_fg3m"],
            "fg3a": g[f"{side}_fg3a"], "opp_fg3m": g[f"{opp}_fg3m"], "opp_fg3a": g[f"{opp}_fg3a"],
            "ast": g[f"{side}_ast"], "reb": g[f"{side}_reb"], "tov": g[f"{side}_tov"],
        })
        poss = lambda s: g[f"{s}_fga"] - g[f"{s}_oreb"] + g[f"{s}_tov"] + 0.44 * g[f"{s}_fta"]
        d["poss"] = (poss(side) + poss(opp)) / 2
        parts.append(d)
    t = pd.concat(parts).sort_values(["team", "date", "row"]).reset_index(drop=True)
    t["played"] = t.pts.notna()
    t["win"] = np.where(t.played, (t.pts > t.opp_pts).astype(float), np.nan)
    t["diff"] = t.pts - t.opp_pts
    t["ortg"] = 100 * t.pts / t.poss
    t["drtg"] = 100 * t.opp_pts / t.poss
    t["net"] = t.ortg - t.drtg
    t["fg_pct"] = t.fgm / t.fga
    t["fg3_pct"] = t.fg3m / t.fg3a
    t["opp_fg3_pct"] = t.opp_fg3m / t.opp_fg3a
    t["three_rate"] = t.fg3a / t.fga
    return t


def _prior_mean(s: pd.Series, n: int = None) -> pd.Series:
    """Mean of previous played values (last n if given), excluding current row."""
    prev = s.shift(1)
    if n is None:
        return prev.expanding().mean()
    return prev.rolling(n, min_periods=1).mean()


def _streak(wins: pd.Series) -> pd.Series:
    """Signed streak before each game (+3 = three straight wins)."""
    out, cur = [], 0
    for w in wins:
        out.append(cur)
        if np.isnan(w):
            continue
        cur = (cur + 1 if cur > 0 else 1) if w == 1 else (cur - 1 if cur < 0 else -1)
    return pd.Series(out, index=wins.index)


def _games_in_window(dates: pd.Series, days: int) -> np.ndarray:
    d = dates.values.astype("datetime64[D]").astype(np.int64)
    out, j = np.zeros(len(d)), 0
    for i in range(len(d)):
        while d[j] < d[i] - days:
            j += 1
        out[i] = i - j
    return out


def _team_features(t: pd.DataFrame) -> pd.DataFrame:
    # Season-scoped rolling stats only use played games: drop unplayed rows
    # from the rolling source but keep them as targets via reindex.
    feats = {}
    by_season = t.groupby(["team", "season"], group_keys=False)
    by_team = t.groupby("team", group_keys=False)

    def roll(col, n=None):
        # Last-N display stats run across seasons, so opening night still shows
        # last season's closing form; season-to-date stats (n=None) reset.
        grp = by_season if n is None else by_team
        return grp[col].apply(lambda s: _prior_mean(s, n))

    for n, tag in ((10, "last10"), (5, "last5"), (3, "last3")):
        feats[f"{tag}_win_pct"] = roll("win", n)
        feats[f"{tag}_net_rating"] = roll("net", n)
        feats[f"{tag}_point_diff"] = roll("diff", n)
        feats[f"{tag}_ppg"] = roll("pts", n)
        feats[f"{tag}_fg_pct"] = roll("fg_pct", n)
        feats[f"{tag}_pace"] = roll("poss", n)
    for col, name in (("ortg", "offensive_rating"), ("drtg", "defensive_rating"), ("fg3_pct", "fg3_pct"),
                      ("opp_fg3_pct", "opp_fg3_pct"), ("ast", "ast"), ("reb", "reb"), ("tov", "tov"),
                      ("opp_pts", "opp_ppg"), ("three_rate", "three_point_rate")):
        feats[f"last10_{name}"] = roll(col, 10)
    feats["season_diff"] = roll("diff")
    feats["season_win_pct"] = roll("win")
    feats["form_diff_ewm"] = by_season["diff"].apply(lambda s: s.shift(1).ewm(span=FORM_SPAN, min_periods=1).mean())
    feats["weighted_recent_form"] = by_season["win"].apply(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean())
    feats["games_played"] = by_season["played"].apply(lambda s: s.shift(1, fill_value=False).cumsum())
    feats["streak"] = by_season["win"].apply(_streak)
    f = pd.DataFrame(feats)
    f["form_acceleration"] = f["last3_win_pct"] - f["last10_win_pct"]

    prev_date = t.groupby("team")["date"].shift(1)
    f["rest_days"] = ((t.date - prev_date).dt.days - 1).clip(lower=0, upper=6).fillna(6)
    f["back_to_back"] = (f.rest_days == 0).astype(int)
    f["games_last5d"] = t.groupby("team")["date"].transform(lambda s: _games_in_window(s, 5))

    # Venue splits over the last 20 home (or road) games, across seasons
    by_venue = t.groupby(["team", "is_home"], group_keys=False)
    f["venue_win_pct"] = by_venue["win"].apply(lambda s: _prior_mean(s, 20))
    f["venue_ppg"] = by_venue["pts"].apply(lambda s: _prior_mean(s, 20))
    f["venue_point_diff"] = by_venue["diff"].apply(lambda s: _prior_mean(s, 20))
    f["venue_fg_pct"] = by_venue["fg_pct"].apply(lambda s: _prior_mean(s, 20))
    f["row"], f["is_home"] = t.row, t.is_home
    return f


def build(history: pd.DataFrame) -> pd.DataFrame:
    """Return one row per game in `history` with all pre-game features.

    `history` rows without points (upcoming games) get features too.
    """
    g = history.copy()
    g["date"] = pd.to_datetime(g.game_date)
    g = g.sort_values(["date", "home"]).reset_index(drop=True)
    g["margin"] = (g.home_pts - g.away_pts).astype(float)
    g = _elo(g)

    tf = _team_features(_team_rows(g))
    team_cols = [c for c in tf.columns if c not in ("row", "is_home")]
    for side, flag in (("home", True), ("away", False)):
        part = tf[tf.is_home == flag].set_index("row")[team_cols]
        part.columns = [f"{side}_{c}" for c in team_cols]
        g = g.join(part)

    for side in ("home", "away"):
        w = g[f"{side}_games_played"] / (g[f"{side}_games_played"] + SHRINK_GAMES)
        g[f"{side}_net_shrunk"] = w * g[f"{side}_season_diff"].fillna(0)
        g[f"{side}_form_shrunk"] = w * g[f"{side}_form_diff_ewm"].fillna(0)
    g["net_diff"] = g.home_net_shrunk - g.away_net_shrunk
    g["form_diff"] = g.home_form_shrunk - g.away_form_shrunk
    g["rest_diff"] = g.home_rest_days - g.away_rest_days
    g["b2b_diff"] = g.home_back_to_back - g.away_back_to_back
    g["g5_diff"] = g.home_games_last5d - g.away_games_last5d

    # Names expected by the thread/chart code
    g = g.rename(columns={
        "home_venue_win_pct": "home_team_home_win_pct", "home_venue_ppg": "home_team_home_ppg",
        "home_venue_point_diff": "home_team_home_point_diff", "home_venue_fg_pct": "home_team_home_fg_pct",
        "away_venue_win_pct": "away_team_road_win_pct", "away_venue_ppg": "away_team_road_ppg",
        "away_venue_point_diff": "away_team_road_point_diff", "away_venue_fg_pct": "away_team_road_fg_pct",
    })
    g["home_win"] = np.where(g.margin.notna(), (g.margin > 0).astype(float), np.nan)
    return g
