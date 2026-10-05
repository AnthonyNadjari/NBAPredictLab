"""Our own pre-game win probability, computed without any bookmaker price.

Production port of research/h7_own_model (Oct 2026), ensemble 'avg3': the mean of the logits of
three model families, each ported from its research run.py with the parameters frozen in
data/own_model_params.json (scripts/export_own_model_params.py writes them):

  elo_plus        margin-of-victory Elo, rest / back-to-back terms, player availability (regulars
                  who missed the previous game(s)) and roster talent        (elo_plus/run.py)
  margin_ratings  recency-weighted ridge (Massey) ratings with 3-point luck removed, schedule,
                  travel, late-season and previous-game-absence terms, probit (margin_ratings/run.py,
                  variant margin_full_avail; absences as research/h1_player_availability/build.py)
  player_impact   Kalman adjusted plus-minus + box-score rates, expected lineup from the last 10
                  team games, ridge rating gap + logit with team features (player_impact/run.py,
                  variant pregame_plus_team)

Inputs: data/games_history.csv (scores + team box scores) and data/player_games.csv (player box
scores, ESPN). Every run recomputes the three states from scratch. Each engine is a chronological
pass in which a date's games are described from the state after all earlier dates, so a game's
probability only ever uses games played strictly before its date (tests/test_own_model.py).

Fallback: when a team's recent player box scores are missing (no data, backfill not run, ESPN
field renamed...) or cover fewer than 82 of its earlier games, the game uses the team-only parts: Elo without availability ('mov_rest',
tuned without it) and the margin ratings without the absence term ('margin_full'), mode
'team_only'. Otherwise mode 'full'.
"""
import json
import logging
import math
from collections import defaultdict, deque
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import norm

from . import arenas
from .teams import TEAMS

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "data" / "own_model_params.json"
EPS = 1e-6
LN10_400 = math.log(10) / 400.0
BUBBLE_START = "2020-07-30"          # 2019-20 restart in Orlando: neutral site
ORLANDO_BUBBLE = (28.337, -81.556, "America/New_York", 30)
DAY0 = pd.Timestamp("2018-01-01")
PRESEASON = "Preseason"
FULL, TEAM_ONLY = "full", "team_only"
MIN_PLAYER_GAMES = 82   # 'full' mode needs player lines for >= 82 earlier games of each team

# elo_plus availability (elo_plus/run.py)
EP_ROT_MIN, EP_ROT_WINDOW, EP_VALUE_WINDOW = 12.0, 10, 82
EP_K_GS, EP_K_OO = 300.0, 2000.0
GAP_BUCKETS = [0, 1, 2, 3, 5, 10, 20, 10_000]
# margin_ratings absences (h1_player_availability/build.py)
H1_ROT_WINDOW, H1_ROT_MIN, H1_VALUE_WINDOW, H1_K_GS, H1_K_OO = 10, 15.0, 82, 300.0, 2000.0
SEASON_GAMES = {"2020-21": 72}       # scheduled regular-season games (82 otherwise)
SCHED_COLS = ["b2b_h", "b2b_a", "rest_diff", "g5_diff", "trav_diff", "alt_diff", "po_home"]
LATE_COLS = ["late_bad_h", "late_good_h", "late_bad_a", "late_good_a"]
# player_impact (player_impact/run.py)
BOX = ["PTS", "FGA", "FTA", "FG3M", "FG3A", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
HL_BOX, HL_PM, K_BOX, K_PM, K_OO = 60.0, 120.0, 1500.0, 4000.0, 4000.0
WIN, N_MIN = 10, 5
RATE_COLS = [f"r_{c}" for c in BOX] + ["r_PM", "r_ONOFF"]
XCOLS = RATE_COLS + ["r_KAL"]
LOGIT_COLS = ["gap", "rest_diff", "b2b_diff", "g5_diff", "elo_diff", "net_diff", "form_diff"]

_STORE_TO_LOG = {"min": "MIN", "pts": "PTS", "fgm": "FGM", "fga": "FGA", "fg3m": "FG3M", "fg3a": "FG3A",
                 "ftm": "FTM", "fta": "FTA", "oreb": "OREB", "dreb": "DREB", "reb": "REB", "ast": "AST",
                 "stl": "STL", "blk": "BLK", "tov": "TOV", "pf": "PF", "plus_minus": "PLUS_MINUS"}


def load_params(path: Path = PARAMS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigm(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


# ============================================================================ game table
def build_games(hist: pd.DataFrame, upcoming: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Research-style game table (one row per game, sorted by date) from the history store, plus
    upcoming games without a result, with research/features.py's pre-game features."""
    h = hist[hist.home.isin(TEAMS) & hist.away.isin(TEAMS)]   # no All-Star (STARS/WORLD...) games
    if upcoming is not None and len(upcoming):
        up = upcoming[["game_id", "game_date", "season", "season_type", "home", "away"]]
        keys = set(zip(up.game_date, up.home, up.away))
        h = h[[k not in keys for k in zip(h.game_date, h.home, h.away)]]
        h = pd.concat([h, up], ignore_index=True)
    num = lambda c: pd.to_numeric(h[c], errors="coerce").to_numpy(float) if c in h else np.full(len(h), np.nan)
    g = pd.DataFrame({"GAME_ID": h.game_id.astype(str).to_numpy(), "date": pd.to_datetime(h.game_date).to_numpy(),
                      "season": h.season.to_numpy(), "season_type": h.season_type.to_numpy(),
                      "home": h.home.to_numpy(), "away": h.away.to_numpy()})
    for side, s in (("h", "home"), ("a", "away")):
        g[f"PTS_{side}"] = num(f"{s}_pts")
        for k in ("fgm", "fga", "fg3m", "fg3a", "fta", "tov", "oreb"):
            g[f"{k.upper()}_{side}"] = num(f"{s}_{k}")
        g[f"poss_{side}"] = g[f"FGA_{side}"] - g[f"OREB_{side}"] + g[f"TOV_{side}"] + 0.44 * g[f"FTA_{side}"]
    g["home_id"], g["away_id"] = g.home, g.away
    g["margin"] = g.PTS_h - g.PTS_a
    g["played"] = g.margin.notna()
    g["home_win"] = np.where(g.played, (g.margin > 0).astype(float), np.nan)
    g = g.sort_values(["date", "GAME_ID"]).reset_index(drop=True)
    return _research_features(g)


def _count_window(dates: np.ndarray, days: int) -> np.ndarray:
    d = dates.astype("datetime64[D]").astype(np.int64)
    out = np.zeros(len(d))
    j = 0
    for i in range(len(d)):
        while d[j] < d[i] - days:
            j += 1
        out[i] = i - j
    return out


def _research_features(g: pd.DataFrame) -> pd.DataFrame:
    """research/features.py add_elo + add_form (plain Elo, net rating, form, rest, games in 5 days,
    win%), from earlier games only. Upcoming rows get features and never update anything."""
    elo, last = {}, {}
    pre_h, pre_a = np.empty(len(g)), np.empty(len(g))
    for i, (h, a, s, m) in enumerate(zip(g.home_id, g.away_id, g.season, g.margin)):
        for t in (h, a):
            if t not in elo:
                elo[t] = 1500.0
            elif last[t] != s:
                elo[t] = 1500.0 + 0.75 * (elo[t] - 1500.0)
            last[t] = s
        eh, ea = elo[h], elo[a]
        pre_h[i], pre_a[i] = eh, ea
        if m != m:
            continue
        hw = 1 if m > 0 else 0
        diff = eh + 70.0 - ea
        p = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        mult = np.log(abs(m) + 1) * 2.2 / ((diff if hw else -diff) * 0.001 + 2.2)
        delta = 20.0 * mult * (hw - p)
        elo[h] += delta
        elo[a] -= delta
    g = g.copy()
    g["elo_diff"] = pre_h + 70.0 - pre_a
    g["elo_prob"] = 1.0 / (1.0 + 10 ** (-g.elo_diff / 400.0))

    parts = []
    for side, opp in (("h", "a"), ("a", "h")):
        parts.append(pd.DataFrame({
            "GAME_ID": g.GAME_ID, "date": g.date, "season": g.season, "team": g[f"{'home' if side == 'h' else 'away'}_id"],
            "is_home": side == "h", "pts": g[f"PTS_{side}"], "opp_pts": g[f"PTS_{opp}"],
            "poss": (g[f"poss_{side}"] + g[f"poss_{opp}"]) / 2,
            "win": g.home_win if side == "h" else 1 - g.home_win}))
    t = pd.concat(parts).sort_values(["team", "date", "GAME_ID"]).reset_index(drop=True)
    t["net"] = 100 * t.pts / t.poss - 100 * t.opp_pts / t.poss
    grp = t.groupby(["team", "season"], group_keys=False)
    t["net_ewm"] = grp["net"].apply(lambda s: s.shift(1).ewm(span=12, min_periods=1).mean())
    t["net_season"] = grp["net"].apply(lambda s: s.shift(1).expanding().mean())
    t["wpct"] = grp["win"].apply(lambda s: s.shift(1).expanding().mean())
    t["games_played"] = grp.cumcount()
    prev_date = t.groupby("team")["date"].shift(1)
    t["rest"] = (t.date - prev_date).dt.days.clip(upper=7).fillna(7)
    t["b2b"] = (t.rest == 1).astype(int)
    t["g5"] = t.groupby("team")["date"].transform(lambda s: _count_window(s.values, 5))
    feats = ["net_ewm", "net_season", "wpct", "games_played", "rest", "b2b", "g5"]
    for side, flag in (("h", True), ("a", False)):
        sub = t[t.is_home == flag][["GAME_ID"] + feats].rename(columns={f: f"{f}_{side}" for f in feats})
        g = g.merge(sub, on="GAME_ID", how="left")
    for side in ("h", "a"):
        w = g[f"games_played_{side}"] / (g[f"games_played_{side}"] + 10)
        g[f"net_shr_{side}"] = w * g[f"net_season_{side}"].fillna(0)
        g[f"net_ewm_{side}"] = g[f"net_ewm_{side}"].fillna(0) * w
    g["net_diff"] = g.net_shr_h - g.net_shr_a
    g["form_diff"] = g.net_ewm_h - g.net_ewm_a
    g["rest_diff"] = g.rest_h - g.rest_a
    g["b2b_diff"] = g.b2b_h - g.b2b_a
    g["g5_diff"] = g.g5_h - g.g5_a
    return g


# ============================================================================ player tables
def build_player_logs(players: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Research-format player logs from the store. GAME_ID is aligned with the game table by
    (date, team): old history rows carry nba_api ids, the store ESPN ids."""
    cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM", "GAME_ID", "date", "SEASON", "SEASON_TYPE", "home"] + \
        list(_STORE_TO_LOG.values())
    if players is not None and len(players):
        players = players[players.team.isin(TEAMS)]      # All-Star / exhibition teams
    if players is None or not len(players):
        return pd.DataFrame(columns=cols)
    p = pd.DataFrame({"PLAYER_ID": players.player_id.astype(str).to_numpy(),
                      "PLAYER_NAME": players.player.to_numpy(), "TEAM": players.team.to_numpy(),
                      "date": pd.to_datetime(players.game_date).to_numpy(), "SEASON": players.season.to_numpy(),
                      "SEASON_TYPE": players.season_type.to_numpy(),
                      "home": pd.to_numeric(players.home, errors="coerce").fillna(0).astype(int).to_numpy()})
    for src, dst in _STORE_TO_LOG.items():
        p[dst] = pd.to_numeric(players[src], errors="coerce").fillna(0.0).astype(float).to_numpy()
    key = {}
    for gid, d, h, a in zip(g.GAME_ID, g.date, g.home, g.away):
        key[(d, h)] = gid
        key[(d, a)] = gid
    own = players.game_id.astype(str).to_numpy()
    p["GAME_ID"] = [key.get((d, t), o) for d, t, o in zip(p.date, p.TEAM, own)]
    p = p.drop_duplicates(["PLAYER_ID", "GAME_ID"])
    return p.sort_values(["date", "GAME_ID", "TEAM", "PLAYER_ID"]).reset_index(drop=True)[cols]


def build_team_games(p: pd.DataFrame) -> pd.DataFrame:
    """Team logs from the player logs (team totals = sums of the player lines, as in nba_api):
    possessions (mean of the two teams), team minutes, margin, home flag."""
    a = p.groupby(["GAME_ID", "TEAM"], sort=False).agg(
        date=("date", "first"), season=("SEASON", "first"), team_min=("MIN", "sum"), pts=("PTS", "sum"),
        fga=("FGA", "sum"), oreb=("OREB", "sum"), tov=("TOV", "sum"), fta=("FTA", "sum"),
        is_home=("home", "max")).reset_index()
    a["poss_t"] = a.fga - a.oreb + a.tov + 0.44 * a.fta
    a["poss"] = a.groupby("GAME_ID").poss_t.transform("mean")
    tot = a.groupby("GAME_ID").pts.transform("sum")
    n = a.groupby("GAME_ID").pts.transform("size")
    a["margin"] = np.where(n == 2, 2 * a.pts - tot, np.nan)
    a["is_home"] = a.is_home.astype(bool)
    return a[["TEAM", "GAME_ID", "date", "season", "poss", "team_min", "margin", "is_home"]]


def _targets(g: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Team-games (both sides) of the game-table rows `rows`."""
    s = g.loc[rows]
    return pd.concat([pd.DataFrame({"GAME_ID": s.GAME_ID, "TEAM": s.home, "date": s.date, "season": s.season}),
                      pd.DataFrame({"GAME_ID": s.GAME_ID, "TEAM": s.away, "date": s.date, "season": s.season})],
                     ignore_index=True)


# ============================================================================ elo_plus
def elo_availability(p: pd.DataFrame, targets: pd.DataFrame, repl: float) -> pd.DataFrame:
    """elo_plus build_availability: one row per (team-game, regular player) with pre-game values,
    the consecutive team games missed and whether he actually played. One pass over dates: every
    team-game of date D (played or upcoming) is described from the state after D-1, then D's box
    scores are ingested. Only played team-games (with player lines) move the counters."""
    out_cols = ["GAME_ID", "TEAM", "v_min", "v_gs", "v_oo", "gap", "absent"]
    if not len(p):
        return pd.DataFrame(columns=out_cols)
    p = p.copy()
    p["gmsc"] = (p.PTS + 0.4 * p.FGM - 0.7 * p.FGA - 0.4 * (p.FTA - p.FTM) + 0.7 * p.OREB + 0.3 * p.DREB
                 + p.STL + 0.7 * p.AST + 0.7 * p.BLK - 0.4 * p.PF - p.TOV)
    tm = p.groupby(["GAME_ID", "TEAM"]).agg(team_pts=("PTS", "sum"), team_min=("MIN", "sum")).reset_index()
    opp = tm.rename(columns={"TEAM": "OPP", "team_pts": "opp_pts", "team_min": "opp_min"})
    tm = tm.merge(opp, on="GAME_ID")
    tm = tm[tm.TEAM != tm.OPP]
    tm["team_margin"] = tm.team_pts - tm.opp_pts
    tm["game_min"] = (tm.team_min / 5.0).clip(lower=48.0)
    p = p.merge(tm[["GAME_ID", "TEAM", "team_margin", "game_min"]], on=["GAME_ID", "TEAM"], how="left")
    p["off_pm"] = p.team_margin - p.PLUS_MINUS
    p["off_min"] = (p.game_min - p.MIN).clip(lower=0)
    p = p.sort_values(["date", "GAME_ID", "TEAM"]).reset_index(drop=True)

    win = defaultdict(lambda: deque(maxlen=EP_VALUE_WINDOW))
    sums = defaultdict(lambda: np.zeros(5))
    last_team, team_players = {}, defaultdict(set)
    team_mins = defaultdict(lambda: deque(maxlen=EP_ROT_WINDOW))
    team_ngames, last_idx = defaultdict(int), {}
    cols = ["MIN", "gmsc", "PLUS_MINUS", "off_pm", "off_min"]
    by_p = dict(tuple(p.groupby("date", sort=True)))
    tgt = targets[targets.date >= p.date.min()]
    by_t = {d: list(zip(x.GAME_ID, x.TEAM, x.season)) for d, x in tgt.groupby("date")}
    out = []
    for d in sorted(set(by_p) | set(by_t)):
        day = by_p.get(d)
        tg_p = list(day[["GAME_ID", "TEAM", "SEASON"]].drop_duplicates().itertuples(index=False, name=None)) \
            if day is not None else []
        seen = {(x[0], x[1]) for x in tg_p}
        tgs = tg_p + [x for x in by_t.get(d, []) if (x[0], x[1]) not in seen]
        played = set(zip(day.GAME_ID, day.TEAM, day.PLAYER_ID)) if day is not None else set()
        for gid, tid, sea in tgs:
            key = (tid, sea)
            j = team_ngames[key]
            for pid in team_players[key]:
                if last_team.get(pid) != tid:
                    continue
                mins = team_mins[(tid, sea, pid)]
                exp_min = sum(mins) / len(mins)
                if exp_min < EP_ROT_MIN:
                    continue
                s = sums[pid]
                gs_rate = (s[1] + repl * EP_K_GS) / (s[0] + EP_K_GS)
                on48 = 48 * s[2] / max(s[0], 1.0)
                off48 = 48 * s[3] / max(s[4], 1.0)
                oo = (on48 - off48) * s[0] / (s[0] + EP_K_OO)
                out.append((gid, tid, exp_min, max(gs_rate - repl, 0.0) * exp_min, oo * exp_min / 48.0,
                            j - 1 - last_idx[(tid, sea, pid)], int((gid, tid, pid) not in played)))
        if day is None:
            continue
        for gid, tid, sea in tg_p:
            team_ngames[(tid, sea)] += 1
        for r in day[["GAME_ID", "TEAM", "SEASON", "PLAYER_ID"] + cols].itertuples(index=False):
            tid, sea, pid = r[1], r[2], r[3]
            vals = np.array(r[4:], dtype=float)
            w = win[pid]
            if len(w) == EP_VALUE_WINDOW:
                sums[pid] -= w[0]
            w.append(vals)
            sums[pid] += vals
            last_team[pid] = tid
            team_players[(tid, sea)].add(pid)
            team_mins[(tid, sea, pid)].append(vals[0])
            last_idx[(tid, sea, pid)] = team_ngames[(tid, sea)] - 1
    return pd.DataFrame(out, columns=out_cols)


def _gap_bucket(gap):
    return np.searchsorted(GAP_BUCKETS, np.asarray(gap), side="right") - 1


def elo_team_game_availability(a: pd.DataFrame, pout) -> pd.DataFrame:
    """Per (GAME_ID, TEAM): expected missing value (pre-game, from the gaps), actual missing value
    (who did not play; only used in the post-game update) and the regulars' total value."""
    pout = np.asarray(pout, float)
    w = pout[_gap_bucket(a.gap.to_numpy(int))] if len(a) else np.zeros(0)
    x = pd.DataFrame({"GAME_ID": a.GAME_ID, "TEAM": a.TEAM})
    for v in ("v_gs", "v_min", "v_oo"):
        x[f"exp_{v[2:]}"] = w * a[v]
        x[f"act_{v[2:]}"] = a.absent * a[v]
        x[f"tot_{v[2:]}"] = a[v]
    return x.groupby(["GAME_ID", "TEAM"]).sum().reset_index()


def _elo_arrays(g: pd.DataFrame, av_tg: Optional[pd.DataFrame]) -> dict:
    tid = {t: i for i, t in enumerate(sorted(set(g.home_id) | set(g.away_id)))}
    sid = {s: i for i, s in enumerate(sorted(g.season.unique()))}
    rest3_h, rest3_a = (g.rest_h >= 3).astype(float), (g.rest_a >= 3).astype(float)
    A = {"H": g.home_id.map(tid).tolist(), "W": g.away_id.map(tid).tolist(), "S": g.season.map(sid).tolist(),
         "Y": g.home_win.tolist(), "PL": g.played.tolist(), "M": g.margin.abs().astype(float).tolist(),
         "NEU": ((g.season == "2019-20") & (g.date >= BUBBLE_START)).astype(int).tolist(),
         "PO": (g.season_type != "Regular Season").astype(int).tolist(),
         "b2b_h": g.b2b_h.to_numpy(float), "b2b_a": g.b2b_a.to_numpy(float),
         "rest3_h": rest3_h.to_numpy(float), "rest3_a": rest3_a.to_numpy(float),
         "dense_h": (g.g5_h >= 3).to_numpy(float), "dense_a": (g.g5_a >= 3).to_numpy(float)}
    for side, col in (("h", "home_id"), ("a", "away_id")):
        if av_tg is not None and len(av_tg):
            m = g[["GAME_ID", col]].merge(av_tg, left_on=["GAME_ID", col], right_on=["GAME_ID", "TEAM"], how="left")
        else:
            m = pd.DataFrame(index=range(len(g)))
        for kind in ("exp", "act", "tot"):
            for v in ("gs", "min", "oo"):
                c = f"{kind}_{v}"
                A[f"{c}_{side}"] = m[c].fillna(0.0).to_numpy(float) if c in m else np.zeros(len(g))
    both = (A["tot_min_h"] > 0) & (A["tot_min_a"] > 0)
    A["tal_gs"] = np.where(both, A["tot_gs_h"] - A["tot_gs_a"], 0.0)
    A["tal_oo"] = np.where(both, A["tot_oo_h"] - A["tot_oo_a"], 0.0)
    return A


def _elo_adjustments(par: dict, A: dict, kind: Optional[str]) -> np.ndarray:
    x = (-par["c_b2b"] * (A["b2b_h"] - A["b2b_a"]) + par["c_rest3"] * (A["rest3_h"] - A["rest3_a"])
         - par["c_dense"] * (A["dense_h"] - A["dense_a"]))
    if kind is not None:
        for v in ("gs", "min", "oo"):
            b = par[f"b_{v}"]
            if b:
                x = x - b * (A[f"{kind}_{v}_h"] - A[f"{kind}_{v}_a"])
        if par["b_tal"]:
            x = x + par["b_tal"] * A["tal_gs"]
        if par["b_tal_oo"]:
            x = x + par["b_tal_oo"] * A["tal_oo"]
    return x


def elo_pass(par: dict, A: dict, avail: Optional[str]) -> np.ndarray:
    """elo_plus elo_pass: pre-game P(home win) for every row. avail=None: no availability term
    (variant mov_rest); 'post': expected absences before the game, actual ones in the update
    (variant elo_plus). Unplayed rows get a probability and never update the ratings."""
    xp = _elo_adjustments(par, A, None if avail is None else "exp").tolist()
    xu = _elo_adjustments(par, A, "act").tolist() if avail == "post" else xp
    H, W, S, Y, M, NEU, PO, PL = A["H"], A["W"], A["S"], A["Y"], A["M"], A["NEU"], A["PO"], A["PL"]
    k, alpha, ac, carry = par["k"], par["alpha"], par["ac"], par["carry"]
    hca, hca_po, k_early, tau, temp = par["hca"], par["hca_po"], par["k_early"], par["tau_early"], par["temp"]
    early = [math.exp(-n / tau) for n in range(120)]
    n_teams = max(max(H, default=0), max(W, default=0)) + 1
    R, ls, gp = [1500.0] * n_teams, [-1] * n_teams, [0] * n_teams
    out = np.empty(len(H))
    for i in range(len(H)):
        h, a, s = H[i], W[i], S[i]
        if ls[h] != s:
            R[h] = 1500.0 + carry * (R[h] - 1500.0)
            ls[h], gp[h] = s, 0
        if ls[a] != s:
            R[a] = 1500.0 + carry * (R[a] - 1500.0)
            ls[a], gp[a] = s, 0
        hc = 0.0 if NEU[i] else (hca + hca_po if PO[i] else hca)
        base = R[h] - R[a] + hc
        p = 1.0 / (1.0 + math.exp(-temp * (base + xp[i]) * LN10_400))
        out[i] = p
        if not PL[i]:
            continue
        y = Y[i]
        du = base + xu[i]
        pu = p if (xu is xp and temp == 1.0) else 1.0 / (1.0 + math.exp(-du * LN10_400))
        den = 7.5 + ac * (du if y else -du)
        if den < 1.0:
            den = 1.0
        kf = k * (1.0 + k_early * 0.5 * (early[min(gp[h], 119)] + early[min(gp[a], 119)]))
        delta = kf * (M[i] + 3.0) ** alpha / den * (y - pu)
        R[h] += delta
        R[a] -= delta
        gp[h] += 1
        gp[a] += 1
    return out


# ============================================================================ margin_ratings
def _haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def _margin_frame(g: pd.DataFrame) -> pd.DataFrame:
    """margin_ratings load_games: team index, day number, neutral sites, travel and altitude."""
    g = g.copy()
    teams = sorted(set(g.home_id) | set(g.away_id))
    tix = {t: i for i, t in enumerate(teams)}
    g["hi"], g["ai"] = g.home_id.map(tix), g.away_id.map(tix)
    g["day"] = ((g.date - DAY0).dt.days).astype(int)
    ds = g.date.dt.strftime("%Y-%m-%d")
    g["neutral_code"] = [arenas.NEUTRAL_GAMES.get((d, frozenset({h, a}))) for d, h, a in zip(ds, g.home, g.away)]
    g["bubble"] = (g.season == "2019-20") & (g.date >= BUBBLE_START)
    g["neutral"] = (g.neutral_code.notna() | g.bubble).astype(int)
    g["poss_game"] = (g.poss_h + g.poss_a) / 2
    venue = [ORLANDO_BUBBLE if b else (arenas.NEUTRAL[c] if isinstance(c, str) else arenas.home_arena(h, s))
             for b, c, h, s in zip(g.bubble, g.neutral_code, g.home, g.season)]
    last = {}
    trav = {"home": np.zeros(len(g)), "away": np.zeros(len(g))}
    alt = {"home": np.zeros(len(g)), "away": np.zeros(len(g))}
    for i, (h, a, s, v) in enumerate(zip(g.home, g.away, g.season, venue)):
        for side, t in (("home", h), ("away", a)):
            own = arenas.home_arena(t, s)
            prev = last.get((t, s), own)
            trav[side][i] = _haversine_km(prev, v) / 1000.0
            alt[side][i] = float(v[3] > 1000 and own[3] < 1000)
            last[(t, s)] = v
    g["trav_h"], g["trav_a"] = trav["home"], trav["away"]
    g["alt_h"], g["alt_a"] = alt["home"], alt["away"]
    return g


def _shooting_luck(g):
    """Points from 3-point shooting above the running league rate (earlier dates only). 0 for
    unplayed games and games without a box score."""
    made, att = g.FG3M_h + g.FG3M_a, g.FG3A_h + g.FG3A_a
    ok = made.notna() & att.notna() & g.played
    day_m = made.where(ok, 0).groupby(g.day).sum().cumsum().shift(1)
    day_a = att.where(ok, 0).groupby(g.day).sum().cumsum().shift(1)
    rate = (day_m / day_a).reindex(g.day).fillna(0.355).to_numpy()
    lh = 3 * (g.FG3M_h.to_numpy() - g.FG3A_h.to_numpy() * rate)
    la = 3 * (g.FG3M_a.to_numpy() - g.FG3A_a.to_numpy() * rate)
    return np.nan_to_num(np.where(ok, lh, 0.0)), np.nan_to_num(np.where(ok, la, 0.0))


def _schedule_terms(g: pd.DataFrame, cut: float) -> pd.DataFrame:
    rest_h, rest_a = g.rest_h.clip(1, 4), g.rest_a.clip(1, 4)
    po = (g.season_type != "Regular Season").astype(float)
    S = pd.DataFrame({
        "b2b_h": g.b2b_h.astype(float), "b2b_a": g.b2b_a.astype(float),
        "rest_diff": (rest_h - rest_a).astype(float), "g5_diff": g.g5_diff.astype(float),
        "trav_diff": (g.trav_h - g.trav_a).astype(float), "alt_diff": (g.alt_h - g.alt_a).astype(float),
        "po_home": po * (1 - g.neutral)}, index=g.index)
    n_sched = g.season.map(SEASON_GAMES).fillna(82).to_numpy()
    rs = (g.season_type == "Regular Season").to_numpy(float)
    for side in ("h", "a"):
        late = (g[f"games_played_{side}"].to_numpy() / n_sched >= cut) * rs
        w = g[f"wpct_{side}"].fillna(0.5).to_numpy()
        S[f"late_bad_{side}"] = late * (w < 0.40)
        S[f"late_good_{side}"] = late * (w > 0.65)
    return S[SCHED_COLS + LATE_COLS]


def margin_rate(g, hl=50.0, carry=0.7, k=8.0, cap=np.inf, per100=False, k_hca=60.0, adj=None):
    """margin_ratings rate(): sequential decayed ridge / Massey ratings. Each date's games are
    predicted from the fit on earlier dates, then added (only played games enter the fit)."""
    n = len(g)
    T = int(max(g.hi.max(), g.ai.max())) + 1
    d = g.day.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(d)) + 1]
    ends = np.r_[starts[1:], n]
    blocks = list(zip(g.season.to_numpy()[starts], starts, ends))
    hi, ai = g.hi.to_numpy(), g.ai.to_numpy()
    home_flag = 1.0 - g.neutral.to_numpy()
    y = g.margin.to_numpy(float)
    if adj is not None:
        y = y - adj
    y = np.clip(y, -cap, cap)
    poss = g.poss_game.to_numpy(float)
    if per100:
        y = 100.0 * y / poss
    fit = np.isfinite(y)
    fit_p = np.isfinite(poss) & g.played.to_numpy()
    P = T + 1
    r = np.arange(n)
    X = np.zeros((n, P))
    X[r, hi], X[r, ai], X[:, T] = 1.0, -1.0, home_flag
    Xp = np.zeros((n, P))
    Xp[r, hi], Xp[r, ai], Xp[:, T] = 1.0, 1.0, 1.0
    mu_out, pace_out = np.empty(n), np.empty(n)
    theta_end = np.zeros(P)
    pace_end = np.r_[np.zeros(T), 100.0]
    lam = np.r_[np.full(T, k), k_hca]
    lam_p = np.r_[np.full(T, k), 1.0]
    D_lam, D_lam_p = np.diag(lam), np.diag(lam_p)
    cur, A = None, None
    last = b = Ap = bp = prior = prior_p = None
    for season, s, e in blocks:
        if season != cur:
            if A is not None:
                theta_end = np.linalg.solve(A + D_lam, b + lam * prior)
                pace_end = np.linalg.solve(Ap + D_lam_p, bp + lam_p * prior_p)
            cur = season
            rr = theta_end[:T] - theta_end[:T].mean()
            prior = np.r_[carry * rr, theta_end[T]]
            pr = pace_end[:T] - pace_end[:T].mean()
            prior_p = np.r_[carry * pr, pace_end[T] + 2 * pace_end[:T].mean()]
            A, b = np.zeros((P, P)), np.zeros(P)
            Ap, bp = np.zeros((P, P)), np.zeros(P)
            last = None
        if last is not None and np.isfinite(hl):
            f = 0.5 ** ((d[s] - last) / hl)
            A *= f
            b *= f
            Ap *= f
            bp *= f
        theta = np.linalg.solve(A + D_lam, b + lam * prior)
        mu_out[s:e] = X[s:e] @ theta
        if per100:
            th_p = np.linalg.solve(Ap + D_lam_p, bp + lam_p * prior_p)
            pace_out[s:e] = Xp[s:e] @ th_p
            m = fit_p[s:e]
            Ap += Xp[s:e][m].T @ Xp[s:e][m]
            bp += Xp[s:e][m].T @ poss[s:e][m]
        m = fit[s:e]
        Xb = X[s:e][m]
        A += Xb.T @ Xb
        b += Xb.T @ y[s:e][m]
        last = d[s]
    if per100:
        mu_out = mu_out * pace_out / 100.0
    return mu_out


def h1_player_values(p: pd.DataFrame, repl: float) -> pd.DataFrame:
    """h1 player_values: shrunk GameScore/min over each player's last 82 appearances, after each
    appearance (looked up later with 'last appearance strictly before the game date')."""
    p = p.copy()
    p["gmsc"] = (p.PTS + 0.4 * p.FGM - 0.7 * p.FGA - 0.4 * (p.FTA - p.FTM) + 0.7 * p.OREB + 0.3 * p.DREB
                 + p.STL + 0.7 * p.AST + 0.7 * p.BLK - 0.4 * p.PF - p.TOV)
    p = p.sort_values(["PLAYER_ID", "date"])
    grp = p.groupby("PLAYER_ID", sort=False)
    s_min = grp["MIN"].rolling(H1_VALUE_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
    s_gs = grp["gmsc"].rolling(H1_VALUE_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
    p["gs_rate"] = (s_gs + repl * H1_K_GS) / (s_min + H1_K_GS)
    return p[["PLAYER_ID", "date", "gs_rate"]]


def h1_absences(p: pd.DataFrame, T: pd.DataFrame, targets: pd.DataFrame, repl: float) -> pd.DataFrame:
    """h1 availability (absent_prev x v_gs): per target team-game, the summed prior value of
    rotation players (>= 15 min over their last 10 appearances for the team this season, last
    seen with this team) who did not play in the team's previous game. Only played team-games
    (with player lines) form the windows."""
    rows = []
    tg_all = T.sort_values(["date", "GAME_ID"])
    for team, tt in targets.groupby("TEAM"):
        tg = tg_all[tg_all.TEAM == team].reset_index(drop=True)
        if not len(tg):
            continue
        pt = p[p.TEAM == team]
        players = pt.PLAYER_ID.unique()
        pidx = {pid: k for k, pid in enumerate(players)}
        gidx = {gid: k for k, gid in enumerate(tg.GAME_ID)}
        minutes = np.full((len(tg), len(players)), np.nan)
        rr = pt[pt.GAME_ID.isin(gidx)]
        minutes[rr.GAME_ID.map(gidx).to_numpy(), rr.PLAYER_ID.map(pidx).to_numpy()] = rr.MIN.to_numpy()
        played = ~np.isnan(minutes)
        seasons = tg.season.to_numpy()
        dates = tg.date.to_numpy()
        for gid, d, season in zip(tt.GAME_ID, tt.date, tt.season):
            j = int(np.searchsorted(dates, np.datetime64(d), side="left"))
            lo = j
            while lo > 0 and seasons[lo - 1] == season and j - lo < H1_ROT_WINDOW:
                lo -= 1
            if lo == j:
                continue
            s0 = lo
            while s0 > 0 and seasons[s0 - 1] == season:
                s0 -= 1
            for c in np.where(played[lo:j].any(axis=0))[0]:
                mins = minutes[s0:j, c]
                mins = mins[~np.isnan(mins)][-H1_ROT_WINDOW:]
                exp_min = mins.mean()
                if exp_min < H1_ROT_MIN:
                    continue
                rows.append((team, gid, d, players[c], exp_min, not played[j - 1, c]))
    a = pd.DataFrame(rows, columns=["TEAM", "GAME_ID", "date", "PLAYER_ID", "exp_min", "absent_prev"])
    if not len(a):
        return pd.DataFrame(columns=["TEAM", "GAME_ID", "miss"])
    a["date"] = pd.to_datetime(a.date)
    lt = p[["PLAYER_ID", "date", "TEAM"]].rename(columns={"date": "last_date", "TEAM": "last_team"})
    a = pd.merge_asof(a.sort_values("date"), lt.sort_values("last_date"), left_on="date", right_on="last_date",
                      by="PLAYER_ID", allow_exact_matches=False, direction="backward")
    a = a[a.last_team == a.TEAM].drop(columns=["last_team", "last_date"])
    v = h1_player_values(p, repl).rename(columns={"date": "vdate"})
    a = pd.merge_asof(a.sort_values("date"), v.sort_values("vdate"), left_on="date", right_on="vdate",
                      by="PLAYER_ID", allow_exact_matches=False, direction="backward")
    a["miss"] = a.absent_prev.astype(float) * (a.gs_rate - repl) * a.exp_min
    return a.groupby(["TEAM", "GAME_ID"]).miss.sum().reset_index()


def margin_probs(g: pd.DataFrame, miss: Optional[pd.DataFrame], mp: dict, players_from: Optional[str]):
    """(P team-only 'margin_full', P with absences 'margin_full_avail') for every row of g."""
    m = _margin_frame(g)
    S = _schedule_terms(m, mp["late_cut"])
    adj_f = S.to_numpy(float) @ np.array([mp["beta"][c] for c in S.columns])
    lh, la = _shooting_luck(m)
    cap = np.inf if mp["cap"] is None else mp["cap"]
    mu_f = margin_rate(m, hl=mp["hl"], carry=mp["carry"], k=mp["k"], cap=cap, per100=mp["per100"],
                       adj=mp["alpha"] * (lh - la) + adj_f) + adj_f
    xa = np.zeros(len(m))
    if miss is not None and len(miss) and players_from is not None:
        for side, col in (("h", "home"), ("a", "away")):
            x = m[["GAME_ID", col]].merge(miss.rename(columns={"TEAM": col}), on=["GAME_ID", col], how="left")
            xa = xa + (1 if side == "h" else -1) * x.miss.fillna(0).to_numpy()
        xa = np.where((m.season >= players_from).to_numpy(), xa, 0.0)
    p_team = norm.cdf(mu_f / mp["sigma_team"])
    p_avail = norm.cdf((mu_f + mp["beta_avail"] * xa) / mp["sigma_avail"])
    return p_team, p_avail


# ============================================================================ player_impact
def pi_player_states(p: pd.DataFrame, T: pd.DataFrame, prior: dict) -> pd.DataFrame:
    """player_impact player_states: recency-weighted per-100 rates after each appearance, shrunk
    toward replacement level (prior, frozen in the parameters)."""
    q = p.merge(T[["TEAM", "GAME_ID", "poss", "team_min", "margin"]], on=["TEAM", "GAME_ID"], how="left")
    q["team_min"] = q.team_min.fillna(240.0)
    q["poss"] = q.poss.fillna(q.poss.median())
    q["margin"] = q.margin.fillna(0.0)
    q["on_poss"] = q.MIN / (q.team_min / 5.0) * q.poss
    q["off_poss"] = (q.poss - q.on_poss).clip(lower=0)
    q["off_pm"] = q.margin - q.PLUS_MINUS
    q = q.sort_values(["PLAYER_ID", "date"]).reset_index(drop=True)
    prior_box = np.array([prior[f"r_{c}"] for c in BOX])
    prior_pm = prior["r_PM"]
    lam_b, lam_p = 0.5 ** (1 / HL_BOX), 0.5 ** (1 / HL_PM)
    xb = q[["on_poss"] + BOX].to_numpy(float)
    xp = q[["on_poss", "PLUS_MINUS", "off_pm", "off_poss"]].to_numpy(float)
    sb, sp = np.empty_like(xb), np.empty_like(xp)
    pid = q.PLAYER_ID.to_numpy()
    cut = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(q)]
    for a, b in zip(cut[:-1], cut[1:]):
        sb[a:b] = lfilter([1.0], [1.0, -lam_b], xb[a:b], axis=0)
        sp[a:b] = lfilter([1.0], [1.0, -lam_p], xp[a:b], axis=0)
    rates = 100 * (sb[:, 1:] + K_BOX * prior_box / 100) / (sb[:, [0]] + K_BOX)
    st = pd.DataFrame(rates, columns=RATE_COLS[:len(BOX)])
    st["r_PM"] = 100 * (sp[:, 1] + K_PM * prior_pm / 100) / (sp[:, 0] + K_PM)
    on = 100 * sp[:, 1] / np.maximum(sp[:, 0], 1e-6)
    off = 100 * sp[:, 2] / np.maximum(sp[:, 3], 1e-6)
    neff = 1.0 / (1.0 / np.maximum(sp[:, 0], 1e-6) + 1.0 / np.maximum(sp[:, 3], 1e-6))
    st["r_ONOFF"] = np.where((sp[:, 0] > 1) & (sp[:, 3] > 1), (on - off) * neff / (neff + K_OO), 0.0)
    st["PLAYER_ID"], st["date"] = q.PLAYER_ID, q.date
    return st


class KalmanAPM:
    """player_impact KalmanAPM: dynamic game-level adjusted plus-minus. State = home court + one
    rating per player, full covariance. R[d] is the state before the games of date d."""

    def __init__(self, p: pd.DataFrame, T: pd.DataFrame, cfg: dict):
        q = p[p.MIN > 0].merge(T[["TEAM", "GAME_ID", "is_home"]], on=["TEAM", "GAME_ID"])
        q["s"] = 5 * q.MIN / q.groupby(["GAME_ID", "TEAM"]).MIN.transform("sum")
        q["a"] = np.where(q.is_home, q.s, -q.s)
        pids = np.sort(q.PLAYER_ID.unique())
        self.k = pd.Series(np.arange(1, len(pids) + 1), index=pids)
        q["k"] = self.k[q.PLAYER_ID].to_numpy()
        h = T[T.is_home].copy()
        h["y"] = 100 * h.margin / h.poss
        h = h[h.GAME_ID.isin(set(q.GAME_ID)) & h.y.notna()].sort_values(["date", "GAME_ID"])
        burnin_end = (p.date.min() + pd.Timedelta(days=cfg["burnin_days"])).to_datetime64() if len(p) else None
        n = len(pids) + 1
        r = np.full(n, cfg["prior_mean"])
        r[0] = cfg["hca0"]
        P = np.diag(np.full(n, cfg["p0"]))
        P[0, 0] = cfg["hca_p0"]
        seen = np.zeros(n, bool)
        seen[0] = True
        self.dates = np.sort(h.date.unique())
        R = np.empty((len(self.dates) + 1, n), dtype=np.float32)
        ref = cfg["prior_mean"]
        REF = np.empty(len(self.dates) + 1)
        qg = dict(tuple(q.groupby("GAME_ID")[["k", "a"]]))
        prev, prev_season = None, None
        for di, (d, hd) in enumerate(h.groupby("date", sort=True)):
            season = hd.season.iloc[0]
            if prev is not None:
                idx = np.flatnonzero(seen)
                add = cfg["q_season"] if season != prev_season else cfg["q_day"] * (d - prev).days
                P[idx, idx] += add
            off = cfg["entry_offset"] if d > burnin_end else 0.0
            r[~seen] = ref + off
            R[di], REF[di] = r, ref + off
            gids = hd.GAME_ID.to_numpy()
            J = np.unique(np.concatenate([[0]] + [qg[gid].k.to_numpy() for gid in gids]))
            pos = pd.Series(np.arange(len(J)), index=J)
            A = np.zeros((len(gids), len(J)))
            A[:, 0] = 1.0
            for m, gid in enumerate(gids):
                A[m, pos[qg[gid].k].to_numpy()] = qg[gid].a.to_numpy()
            PAt = P[:, J] @ A.T
            S = A @ PAt[J] + cfg["sigma2"] * np.eye(len(gids))
            Kt = np.linalg.solve(S, PAt.T)
            pre = A @ r[J]
            est = seen[J] & (J != 0)
            if est.any():
                w = np.abs(A).sum(0)[est]
                ref = (1 - cfg["ref_rho"]) * ref + cfg["ref_rho"] * float(w @ r[J][est] / w.sum())
            r = r + Kt.T @ (hd.y.to_numpy() - pre)
            P -= PAt @ Kt
            seen[J] = True
            prev, prev_season = d, season
        r[~seen] = ref + cfg["entry_offset"]
        R[-1], REF[-1] = r, ref + cfg["entry_offset"]
        self.R, self.REF = R, REF

    def lookup(self, pid, dates) -> np.ndarray:
        di = np.searchsorted(self.dates, np.asarray(dates, dtype="datetime64[ns]"), side="left")
        k = self.k.reindex(pid).to_numpy()
        out = self.REF[di].copy()
        ok = ~np.isnan(k)
        out[ok] = self.R[di[ok], k[ok].astype(int)]
        return out


def pi_candidates(p: pd.DataFrame, T: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """player_impact candidate_rows for the target team-games: every player seen for the team in
    its previous WIN played games (across seasons), projected minutes (mean of his last N_MIN
    appearances in that window) and team games missed in a row."""
    out = []
    pp = p[["PLAYER_ID", "TEAM", "GAME_ID", "MIN"]]
    tg_all = T.sort_values(["date", "GAME_ID"])
    for team, tt in targets.groupby("TEAM"):
        tg = tg_all[tg_all.TEAM == team]
        if not len(tg):
            continue
        gids, dates = tg.GAME_ID.to_numpy(), tg.date.to_numpy()
        pt = pp[pp.TEAM == team]
        players = pt.PLAYER_ID.unique()
        pidx = pd.Series(np.arange(len(players)), index=players)
        gidx = pd.Series(np.arange(len(gids)), index=gids)
        pt = pt[pt.GAME_ID.isin(gidx.index)]
        M = np.full((len(gids), len(players)), np.nan)
        M[gidx[pt.GAME_ID].to_numpy(), pidx[pt.PLAYER_ID].to_numpy()] = pt.MIN.to_numpy()
        Pm = ~np.isnan(M)
        for gid, d in zip(tt.GAME_ID, tt.date):
            j = int(np.searchsorted(dates, np.datetime64(d), side="left"))
            if j == 0:
                continue
            lo = max(0, j - WIN)
            W = Pm[lo:j]
            cand = np.flatnonzero(W.any(0))
            Wc, Mc = W[:, cand][::-1], np.nan_to_num(M[lo:j, cand][::-1])
            streak = np.where(Wc.any(0), Wc.argmax(0), Wc.shape[0])
            take = Wc & (np.cumsum(Wc, 0) <= N_MIN)
            m_hat = (Mc * take).sum(0) / take.sum(0)
            out.append(pd.DataFrame({"TEAM": team, "GAME_ID": gid, "date": d, "PLAYER_ID": players[cand],
                                     "m_hat": m_hat, "streak": streak}))
    if not out:
        return pd.DataFrame(columns=["TEAM", "GAME_ID", "date", "PLAYER_ID", "m_hat", "streak"])
    return pd.concat(out, ignore_index=True)


def _play_key(c: pd.DataFrame) -> np.ndarray:
    s, m = c.streak.to_numpy(), c.m_hat.to_numpy()
    sb = np.select([s == 0, s == 1, s == 2, s <= 4, s <= 9], [0, 1, 2, 3, 4], 5)
    return sb * 10 + np.select([m < 12, m < 24], [0, 1], 2)


def pi_diffs(p_all: pd.DataFrame, p: pd.DataFrame, T: pd.DataFrame, g: pd.DataFrame, rows: np.ndarray,
             pp: dict) -> pd.DataFrame:
    """Home-minus-away expected-lineup composites (d_r_*) for the game-table rows `rows`."""
    st = pi_player_states(p, T, pp["prior"])
    kal = KalmanAPM(p, T, pp["kalman"])
    cand = pi_candidates(p, T, _targets(g, rows))
    if not len(cand):
        return pd.DataFrame(columns=["GAME_ID"] + [f"d_{c}" for c in XCOLS])
    cand["date"] = pd.to_datetime(cand.date)
    cand = cand.sort_values("date").reset_index(drop=True)
    cand = pd.merge_asof(cand, st.sort_values("date"), on="date", by="PLAYER_ID", allow_exact_matches=False,
                         direction="backward")
    for c in RATE_COLS:
        cand[c] = cand[c].fillna(pp["prior"][c])
    cand["r_KAL"] = kal.lookup(cand.PLAYER_ID.to_numpy(), cand.date.to_numpy())
    # traded / released since: his last appearance strictly before the game (preseason
    # included: it shows summer moves) must be for this team
    lt = p_all[["PLAYER_ID", "date", "TEAM"]].rename(columns={"TEAM": "last_team", "date": "lt_date"})
    cand = pd.merge_asof(cand, lt.sort_values("lt_date"), left_on="date", right_on="lt_date", by="PLAYER_ID",
                         allow_exact_matches=False, direction="backward")
    cand = cand[cand.last_team == cand.TEAM].copy()
    tab = {int(k): v for k, v in pp["p_play"].items()}
    cand["e_min"] = pd.Series(_play_key(cand)).map(tab).fillna(0.5).to_numpy() * cand.m_hat.to_numpy()
    rows_ = cand[cand.e_min > 0]
    s = 5 * rows_.e_min / rows_.groupby(["GAME_ID", "TEAM"]).e_min.transform("sum")
    X = rows_[XCOLS].mul(s, axis=0)
    X["GAME_ID"], X["TEAM"] = rows_.GAME_ID.to_numpy(), rows_.TEAM.to_numpy()
    X = X.groupby(["GAME_ID", "TEAM"], as_index=False)[XCOLS].sum()
    sub = g.loc[rows, ["GAME_ID", "home", "away"]]
    h = sub.merge(X, left_on=["GAME_ID", "home"], right_on=["GAME_ID", "TEAM"])
    a = sub.merge(X, left_on=["GAME_ID", "away"], right_on=["GAME_ID", "TEAM"])
    d = h[["GAME_ID"] + XCOLS].merge(a[["GAME_ID"] + XCOLS], on="GAME_ID", suffixes=("_h", "_a"))
    out = pd.DataFrame({"GAME_ID": d.GAME_ID})
    for c in XCOLS:
        out[f"d_{c}"] = d[f"{c}_h"] - d[f"{c}_a"]
    return out


def _pi_coefs(pp: dict, dates: pd.Series) -> List[dict]:
    """The gap / logit coefficients in force at each date (one frozen set, or monthly refits)."""
    if "refits" not in pp:
        return [pp] * len(dates)
    out = []
    for d in dates:
        ds = pd.Timestamp(d).strftime("%Y-%m-%d")
        c = [r for r in pp["refits"] if r["from"] <= ds]
        out.append(c[-1] if c else pp["refits"][0])
    return out


def player_impact_probs(x: pd.DataFrame, pp: dict) -> np.ndarray:
    """Gap = ridge weights . composites; P = logistic(gap, schedule and team differences)."""
    out = np.empty(len(x))
    for i, (row, c) in enumerate(zip(x.itertuples(index=False), _pi_coefs(pp, x.date))):
        r = row._asdict()
        gap = sum(c["gap_beta"][k] * r[f"d_{k}"] for k in XCOLS)
        z = c["logit"]["intercept"] + c["logit"]["gap"] * gap + sum(
            c["logit"][k] * r[k] for k in LOGIT_COLS[1:])
        out[i] = 1.0 / (1.0 + math.exp(-z))
    return out


# ============================================================================ ensemble
def _player_ok(g: pd.DataFrame, rows: np.ndarray, p: pd.DataFrame, n_games: int,
               min_games: int = 0) -> np.ndarray:
    """Per row: both teams have player lines for each of their last `n_games` played games
    before the date (else the player terms would be built on missing or stale data), and for at
    least `min_games` games in all before the date (a store holding only the last few weeks, e.g.
    the daily run without the backfill, gives player ratings that are mostly prior: on 2025-26
    such a 'full' mode was worse than the team-only parts)."""
    have = set(zip(p.date, p.TEAM))
    depth = {t: np.sort(x.date.to_numpy()) for t, x in p[["date", "TEAM"]].drop_duplicates().groupby("TEAM")}
    tl = pd.concat([pd.DataFrame({"team": g.home, "date": g.date, "played": g.played}),
                    pd.DataFrame({"team": g.away, "date": g.date, "played": g.played})])
    tl = tl[tl.played]
    by_team = {t: np.sort(x.date.to_numpy()) for t, x in tl.groupby("team")}
    ok = np.ones(len(rows), bool)
    for i, (d, h, a) in enumerate(zip(g.date.to_numpy()[rows], g.home.to_numpy()[rows], g.away.to_numpy()[rows])):
        for t in (h, a):
            ds = by_team.get(t, np.array([], dtype="datetime64[ns]"))
            prev = ds[:np.searchsorted(ds, d, side="left")][-n_games:]
            if len(prev) < n_games or not all((pd.Timestamp(x), t) in have for x in prev):
                ok[i] = False
            elif min_games and np.searchsorted(depth.get(t, np.array([], dtype="datetime64[ns]")), d,
                                               side="left") < min_games:
                ok[i] = False
    return ok


def compute(hist: pd.DataFrame, players: Optional[pd.DataFrame], params: dict,
            upcoming: Optional[pd.DataFrame] = None, rows: Optional[Iterable[bool]] = None) -> pd.DataFrame:
    """Pre-game probabilities of the three families and the ensemble.

    hist: games_history rows; players: player_games rows (may be empty); upcoming: games to
    predict (game_id, game_date, season, season_type, home, away), added without a result.
    rows: games to return, as a boolean mask over the game table or a function of it (e.g.
    lambda g: g.season == '2024-25'); default: the upcoming games, or every game. Returns one row per returned game with p_elo_plus,
    p_elo_team, p_margin_avail, p_margin_team, p_player, own_home_prob and own_model_mode."""
    g = build_games(hist, upcoming)
    if callable(rows):
        rows = rows(g)
    if rows is None:
        if upcoming is not None and len(upcoming):
            keys = set(zip(upcoming.game_date.astype(str), upcoming.home, upcoming.away))
            rows = np.array([k in keys for k in zip(g.date.dt.strftime("%Y-%m-%d"), g.home, g.away)])
        else:
            rows = np.ones(len(g), bool)
    rows = np.flatnonzero(np.asarray(rows, bool)) if np.asarray(rows).dtype == bool else np.asarray(rows)
    p_all = build_player_logs(players, g)
    p = p_all[p_all.SEASON_TYPE != PRESEASON].reset_index(drop=True)
    has_players = len(p) > 0
    T = build_team_games(p) if has_players else None

    # elo_plus (+ the team-only variant mov_rest)
    ep = params["elo_plus"]
    av = None
    if has_players:
        av = elo_team_game_availability(elo_availability(p, _targets(g, np.arange(len(g))), ep["repl"]),
                                        ep["p_out"])
    A = _elo_arrays(g, av)
    p_elo_plus = elo_pass(ep["par"], A, "post")[rows]
    p_elo_team = elo_pass(params["elo_team"]["par"], A, None)[rows]

    # margin ratings (team-only and with the previous-game absences)
    mp = params["margin"]
    miss = h1_absences(p, T, _targets(g, rows), mp["repl"]) if has_players else None
    players_from = p.SEASON.min() if has_players else None
    pm_team, pm_avail = margin_probs(g, miss, mp, players_from)
    pm_team, pm_avail = pm_team[rows], pm_avail[rows]

    out = g.loc[rows, ["GAME_ID", "date", "season", "home", "away"] + LOGIT_COLS[1:]].reset_index(drop=True)
    out["game_date"] = out.date.dt.strftime("%Y-%m-%d")
    out["p_elo_plus"], out["p_elo_team"] = p_elo_plus, p_elo_team
    out["p_margin_avail"], out["p_margin_team"] = pm_avail, pm_team
    out["p_player"] = np.nan
    if has_players:
        D = pi_diffs(p_all, p, T, g, rows, params["player_impact"])
        x = out.drop(columns="p_player").merge(D, on="GAME_ID", how="inner")
        if len(x):
            pi = dict(zip(x.GAME_ID, player_impact_probs(x, params["player_impact"])))
            out["p_player"] = out.GAME_ID.map(pi).astype(float)
        mode = params.get("mode", {})
        ok = _player_ok(g, rows, p, mode.get("stale_games", 3), mode.get("min_player_games", MIN_PLAYER_GAMES))
    else:
        ok = np.zeros(len(rows), bool)
    full = ok & out.p_player.notna().to_numpy()
    z_full = np.mean([_logit(out.p_elo_plus), _logit(out.p_margin_avail), _logit(out.p_player.fillna(0.5))], 0)
    z_team = np.mean([_logit(out.p_elo_team), _logit(out.p_margin_team)], 0)
    out["own_home_prob"] = _sigm(np.where(full, z_full, z_team))
    out["own_model_mode"] = np.where(full, FULL, TEAM_ONLY)
    return out


def predict(games: List[Dict], hist: pd.DataFrame, players: Optional[pd.DataFrame] = None,
            params: Optional[dict] = None) -> Dict[tuple, Dict]:
    """Own probabilities for ESPN scoreboard games (dicts with game_date, season, season_type,
    home, away, event_id). Unplayed games of earlier dates passed along serve as schedule context
    (rest, back-to-backs). Returns {(game_date, home, away): {own_home_prob, own_away_prob,
    own_model_mode, own_components}}."""
    games = [g for g in games if g["home"] in TEAMS and g["away"] in TEAMS]   # no All-Star games
    if not games:
        return {}
    params = params or load_params()
    if players is None:
        from . import player_store
        players = player_store.load()
    up = pd.DataFrame([{"game_id": f"espn_{g['event_id']}", "game_date": g["game_date"], "season": g["season"],
                        "season_type": g["season_type"], "home": g["home"], "away": g["away"]} for g in games])
    up = up.drop_duplicates(["game_date", "home", "away"])
    res = compute(hist, players, params, upcoming=up)
    out = {}
    for r in res.itertuples(index=False):
        ph = float(r.own_home_prob)
        out[(r.game_date, r.home, r.away)] = {
            "own_home_prob": round(ph, 4), "own_away_prob": round(1 - ph, 4), "own_model_mode": r.own_model_mode,
            "own_components": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in (
                ("elo_plus", r.p_elo_plus), ("elo_team", r.p_elo_team), ("margin_avail", r.p_margin_avail),
                ("margin_team", r.p_margin_team), ("player_impact", r.p_player))}}
    return out
