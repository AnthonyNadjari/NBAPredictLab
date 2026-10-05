"""H7 / elo_plus: our own NBA win probabilities from a better Elo, no bookmaker input.

    python research/h7_own_model/elo_plus/run.py            # uses the availability cache
    python research/h7_own_model/elo_plus/run.py --rebuild  # rebuild it from the player logs

Model family: margin-of-victory Elo (FiveThirtyEight multiplier with autocorrelation
correction) + tuned K / home advantage / season-start regression / early-season K +
rest and back-to-back adjustments + a player-availability adjustment built from the
player box scores (regulars who missed the previous game(s), valued by GameScore,
minutes and on/off) + a roster-talent term (summed value of the current regulars).
Every parameter is tuned walk-forward: for test season S only games of seasons < S are
used (in-season the ratings themselves update game by game, using only games already
played). Odds are read only to compare.

Leak rules enforced by construction:
  * ratings used for a game are the state after all games of earlier dates (a team
    plays at most once a day, and each team's rating only moves with its own games);
  * player availability for a game uses box scores of strictly earlier dates
    (the chronological pass computes a date's features before ingesting that date);
  * the post-game rating update may use who actually played in that (finished) game;
    this only affects later dates;
  * no market column is ever an input.
"""
import argparse
import glob
import json
import math
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "research" / "data"
PLAYERS = DATA / "h1_player_availability"
CACHE = HERE / "cache"

TEST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
FIRST_TUNE_SEASON = "2019-20"          # 2018-19 is the Elo burn-in season
FEATS = ["elo_diff", "net_diff", "form_diff", "rest_diff", "b2b_diff", "g5_diff"]  # backtest.py
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
BUBBLE_START = "2020-07-30"            # 2019-20 restart in Orlando: neutral site

# player availability
ROT_MIN = 12.0        # expected minutes (last 10 appearances for the team) to count as a regular
ROT_WINDOW = 10       # appearances used for expected minutes
VALUE_WINDOW = 82     # appearances (any team) used for per-minute value
K_GS = 300.0          # minutes of shrinkage of GameScore/min toward replacement
K_OO = 2000.0         # minutes of shrinkage of on/off toward 0
GAP_BUCKETS = [0, 1, 2, 3, 5, 10, 20, 10_000]  # consecutive team games missed before the game

LN10_400 = math.log(10) / 400.0


# =============================================================================== data
def load_games():
    g = pd.read_csv(DATA / "games_features_odds.csv")
    g["date"] = pd.to_datetime(g.date)
    g = g.sort_values(["date", "GAME_ID"]).reset_index(drop=True)
    g["neutral"] = ((g.season == "2019-20") & (g.date >= BUBBLE_START)).astype(int)
    g["playoff"] = (g.season_type != "Regular Season").astype(int)
    g["rest3_h"] = (g.rest_h >= 3).astype(int)
    g["rest3_a"] = (g.rest_a >= 3).astype(int)
    g["dense_h"] = (g.g5_h >= 3).astype(int)    # 3+ games in the previous 5 days
    g["dense_a"] = (g.g5_a >= 3).astype(int)
    return g


def am_to_p(a):
    a = pd.to_numeric(a, errors="coerce")
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def load_market():
    """De-vigged (proportional) ESPN main-book moneyline, close and open, keyed by
    (ET game date, home, away). Same construction as research/upsets.py."""
    o = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(DATA / "espn_*.csv")))], ignore_index=True)
    o["home"] = o.home.map(lambda x: ESPN_TO_NBA.get(x, x))
    o["away"] = o.away.map(lambda x: ESPN_TO_NBA.get(x, x))
    o["gdate"] = (pd.to_datetime(o.date_utc, utc=True).dt.tz_convert("America/New_York")
                  .dt.strftime("%Y-%m-%d"))
    for k in ("close", "open"):
        h, a = am_to_p(o[f"home_ml_{k}"]), am_to_p(o[f"away_ml_{k}"])
        o[f"mkt_{k}"] = h / (h + a)
    o = o.drop_duplicates(["gdate", "home", "away"], keep="last")
    return o[["gdate", "home", "away", "mkt_close", "mkt_open"]]


# =============================================================================== players
def _player_logs():
    files = sorted(PLAYERS.glob("player_logs_*.csv"))
    if not files:
        raise SystemExit("no player logs in research/data/h1_player_availability/")
    p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    p = p.drop_duplicates(["PLAYER_ID", "GAME_ID"])
    p["date"] = pd.to_datetime(p.GAME_DATE)
    p["MIN"] = p.MIN.fillna(0).astype(float)
    p["gmsc"] = (p.PTS + 0.4 * p.FGM - 0.7 * p.FGA - 0.4 * (p.FTA - p.FTM) + 0.7 * p.OREB + 0.3 * p.DREB
                 + p.STL + 0.7 * p.AST + 0.7 * p.BLK - 0.4 * p.PF - p.TOV)
    # team margin and game length (for on/off) from the box score itself
    tm = p.groupby(["GAME_ID", "TEAM_ID"]).agg(team_pts=("PTS", "sum"), team_min=("MIN", "sum")).reset_index()
    opp = tm.rename(columns={"TEAM_ID": "OPP_ID", "team_pts": "opp_pts", "team_min": "opp_min"})
    tm = tm.merge(opp, on="GAME_ID")
    tm = tm[tm.TEAM_ID != tm.OPP_ID]
    tm["team_margin"] = tm.team_pts - tm.opp_pts
    tm["game_min"] = (tm.team_min / 5.0).clip(lower=48.0)
    p = p.merge(tm[["GAME_ID", "TEAM_ID", "team_margin", "game_min"]], on=["GAME_ID", "TEAM_ID"], how="left")
    p["off_pm"] = p.team_margin - p.PLUS_MINUS
    p["off_min"] = (p.game_min - p.MIN).clip(lower=0)
    return p.sort_values(["date", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)


def _replacement_rate(p):
    """GameScore per minute of fringe players (< 15 min/game) in 2020-21 only. 2020-21 is
    before every test season (first test season 2022-23)."""
    q = p[p.SEASON == "2020-21"]
    avg = q.groupby("PLAYER_ID").MIN.mean()
    fringe = q[q.PLAYER_ID.isin(avg[avg < 15].index)]
    return float(fringe.gmsc.sum() / fringe.MIN.sum())


def build_availability():
    """One row per (team-game, regular player) with pre-game values and the gap (consecutive
    team games missed just before the game). One chronological pass over dates: all
    rows of date D are computed from the state after date D-1, then D is ingested."""
    p = _player_logs()
    repl = _replacement_rate(p)
    print(f"  replacement GameScore/min (2020-21 fringe) = {repl:.3f}")
    # player state: rolling window over last VALUE_WINDOW appearances (any team)
    win = defaultdict(lambda: deque(maxlen=VALUE_WINDOW))
    sums = defaultdict(lambda: np.zeros(5))            # min, gmsc, pm, off_pm, off_min
    last_team = {}
    team_players = defaultdict(set)                    # (team, season) -> players seen
    team_mins = defaultdict(lambda: deque(maxlen=ROT_WINDOW))  # (team, season, pid) -> minutes
    team_ngames = defaultdict(int)                     # (team, season) -> games played
    last_idx = {}                                      # (team, season, pid) -> team game index
    out = []
    cols = ["MIN", "gmsc", "PLUS_MINUS", "off_pm", "off_min"]
    for d, day in p.groupby("date", sort=True):
        tgs = day[["GAME_ID", "TEAM_ID", "SEASON"]].drop_duplicates()
        played = set(zip(day.GAME_ID, day.TEAM_ID, day.PLAYER_ID))
        # ---- features for every team-game of date d (state = strictly earlier dates)
        for gid, tid, sea in tgs.itertuples(index=False):
            key = (tid, sea)
            j = team_ngames[key]
            for pid in team_players[key]:
                if last_team.get(pid) != tid:
                    continue                           # traded / signed elsewhere since
                mins = team_mins[(tid, sea, pid)]
                exp_min = sum(mins) / len(mins)
                if exp_min < ROT_MIN:
                    continue
                s = sums[pid]
                gs_rate = (s[1] + repl * K_GS) / (s[0] + K_GS)
                on48 = 48 * s[2] / max(s[0], 1.0)
                off48 = 48 * s[3] / max(s[4], 1.0)
                oo = (on48 - off48) * s[0] / (s[0] + K_OO)
                out.append((gid, tid, sea, d, pid, exp_min, max(gs_rate - repl, 0.0) * exp_min,
                            oo * exp_min / 48.0, j - 1 - last_idx[(tid, sea, pid)],
                            int((gid, tid, pid) not in played), j))
        # ---- ingest date d
        for gid, tid, sea in tgs.itertuples(index=False):
            team_ngames[(tid, sea)] += 1
        for r in day[["GAME_ID", "TEAM_ID", "SEASON", "PLAYER_ID"] + cols].itertuples(index=False):
            gid, tid, sea, pid = r[0], r[1], r[2], r[3]
            vals = np.array(r[4:], dtype=float)
            w = win[pid]
            if len(w) == VALUE_WINDOW:
                sums[pid] -= w[0]
            w.append(vals)
            sums[pid] += vals
            last_team[pid] = tid
            team_players[(tid, sea)].add(pid)
            team_mins[(tid, sea, pid)].append(vals[0])
            last_idx[(tid, sea, pid)] = team_ngames[(tid, sea)] - 1
    a = pd.DataFrame(out, columns=["GAME_ID", "TEAM_ID", "season", "date", "PLAYER_ID", "v_min", "v_gs",
                                   "v_oo", "gap", "absent", "team_game_no"])
    return a


def load_availability(rebuild=False):
    f = CACHE / "availability_players.csv"
    if f.exists() and not rebuild:
        return pd.read_csv(f, parse_dates=["date"])
    CACHE.mkdir(exist_ok=True)
    t0 = time.time()
    a = build_availability()
    a.to_csv(f, index=False)
    print(f"  availability rows {len(a):,} built in {time.time() - t0:.0f}s")
    return a


def gap_bucket(gap):
    return np.searchsorted(GAP_BUCKETS, np.asarray(gap), side="right") - 1


def p_out_table(a, seasons):
    """P(absent today | gap bucket), estimated on the given (earlier) seasons only."""
    q = a[a.season.isin(seasons)]
    b = gap_bucket(q.gap)
    tab = pd.Series(q.absent.to_numpy()).groupby(b).mean()
    return np.array([tab.get(k, np.nan) for k in range(len(GAP_BUCKETS) - 1)])


def team_game_availability(a, pout):
    """Per (GAME_ID, TEAM_ID): expected missing value (pre-game, from gaps) and actual
    missing value (who did not play; known only after the game)."""
    w = pout[gap_bucket(a.gap)]
    x = pd.DataFrame({"GAME_ID": a.GAME_ID, "TEAM_ID": a.TEAM_ID})
    for v in ("v_gs", "v_min", "v_oo"):
        x[f"exp_{v[2:]}"] = w * a[v]
        x[f"act_{v[2:]}"] = a.absent * a[v]
        x[f"prev_{v[2:]}"] = (a.gap >= 1) * a[v]
        x[f"tot_{v[2:]}"] = a[v]
    return x.groupby(["GAME_ID", "TEAM_ID"]).sum().reset_index()


# =============================================================================== Elo engine
DEFAULTS = dict(k=20.0, alpha=0.8, ac=0.006, carry=0.75, hca=100.0, hca_po=0.0,
                k_early=0.0, tau_early=10.0, c_b2b=0.0, c_rest3=0.0, c_dense=0.0,
                b_gs=0.0, b_min=0.0, b_oo=0.0, b_tal=0.0, b_tal_oo=0.0, temp=1.0)
# k      : K factor              alpha : MOV exponent in (|MOV|+3)^alpha / (7.5 + ac * winner diff)
# carry  : share of (rating - 1500) kept at a team's first game of a new season
# hca    : home advantage (Elo pts), hca_po: extra home advantage in play-in/playoffs, 0 in the bubble
# k_early: K multiplier 1 + k_early * exp(-games played / 10) (faster learning early in a season)
# c_*    : rest terms (Elo pts): back-to-back penalty, 3+ days rest bonus, 3+ games in 5 days penalty
# b_gs / b_min / b_oo: Elo pts per unit of missing player value (GameScore above replacement x min,
#          minutes, on/off x min), b_tal / b_tal_oo: Elo pts per unit of roster talent difference
# temp   : scale on the published logit only (tested, not used: tuned to ~1)

# variants: free parameters (name, lower, upper) and how availability enters
#   avail = None   : no availability term
#   avail = "exp"  : pre-game expected absences (from gaps) in the prediction AND the update
#   avail = "post" : expected absences in the prediction; the post-game update uses who
#                    actually played in that finished game (ratings = full-strength teams)
#   avail = "oracle": who actually played, also in the prediction (NOT leak-free, upper bound)
_ELO = [("k", 5, 60), ("alpha", 0.3, 1.5), ("carry", 0.3, 1.0), ("hca", 0, 160), ("k_early", 0, 3)]
_REST = [("c_b2b", -30, 80), ("c_rest3", -30, 60), ("c_dense", -30, 60), ("hca_po", -60, 100)]
_AV = [("b_gs", -5, 20), ("b_min", -2, 5), ("b_oo", -10, 20)]
_TAL = [("b_tal", -5, 10), ("b_tal_oo", -10, 20)]
VARIANTS = {
    "elo_538": dict(free=[], avail=None),                     # untuned FiveThirtyEight NBA Elo
    "mov_tuned": dict(free=_ELO, avail=None),
    "mov_rest": dict(free=_ELO + _REST, avail=None),
    "avail_prev": dict(free=_ELO + _REST + _AV, avail="exp"),
    "avail_post": dict(free=_ELO + _REST + _AV, avail="post"),
    "elo_plus": dict(free=_ELO + _REST + _AV + _TAL, avail="post"),
    "oracle_bound": dict(free=_ELO + _REST + _AV + _TAL, avail="oracle"),
}
PRIMARY = "elo_plus"     # the full model; reported as the best leak-free variant


def adjustments(par, A, kind):
    """Home-minus-away Elo adjustment (rest + availability) as a numpy vector."""
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


def elo_pass(par, A, avail, tune=None, keep=False):
    """One chronological pass. Returns (mean log-loss on `tune` rows, pre-game probs)."""
    pre_kind = {None: None, "exp": "exp", "post": "exp", "oracle": "act"}[avail]
    upd_kind = {None: None, "exp": "exp", "post": "act", "oracle": "act"}[avail]
    xp = adjustments(par, A, pre_kind).tolist()
    xu = adjustments(par, A, upd_kind).tolist() if upd_kind != pre_kind else xp
    H, W, S, Y, M, NEU, PO = A["H"], A["W"], A["S"], A["Y"], A["M"], A["NEU"], A["PO"]
    k, alpha, ac, carry = par["k"], par["alpha"], par["ac"], par["carry"]
    hca, hca_po = par["hca"], par["hca_po"]
    k_early, tau, temp = par["k_early"], par["tau_early"], par["temp"]
    early = [math.exp(-n / tau) for n in range(120)]
    R = [1500.0] * 30
    ls = [-1] * 30
    gp = [0] * 30
    exp_, log_ = math.exp, math.log
    ll, nt = 0.0, 0
    tl = tune if tune is not None else [False] * len(H)
    out = [0.0] * len(H) if keep else None
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
        zp = (base + xp[i]) * LN10_400
        p = 1.0 / (1.0 + exp_(-temp * zp))          # published probability
        y = Y[i]
        if tl[i]:                                    # tuning weight of this game (0 = not used)
            ll -= tl[i] * (log_(p) if y else log_(1.0 - p))
            nt += tl[i]
        if keep:
            out[i] = p
        du = base + xu[i]
        pu = p if (xu is xp and temp == 1.0) else 1.0 / (1.0 + exp_(-du * LN10_400))  # Elo expectation
        dw = du if y else -du
        den = 7.5 + ac * dw
        if den < 1.0:
            den = 1.0
        kf = k * (1.0 + k_early * 0.5 * (early[min(gp[h], 119)] + early[min(gp[a], 119)]))
        delta = kf * (M[i] + 3.0) ** alpha / den * (y - pu)
        R[h] += delta
        R[a] -= delta
        gp[h] += 1
        gp[a] += 1
    return (ll / nt if nt else float("nan")), out


def model_arrays(g, av_tg):
    """Python lists / numpy vectors the engine needs, aligned with g."""
    tid = {t: i for i, t in enumerate(sorted(set(g.home_id) | set(g.away_id)))}
    sid = {s: i for i, s in enumerate(sorted(g.season.unique()))}
    A = {"H": g.home_id.map(tid).tolist(), "W": g.away_id.map(tid).tolist(),
         "S": g.season.map(sid).tolist(), "Y": g.home_win.astype(int).tolist(),
         "M": g.margin.abs().astype(float).tolist(), "NEU": g.neutral.tolist(), "PO": g.playoff.tolist()}
    for c in ("b2b", "rest3", "dense"):
        A[f"{c}_h"], A[f"{c}_a"] = g[f"{c}_h"].to_numpy(float), g[f"{c}_a"].to_numpy(float)
    if av_tg is not None:
        for side, col in (("h", "home_id"), ("a", "away_id")):
            m = g[["GAME_ID", col]].merge(av_tg, left_on=["GAME_ID", col], right_on=["GAME_ID", "TEAM_ID"],
                                          how="left")
            for kind in ("exp", "act", "prev", "tot"):
                for v in ("gs", "min", "oo"):
                    A[f"{kind}_{v}_{side}"] = m[f"{kind}_{v}"].fillna(0.0).to_numpy(float)
        # roster talent term only when both teams already have regulars this season
        both = (A["tot_min_h"] > 0) & (A["tot_min_a"] > 0)
        A["tal_gs"] = np.where(both, A["tot_gs_h"] - A["tot_gs_a"], 0.0)
        A["tal_oo"] = np.where(both, A["tot_oo_h"] - A["tot_oo_a"], 0.0)
    return A


def head(A, n):
    """The first n games of the arrays (the tuning pass never needs later games)."""
    return {k: v[:n] for k, v in A.items()}


# =============================================================================== tuning
def tune(variant, A, tune_mask, x0=None, maxiter=80):
    """Minimise mean log-loss of the pre-game probabilities on `tune_mask` rows over the
    variant's free parameters (box-bounded, optimised on a [0, 1] scale)."""
    spec = VARIANTS[variant]
    free = spec["free"]
    par = dict(DEFAULTS)
    if not free:
        return par, None, 0
    lo = np.array([f[1] for f in free], float)
    hi = np.array([f[2] for f in free], float)
    start = np.array([(x0 or DEFAULTS)[f[0]] for f in free], float)
    z0 = np.clip((start - lo) / (hi - lo), 0, 1)
    n = int(np.nonzero(tune_mask)[0].max()) + 1
    Ah = head(A, n)
    tl = [float(w) for w in tune_mask[:n]]
    calls = [0]

    def f(z):
        calls[0] += 1
        q = dict(par)
        q.update({name: float(lo[j] + z[j] * (hi[j] - lo[j])) for j, (name, _, _) in enumerate(free)})
        return elo_pass(q, Ah, spec["avail"], tl)[0]

    r = minimize(f, z0, method="L-BFGS-B", bounds=[(0, 1)] * len(free),
                 options={"maxiter": maxiter, "eps": 2e-4, "ftol": 1e-9})
    par.update({name: float(lo[j] + r.x[j] * (hi[j] - lo[j])) for j, (name, _, _) in enumerate(free)})
    return par, float(r.fun), calls[0]


# =============================================================================== baselines
def fallback_logit(g):
    """research/backtest.py production fallback: logit on FEATS, C=1, trained on seasons
    2019-20..S-1, predicting S (same code as backtest.run)."""
    p = pd.Series(np.nan, index=g.index)
    for s in TEST_SEASONS:
        tr = g[(g.season < s) & (g.season >= "2019-20")]
        te = g.season == s
        m = LogisticRegression(C=1.0, max_iter=2000).fit(tr[FEATS], tr.home_win)
        p[te] = m.predict_proba(g.loc[te, FEATS])[:, 1]
    return p


# =============================================================================== evaluation
def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": int(len(y)), "logloss": float(ll_vec(y, p).mean()), "brier": float(((p - y) ** 2).mean()),
            "acc": float(((p > 0.5) == (y == 1)).mean())}


def paired_ci(y, p_model, p_base, seasons, n_boot=2000, seed=0):
    """Paired bootstrap (over games) of mean log-loss(model) - log-loss(base)."""
    d = ll_vec(y, p_model) - ll_vec(y, p_base)
    rng = np.random.default_rng(seed)
    n = len(d)
    boots = np.concatenate([d[rng.integers(0, n, (250, n))].mean(1) for _ in range(n_boot // 250)])
    per = pd.Series(d).groupby(np.asarray(seasons)).mean()
    return {"n": int(n), "delta": float(d.mean()), "ci95": [float(np.quantile(boots, .025)),
                                                          float(np.quantile(boots, .975))],
            "per_season": {s: float(v) for s, v in per.items()},
            "seasons_better": int((per < 0).sum())}


def date_block_ci(y, p_model, p_base, dates, n_boot=2000, seed=1):
    """Same, resampling whole game-days (same-day games share news / league-wide noise)."""
    d = pd.Series(ll_vec(y, p_model) - ll_vec(y, p_base))
    day = pd.Series(np.asarray(dates)).astype(str)
    s = d.groupby(day).sum()
    c = d.groupby(day).size()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    boots = s.to_numpy()[idx].sum(1) / c.to_numpy()[idx].sum(1)
    return [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]


# =============================================================================== main
def walk_forward(g, a, variants, log):
    """Tune every variant on seasons < S, then predict season S. Returns (preds, params)."""
    preds = {v: pd.Series(np.nan, index=g.index) for v in variants}
    params = {v: {} for v in variants}
    warm = {}
    for s in TEST_SEASONS:
        prior_player_seasons = sorted(x for x in a.season.unique() if x < s)
        pout = p_out_table(a, prior_player_seasons)
        log(f"\n[{s}] P(absent | consecutive games missed {GAP_BUCKETS[:-1]}) from "
            f"{prior_player_seasons}: {np.round(pout, 3).tolist()}")
        A = model_arrays(g, team_game_availability(a, pout))
        mask = ((g.season >= FIRST_TUNE_SEASON) & (g.season < s)).to_numpy()
        te = (g.season == s).to_numpy()
        seasons_back = pd.Series(sorted(g.season.unique())).pipe(lambda x: dict(zip(x, range(len(x)))))
        age = (seasons_back[s] - 1 - g.season.map(seasons_back)).to_numpy()   # 0 = season S-1
        prev = None
        for v in variants:
            t0 = time.time()
            x0 = warm.get(v, prev)
            w = mask * VARIANTS[v].get("decay", 1.0) ** age
            par, f, calls = tune(v, A, w, x0=x0)
            _, p = elo_pass(par, A, VARIANTS[v]["avail"], keep=True)
            preds[v][te] = np.asarray(p)[te]
            params[v][s] = {k: round(x, 4) for k, x in par.items()}
            warm[v] = par
            if v != "oracle_bound":
                prev = par
            log(f"  {v:13s} tune-ll {f if f is not None else float('nan'):.4f} ({calls} passes, "
                f"{time.time() - t0:.0f}s)  " + " ".join(f"{n}={par[n]:.3g}" for n, _, _ in VARIANTS[v]["free"]))
    return preds, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the availability cache")
    args = ap.parse_args()
    t_start = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    g = load_games()
    mk = load_market()
    g["gdate"] = g.date.dt.strftime("%Y-%m-%d")
    g = g.merge(mk, on=["gdate", "home", "away"], how="left")
    assert g.GAME_ID.is_unique
    a = load_availability(args.rebuild)
    log(f"games {len(g):,} ({g.season.min()}..{g.season.max()}), availability rows {len(a):,}")

    variants = list(VARIANTS)
    preds, params = walk_forward(g, a, variants, log)
    P = {"market_close": g.mkt_close, "market_open": g.mkt_open,
         "fallback_logit": fallback_logit(g), "elo_prod": g.elo_prob}
    P.update(preds)

    # ---------------------------------------------------------------- evaluation set
    ev = g.season.isin(TEST_SEASONS) & g.mkt_close.notna()
    for k, v in P.items():
        if k != "market_open":
            assert v[ev].notna().all(), k
    E = g[ev].reset_index(drop=True)
    PE = {k: v[ev].reset_index(drop=True) for k, v in P.items()}
    op = PE["market_open"].notna().to_numpy()
    log(f"\nevaluation set: {len(E):,} test-season games with a closing line "
        f"({E.groupby('season').size().to_dict()}); opening line on {op.sum():,} of them")

    names = ["market_close", "market_open", "fallback_logit", "elo_prod"] + variants
    res = {"eval_set": {"n": int(len(E)), "n_open": int(op.sum()),
                        "per_season": E.groupby("season").size().to_dict(),
                        "definition": "test-season games (2022-23..2025-26, all game types) with an ESPN "
                                      "closing moneyline; the open subset also has an opening line"},
           "predictors": {}, "vs_fallback": {}, "vs_market_close": {}, "vs_market_open": {}, "params": params}
    for k in names:
        m = op if k == "market_open" else np.ones(len(E), bool)
        p, y, se = PE[k].to_numpy()[m], E.home_win.to_numpy()[m], E.season.to_numpy()[m]
        r = {"pooled": metrics(y, p), "per_season": {s: metrics(y[se == s], p[se == s]) for s in sorted(set(se))}}
        if k != "market_open":
            yo, po, so = E.home_win.to_numpy()[op], PE[k].to_numpy()[op], E.season.to_numpy()[op]
            r["open_subset"] = {"pooled": metrics(yo, po),
                                "per_season": {s: metrics(yo[so == s], po[so == s]) for s in sorted(set(so))}}
        res["predictors"][k] = r
    y = E.home_win.to_numpy()
    for k in ["elo_prod"] + variants:
        res["vs_fallback"][k] = paired_ci(y, PE[k], PE["fallback_logit"], E.season)
        res["vs_fallback"][k]["ci95_dayblock"] = date_block_ci(y, PE[k], PE["fallback_logit"], E.gdate)
    for k in ["fallback_logit", "elo_prod"] + variants:
        res["vs_market_close"][k] = paired_ci(y, PE[k], PE["market_close"], E.season)
        res["vs_market_close"][k]["ci95_dayblock"] = date_block_ci(y, PE[k], PE["market_close"], E.gdate)
        res["vs_market_open"][k] = paired_ci(y[op], PE[k].to_numpy()[op], PE["market_open"].to_numpy()[op],
                                             E.season.to_numpy()[op])
    # what each step of the ladder adds (each variant vs the previous one)
    ladder = ["elo_538", "mov_tuned", "mov_rest", "avail_prev", "avail_post", "elo_plus"]
    pairs = list(zip(ladder[1:], ladder[:-1])) + [("elo_plus", "avail_prev"), ("elo_plus", "mov_rest")]
    res["ladder"] = {f"{a_} - {b_}": paired_ci(y, PE[a_], PE[b_], E.season) for a_, b_ in pairs}
    # where the gap to the market sits
    gp = np.minimum(E.games_played_h, E.games_played_a)
    segs = {"regular season": E.playoff == 0, "play-in + playoffs": E.playoff == 1,
            "games played < 10": gp < 10, "games played 10-29": (gp >= 10) & (gp < 30),
            "games played 30-59": (gp >= 30) & (gp < 60), "games played 60+": gp >= 60}
    res["segments"] = {}
    for name, m in segs.items():
        m = m.to_numpy()
        res["segments"][name] = {k: metrics(y[m], PE[k].to_numpy()[m])
                                 for k in ("market_close", "fallback_logit", PRIMARY)}
    lp = np.log(np.clip(PE[PRIMARY], 1e-6, 1 - 1e-6) / (1 - np.clip(PE[PRIMARY], 1e-6, 1 - 1e-6)))
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(lp.to_numpy()[:, None], y)
    res["calibration_" + PRIMARY] = {"slope": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0]),
                                     "note": "diagnostic only, fitted on the test games; 1/0 = calibrated"}

    # ---------------------------------------------------------------- report
    log("\nlog-loss / accuracy (evaluation set; market_open row and 'open' column on the open subset)")
    hdr = f"{'predictor':15s} " + " ".join(f"{s:>13s}" for s in TEST_SEASONS) + f" {'pooled':>15s} {'brier':>7s} {'open-sub':>9s}"
    log(hdr)
    for k in names:
        r = res["predictors"][k]
        cells = " ".join(f"{r['per_season'][s]['logloss']:.4f}/{r['per_season'][s]['acc']:.3f}"
                         if s in r["per_season"] else f"{'-':>13s}" for s in TEST_SEASONS)
        o = r.get("open_subset", {}).get("pooled", r["pooled"])["logloss"]
        log(f"{k:15s} {cells} {r['pooled']['logloss']:.4f}/{r['pooled']['acc']:.3f} {r['pooled']['brier']:.4f} {o:9.4f}")
    log("\npaired bootstrap, mean log-loss difference (negative = better), 95% CI over games [over game-days]")
    for ref in ("vs_fallback", "vs_market_close", "vs_market_open"):
        log(f"  {ref}:")
        for k, c in res[ref].items():
            blk = c.get("ci95_dayblock")
            log(f"    {k:15s} {c['delta'] * 1e3:+7.2f}e-3  [{c['ci95'][0] * 1e3:+6.2f}, {c['ci95'][1] * 1e3:+6.2f}]"
                + (f" [{blk[0] * 1e3:+6.2f}, {blk[1] * 1e3:+6.2f}]" if blk else "")
                + f"  better in {c['seasons_better']}/{len(c['per_season'])} seasons")
    log("  ladder (variant - previous variant):")
    for k, c in res["ladder"].items():
        log(f"    {k:26s} {c['delta'] * 1e3:+7.2f}e-3  [{c['ci95'][0] * 1e3:+6.2f}, {c['ci95'][1] * 1e3:+6.2f}]"
            f"  better in {c['seasons_better']}/4 seasons")
    log("\nsegments (log-loss): market_close / fallback_logit / " + PRIMARY)
    for name, r in res["segments"].items():
        log(f"  {name:20s} n={r[PRIMARY]['n']:5d}  " + " / ".join(f"{r[k]['logloss']:.4f}" for k in r))
    c = res["calibration_" + PRIMARY]
    log(f"\ncalibration of {PRIMARY} on the test games: slope {c['slope']:.3f}, intercept {c['intercept']:+.3f}")

    # ---------------------------------------------------------------- outputs
    best = PRIMARY
    te = g.season.isin(TEST_SEASONS)
    out = pd.DataFrame({"GAME_ID": g.GAME_ID[te], "season": g.season[te], "date": g.gdate[te],
                        "home": g.home[te], "away": g.away[te], "home_win": g.home_win[te],
                        "p": preds[best][te].round(5)})
    out.to_csv(HERE / "preds.csv", index=False)
    res["best_variant"] = best
    res["runtime_s"] = round(time.time() - t_start, 1)
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    log(f"\nwrote preds.csv ({len(out):,} rows, variant {best}), results.json; {time.time() - t_start:.0f}s")
    (HERE / "run_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
