"""H7 / margin_ratings: our own NBA win probabilities from point-margin team ratings.

No bookmaker number is ever an input. Odds are loaded only to score the market as a benchmark.

Model, for a game on date D (only games on dates < D are used):
    margin_hat = r_home - r_away + HCA * (not neutral) + schedule/context terms [+ availability]
    P(home win) = Phi(margin_hat / sigma)
Team ratings r and HCA are a weighted ridge (Massey / SRS) least-squares fit to past score margins:
    * exponential recency weights, half-life `hl` days, within the season,
    * shrinkage toward a prior = `carry` x last season's final rating, worth `k` fresh games,
    * margins capped at +-`cap` points (blow-out / garbage-time damping),
    * 3-point luck removed from past margins with weight `alpha` (3PM above the league 3P% x 3PA),
    * optional possession adjustment (per-100 ratings rescaled by a pace model).
Schedule/context terms (back-to-back, rest, games in 5 days, travel km, altitude, playoff home
court, late-season standings) are fit on earlier seasons' out-of-sample residuals, removed from
the targets of the rating fit, and added back to the prediction.
Everything is walk-forward: for test season S, hyper-parameters, sigma and coefficients are chosen
on seasons [2019-20, S) only (2018-19 is the burn-in season: no prior).

Extension (variant margin_full_avail, the best one): + previous-game absence proxy from player box
scores (research/h1_player_availability: rotation players who sat out the team's last game, valued
by prior Game Score above replacement x expected minutes).

Run:  python research/h7_own_model/margin_ratings/run.py     (about 5 min, no network)
Writes preds.csv (best variant), preds_team_only.csv (margin_full), results.json, grid.csv.
"""
import itertools
import json
import os
import sys
import time

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # tiny 31x31 solves: BLAS threading makes them ~70x slower
from math import asin, cos, radians, sin, sqrt  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "research" / "data"
sys.path.insert(0, str(ROOT / "research" / "h2_travel_schedule"))
from arenas import NEUTRAL, NEUTRAL_GAMES, home_arena  # noqa: E402  (static arena coordinates)

TEST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
VALID_SEASON = "2021-22"       # extra walk-forward season, never looked at while designing the model
TUNE_FROM = "2019-20"          # first season with a prior (2018-19 is burn-in)
FALLBACK_FEATS = ["elo_diff", "net_diff", "form_diff", "rest_diff", "b2b_diff", "g5_diff"]
EPS = 1e-4
N_TEAMS = 30
BUBBLE_START = "2020-07-30"    # 2019-20 restart in Orlando: every game on a neutral floor
ORLANDO_BUBBLE = (28.337, -81.556, "America/New_York", 30)
SEASON_GAMES = {"2020-21": 72}  # scheduled regular-season games (82 otherwise)
LATE_CUTS = [0.6, 0.67, 0.75, 0.82]  # candidate "late season" thresholds (share of schedule played)
BOOT_REPS = 10000

# hyper-parameter grid (walk-forward selection on earlier seasons only)
GRID = {
    "hl": [30.0, 45.0, 65.0, 90.0, 130.0],   # recency half-life, days
    "carry": [0.55, 0.7, 0.85],              # share of last season's rating kept in the prior
    "k": [5.0, 8.0, 12.0],                   # prior strength, in "fresh games"
    "cap": [np.inf, 25.0, 18.0],             # margin cap (points)
    "alpha": [0.0, 0.2, 0.35],               # share of 3-point luck removed from past margins
    "per100": [False, True],                 # possession-adjusted ratings
}
OD_GRID = [(0.15, 0.5), (0.3, 0.6), (0.15, 0.7)]  # (alpha offense, alpha defense) for the O/D split


# ============================================================================ data
def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(radians, (a[0], a[1], b[0], b[1]))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def load_games():
    """All games 2018-19..2025-26. The pre-game features come from research/features.py (shift(1)).
    The game's own box score (margin, possessions, 3PM/3PA) is used only as a TARGET for the
    ratings of LATER dates."""
    g = pd.read_csv(DATA / "games_features_odds.csv")
    g = g.sort_values(["date", "GAME_ID"]).reset_index(drop=True)
    g = g.drop(columns=["mkt", "spread", "book"])  # odds: never reachable by a model
    teams = sorted(set(g.home_id) | set(g.away_id))
    assert len(teams) == N_TEAMS
    tix = {t: i for i, t in enumerate(teams)}
    g["hi"], g["ai"] = g.home_id.map(tix), g.away_id.map(tix)
    g["day"] = (pd.to_datetime(g.date) - pd.Timestamp("2018-01-01")).dt.days.astype(int)
    keys = [(d, frozenset({h, a})) for d, h, a in zip(g.date, g.home, g.away)]
    g["neutral_code"] = [NEUTRAL_GAMES.get(k) for k in keys]
    g["bubble"] = (g.season == "2019-20") & (g.date >= BUBBLE_START)
    g["neutral"] = (g.neutral_code.notna() | g.bubble).astype(int)
    g["poss_game"] = (g.poss_h + g.poss_a) / 2
    t = pd.read_csv(DATA / "team_logs.csv", usecols=["GAME_ID", "TEAM_ID", "FG3M"])
    for side, col in (("h", "home_id"), ("a", "away_id")):
        g = g.merge(t.rename(columns={"TEAM_ID": col, "FG3M": f"FG3M_{side}"}),
                    on=["GAME_ID", col], how="left", validate="1:1")
    assert g[["FG3M_h", "FG3M_a"]].notna().all().all()
    g = add_travel(g)
    g = add_availability(g)
    return g


def add_travel(g):
    """Per side: km from the previous game's venue (same season; first game: from own arena) and a
    flag for a low-altitude visitor in Denver / Utah / Mexico City. Uses only the schedule."""
    venue = []
    for r in g.itertuples():
        if r.bubble:
            venue.append(ORLANDO_BUBBLE)
        elif isinstance(r.neutral_code, str):
            venue.append(NEUTRAL[r.neutral_code])
        else:
            venue.append(home_arena(r.home, r.season))
    g = g.copy()
    last = {}
    trav = {"home": np.zeros(len(g)), "away": np.zeros(len(g))}
    alt = {"home": np.zeros(len(g)), "away": np.zeros(len(g))}
    for i, (r, v) in enumerate(zip(g.itertuples(), venue)):
        for side in ("home", "away"):
            t = getattr(r, side)
            own = home_arena(t, r.season)
            prev = last.get((t, r.season), own)
            trav[side][i] = haversine_km(prev, v) / 1000.0
            alt[side][i] = float(v[3] > 1000 and own[3] < 1000)
            last[(t, r.season)] = v
    g["trav_h"], g["trav_a"] = trav["home"], trav["away"]
    g["alt_h"], g["alt_a"] = alt["home"], alt["away"]
    return g


def add_availability(g):
    """Previous-game absence proxy (H1, leak-free): sum of prior player value (GameScore above
    replacement x expected minutes) of rotation players who did NOT play in the team's previous
    game. Only from 2020-21 (player logs); 0 before. Built by research/h1_player_availability."""
    path = DATA / "h1_player_availability" / "rotation_player_games.csv"
    g = g.copy()
    if not path.exists():
        g["avail_prev_diff"], g["has_players"] = 0.0, False
        return g
    r = pd.read_csv(path, usecols=["team", "GAME_ID", "absent_prev", "v_gs"])
    r["miss"] = r.absent_prev.astype(float) * r.v_gs
    a = r.groupby(["team", "GAME_ID"]).miss.sum().reset_index()
    for side, col in (("h", "home"), ("a", "away")):
        g = g.merge(a.rename(columns={"team": col, "miss": f"miss_{side}"}), on=["GAME_ID", col],
                    how="left", validate="1:1")
    g["has_players"] = g.season >= "2020-21"
    g["avail_prev_diff"] = np.where(g.has_players, g.miss_h.fillna(0) - g.miss_a.fillna(0), 0.0)
    return g


def shooting_luck(g):
    """Points from 3-point shooting above/below the league rate, per side, for each game.
    League rate = running league 3P% over all games on EARLIER dates (0.355 before any)."""
    made, att = g.FG3M_h + g.FG3M_a, g.FG3A_h + g.FG3A_a
    day_m = made.groupby(g.day).sum().cumsum().shift(1)
    day_a = att.groupby(g.day).sum().cumsum().shift(1)
    rate = (day_m / day_a).reindex(g.day).fillna(0.355).to_numpy()
    lh = 3 * (g.FG3M_h.to_numpy() - g.FG3A_h.to_numpy() * rate)
    la = 3 * (g.FG3M_a.to_numpy() - g.FG3A_a.to_numpy() * rate)
    return lh, la


def schedule_terms(g):
    """Schedule terms known before tip-off (home-minus-away where it applies)."""
    rest_h, rest_a = g.rest_h.clip(1, 4), g.rest_a.clip(1, 4)
    po = (g.season_type != "Regular Season").astype(float)
    return pd.DataFrame({
        "b2b_h": g.b2b_h.astype(float), "b2b_a": g.b2b_a.astype(float),
        "rest_diff": (rest_h - rest_a).astype(float),
        "g5_diff": g.g5_diff.astype(float),
        "trav_diff": (g.trav_h - g.trav_a).astype(float),
        "alt_diff": (g.alt_h - g.alt_a).astype(float),
        "po_home": po * (1 - g.neutral),          # extra home edge in the playoffs / play-in
    }, index=g.index)


def late_terms(g, cut):
    """Late regular season (share of the schedule played >= cut): bad teams (win% < .40 so far)
    under-perform their rating (tanking, resting), good teams (> .65) slightly over-perform."""
    n_sched = g.season.map(SEASON_GAMES).fillna(82).to_numpy()
    rs = (g.season_type == "Regular Season").to_numpy(float)
    out = {}
    for side in ("h", "a"):
        late = (g[f"games_played_{side}"].to_numpy() / n_sched >= cut) * rs
        w = g[f"wpct_{side}"].fillna(0.5).to_numpy()
        out[f"late_bad_{side}"] = late * (w < 0.40)
        out[f"late_good_{side}"] = late * (w > 0.65)
    return pd.DataFrame(out, index=g.index)


def load_eval():
    """The canonical 5,197 test games (research/upsets.py load()): de-vigged ESPN close and open."""
    u = pd.read_csv(DATA / "upsets_dataset.csv",
                    usecols=["game_id", "game_date", "season", "home", "away", "home_win", "mkt_close", "mkt_open"])
    return u.rename(columns={"game_id": "GAME_ID"})


# ============================================================================ rating engines
def date_blocks(g):
    """(season, start, end) index blocks of games sharing a date (g sorted by date)."""
    d = g.day.to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(d)) + 1]
    ends = np.r_[starts[1:], len(d)]
    return list(zip(g.season.to_numpy()[starts], starts, ends))


def rate(g, blocks, hl=50.0, carry=0.7, k=8.0, cap=np.inf, per100=False, k_hca=60.0, adj=None):
    """Sequential decayed ridge / Massey net ratings. Returns the pre-game model margin for every
    game (fit on games of EARLIER dates only: a date's games are added after it is predicted)
    and the pre-game HCA estimate.
    adj: per-game points removed from the target before fitting (3P luck, schedule terms)."""
    n = len(g)
    hi, ai = g.hi.to_numpy(), g.ai.to_numpy()
    home_flag = 1.0 - g.neutral.to_numpy()
    day = g.day.to_numpy()
    y = g.margin.to_numpy(float)
    if adj is not None:
        y = y - adj
    y = np.clip(y, -cap, cap)
    poss = g.poss_game.to_numpy(float)
    if per100:
        y = 100.0 * y / poss
    P = N_TEAMS + 1
    r = np.arange(n)
    X = np.zeros((n, P))
    X[r, hi], X[r, ai], X[:, N_TEAMS] = 1.0, -1.0, home_flag
    Xp = np.zeros((n, P))  # pace model (per100 only): poss = m + p_home + p_away
    Xp[r, hi], Xp[r, ai], Xp[:, N_TEAMS] = 1.0, 1.0, 1.0
    mu_out, hca_out, pace_out = np.empty(n), np.empty(n), np.empty(n)
    theta_end = np.zeros(P)
    pace_end = np.r_[np.zeros(N_TEAMS), 100.0]
    lam = np.r_[np.full(N_TEAMS, k), k_hca]
    lam_p = np.r_[np.full(N_TEAMS, k), 1.0]
    D_lam, D_lam_p = np.diag(lam), np.diag(lam_p)
    cur, A = None, None
    for season, s, e in blocks:
        if season != cur:  # new season: drop the data, prior = carry x last season's final fit
            if A is not None:
                theta_end = np.linalg.solve(A + D_lam, b + lam * prior)
                pace_end = np.linalg.solve(Ap + D_lam_p, bp + lam_p * prior_p)
            cur = season
            rr = theta_end[:N_TEAMS] - theta_end[:N_TEAMS].mean()
            prior = np.r_[carry * rr, theta_end[N_TEAMS]]
            pr = pace_end[:N_TEAMS] - pace_end[:N_TEAMS].mean()
            prior_p = np.r_[carry * pr, pace_end[N_TEAMS] + 2 * pace_end[:N_TEAMS].mean()]
            A, b = np.zeros((P, P)), np.zeros(P)
            Ap, bp = np.zeros((P, P)), np.zeros(P)
            last = None
        if last is not None and np.isfinite(hl):
            f = 0.5 ** ((day[s] - last) / hl)
            A *= f
            b *= f
            Ap *= f
            bp *= f
        theta = np.linalg.solve(A + D_lam, b + lam * prior)
        Xb = X[s:e]
        mu_out[s:e] = Xb @ theta
        hca_out[s:e] = theta[N_TEAMS]
        if per100:
            th_p = np.linalg.solve(Ap + D_lam_p, bp + lam_p * prior_p)
            pace_out[s:e] = Xp[s:e] @ th_p
            Ap += Xp[s:e].T @ Xp[s:e]
            bp += Xp[s:e].T @ poss[s:e]
        A += Xb.T @ Xb
        b += Xb.T @ y[s:e]
        last = day[s]
    if per100:
        mu_out = mu_out * pace_out / 100.0
    return mu_out, hca_out


def rate_od(g, blocks, hl=50.0, carry=0.7, k=8.0, cap=np.inf, k_hca=60.0, luck=None, a_off=0.15,
            a_def=0.5, adj=None):
    """Offense / defense split, per 100 possessions. Two observations per game:
        ortg_home = m + o_home + d_away + h/2,   ortg_away = m + o_away + d_home - h/2.
    Offensive ratings come from a fit where each team's own 3-point luck is removed with weight
    a_off, defensive ratings from a fit with weight a_def (3P% allowed is mostly noise).
    adj (points, home-minus-away) is split half/half between the two scores.
    Returns the pre-game expected margin in points (pace model as in rate())."""
    n = len(g)
    hi, ai = g.hi.to_numpy(), g.ai.to_numpy()
    hf = 1.0 - g.neutral.to_numpy()
    day = g.day.to_numpy()
    poss = g.poss_game.to_numpy(float)
    lh, la = luck if luck is not None else (np.zeros(n), np.zeros(n))
    adj = np.zeros(n) if adj is None else adj
    T = N_TEAMS
    P = 2 * T + 2  # o, d, league mean, hca
    r = np.arange(n)
    Xh, Xa = np.zeros((n, P)), np.zeros((n, P))
    Xh[r, hi], Xh[r, T + ai], Xh[:, 2 * T], Xh[:, 2 * T + 1] = 1.0, 1.0, 1.0, 0.5 * hf
    Xa[r, ai], Xa[r, T + hi], Xa[:, 2 * T], Xa[:, 2 * T + 1] = 1.0, 1.0, 1.0, -0.5 * hf
    pts_h, pts_a = g.PTS_h.to_numpy(float), g.PTS_a.to_numpy(float)
    targets = {}
    for name, a in (("off", a_off), ("def", a_def)):
        mh = 100 * (pts_h - a * lh - adj / 2) / poss
        ma = 100 * (pts_a - a * la + adj / 2) / poss
        if np.isfinite(cap):  # cap the margin, keep the game total
            c = 100 * cap / poss
            mid, half = (mh + ma) / 2, np.clip((mh - ma) / 2, -c / 2, c / 2)
            mh, ma = mid + half, mid - half
        targets[name] = (mh, ma)
    Xp = np.zeros((n, T + 1))
    Xp[r, hi], Xp[r, ai], Xp[:, T] = 1.0, 1.0, 1.0
    lam = np.r_[np.full(2 * T, k), 1e-3, k_hca]
    lam_p = np.r_[np.full(T, k), 1e-3]
    D_lam, D_lam_p = np.diag(lam), np.diag(lam_p)
    out = np.empty(n)
    end = {nm: np.r_[np.zeros(2 * T), 110.0, 0.0] for nm in ("off", "def")}
    pace_end = np.r_[np.zeros(T), 100.0]
    cur, st = None, None
    for season, s, e in blocks:
        if season != cur:
            if st is not None:
                for nm in ("off", "def"):
                    A, b, pr = st[nm]
                    end[nm] = np.linalg.solve(A + D_lam, b + lam * pr)
                Ap, bp, prp = st["pace"]
                pace_end = np.linalg.solve(Ap + D_lam_p, bp + lam_p * prp)
            cur = season
            st = {}
            for nm in ("off", "def"):
                th = end[nm]
                o, d = th[:T] - th[:T].mean(), th[T:2 * T] - th[T:2 * T].mean()
                pr = np.r_[carry * o, carry * d, th[2 * T] + th[:T].mean() + th[T:2 * T].mean(), th[2 * T + 1]]
                st[nm] = [np.zeros((P, P)), np.zeros(P), pr]
            pp = pace_end[:T] - pace_end[:T].mean()
            st["pace"] = [np.zeros((T + 1, T + 1)), np.zeros(T + 1),
                          np.r_[carry * pp, pace_end[T] + 2 * pace_end[:T].mean()]]
            last = None
        if last is not None and np.isfinite(hl):
            f = 0.5 ** ((day[s] - last) / hl)
            for v in st.values():
                v[0] *= f
                v[1] *= f
        A, b, pr = st["off"]
        th_o = np.linalg.solve(A + D_lam, b + lam * pr)
        A, b, pr = st["def"]
        th_d = np.linalg.solve(A + D_lam, b + lam * pr)
        Ap, bp, prp = st["pace"]
        th_p = np.linalg.solve(Ap + D_lam_p, bp + lam_p * prp)
        hs, as_ = hi[s:e], ai[s:e]
        net_h = th_o[hs] - th_d[T + hs]
        net_a = th_o[as_] - th_d[T + as_]
        hca = 0.5 * (th_o[2 * T + 1] + th_d[2 * T + 1])
        pace = th_p[hs] + th_p[as_] + th_p[T]
        out[s:e] = (net_h - net_a + hca * hf[s:e]) * pace / 100.0
        for nm in ("off", "def"):
            mh, ma = targets[nm]
            v = st[nm]
            v[0] += Xh[s:e].T @ Xh[s:e] + Xa[s:e].T @ Xa[s:e]
            v[1] += Xh[s:e].T @ mh[s:e] + Xa[s:e].T @ ma[s:e]
        st["pace"][0] += Xp[s:e].T @ Xp[s:e]
        st["pace"][1] += Xp[s:e].T @ poss[s:e]
        last = day[s]
    return out


# ============================================================================ link + metrics
def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_sigma(mu, y):
    """MLE of sigma in P = Phi(mu / sigma) (the probit spread of the margin)."""
    f = lambda s: ll_vec(y, norm.cdf(mu / s)).mean()
    return float(minimize_scalar(f, bounds=(4.0, 30.0), method="bounded").x)


def metrics(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return {"n": int(len(y)), "logloss": round(float(ll_vec(y, p).mean()), 5),
            "brier": round(float(((p - y) ** 2).mean()), 5), "acc": round(float(((p > 0.5) == (y == 1)).mean()), 4)}


def boot_ci(y, p_a, p_b, reps=BOOT_REPS, seed=7):
    """Paired bootstrap over games of mean log-loss(a) - log-loss(b). Negative = a is better."""
    d = ll_vec(y, p_a) - ll_vec(y, p_b)
    rng = np.random.default_rng(seed)
    bs = np.concatenate([d[rng.integers(0, len(d), size=(500, len(d)))].mean(1) for _ in range(reps // 500)])
    return {"diff": round(float(d.mean()), 5), "lo": round(float(np.quantile(bs, 0.025)), 5),
            "hi": round(float(np.quantile(bs, 0.975)), 5), "share_boot_better": round(float((bs < 0).mean()), 4),
            "n": int(len(d))}


# ============================================================================ walk-forward pieces
def season_mask(g, lo, hi_excl):
    return ((g.season >= lo) & (g.season < hi_excl)).to_numpy()


def ols(X, r):
    beta, *_ = np.linalg.lstsq(np.asarray(X, float), r, rcond=None)
    return beta


def choose_late_cut(g, S, resid, tr):
    """Late-season threshold by leave-one-season-out CV of the residual regression (train only)."""
    seasons = sorted(set(g.season[tr]))
    best = None
    for cut in LATE_CUTS:
        X = pd.concat([S, late_terms(g, cut)], axis=1).to_numpy(float)
        sse = 0.0
        for s in seasons:
            va = tr & (g.season == s).to_numpy()
            fi = tr & ~va
            beta = ols(X[fi], resid[fi])
            sse += ((resid[va] - X[va] @ beta) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, cut)
    return best[1]


def run_grid(g, blocks, luck):
    """One leak-free pass per hyper-parameter combination: pre-game margins for all games."""
    lh, la = luck
    keys = list(itertools.product(*GRID.values()))
    preds = {}
    for hl, carry, k, cap, alpha, per100 in keys:
        mu, _ = rate(g, blocks, hl=hl, carry=carry, k=k, cap=cap, per100=per100,
                     adj=alpha * (lh - la) if alpha else None)
        preds[(hl, carry, k, cap, alpha, per100)] = mu
    return preds


def select(preds, y, tr, allow=lambda cfg: True, log=None):
    """Best config by log-loss on the training seasons (sigma fit there too)."""
    best = None
    for cfg, mu in preds.items():
        if not allow(cfg):
            continue
        sig = fit_sigma(mu[tr], y[tr])
        ll = ll_vec(y[tr], norm.cdf(mu[tr] / sig)).mean()
        if log is not None:
            log.append({**cfg_dict(cfg), "sigma": round(sig, 3), "train_logloss": round(float(ll), 5)})
        if best is None or ll < best[0]:
            best = (ll, cfg, sig)
    return best


def cfg_dict(cfg):
    hl, carry, k, cap, alpha, per100 = cfg
    return {"hl_days": None if not np.isfinite(hl) else hl, "carry": carry, "k_games": k,
            "cap": None if not np.isfinite(cap) else cap, "alpha_3p_luck": alpha, "per100": per100}


def fallback_probs(g):
    """research/backtest.py: LogisticRegression(C=1) on FEATS, train seasons [2019-20, S)."""
    p = np.full(len(g), np.nan)
    for s in [VALID_SEASON] + TEST_SEASONS:
        tr = g[(g.season < s) & (g.season >= "2019-20")]
        te = (g.season == s).to_numpy()
        m = LogisticRegression(C=1.0, max_iter=2000).fit(tr[FALLBACK_FEATS], tr.home_win)
        p[te] = m.predict_proba(g.loc[te, FALLBACK_FEATS])[:, 1]
    return p


# ============================================================================ main
def main():
    t0 = time.time()
    g = load_games()
    blocks = date_blocks(g)
    y = g.home_win.to_numpy(float)
    margin = g.margin.to_numpy(float)
    luck = shooting_luck(g)
    S = schedule_terms(g)
    seas = g.season.to_numpy()
    print(f"[data] {len(g)} games, {len(blocks)} dates ({time.time() - t0:.0f}s)", flush=True)

    preds = run_grid(g, blocks, luck)
    print(f"[grid] {len(preds)} configs ({time.time() - t0:.0f}s)", flush=True)

    names = ["srs_season", "margin_decay", "margin_luck", "margin_luck_sched", "margin_full",
             "od_full", "margin_full_avail"]
    P = {k: np.full(len(g), np.nan) for k in names}
    chosen = {}
    grid_log = []
    # family baseline: plain season-to-date SRS (no recency, no memory of last season, no cap)
    mu_srs, _ = rate(g, blocks, hl=np.inf, carry=0.0, k=1.0, cap=np.inf)
    for s in [VALID_SEASON] + TEST_SEASONS:
        tr = season_mask(g, TUNE_FROM, s)
        te = seas == s
        info = {}
        P["srs_season"][te] = norm.cdf(mu_srs[te] / fit_sigma(mu_srs[tr], y[tr]))
        # tuned decay / prior / cap / per100, no luck adjustment
        ll, cfg0, sig = select(preds, y, tr, allow=lambda c: c[4] == 0.0)
        P["margin_decay"][te] = norm.cdf(preds[cfg0][te] / sig)
        info["margin_decay"] = {**cfg_dict(cfg0), "sigma": round(sig, 3), "train_logloss": round(ll, 5)}
        # + 3-point luck removal (alpha in the grid)
        log = []
        ll, cfg, sig = select(preds, y, tr, log=log)
        grid_log += [{"test_season": s, **row} for row in log]
        mu = preds[cfg]
        P["margin_luck"][te] = norm.cdf(mu[te] / sig)
        info["margin_luck"] = {**cfg_dict(cfg), "sigma": round(sig, 3), "train_logloss": round(ll, 5)}
        hl, carry, k, cap, alpha, per100 = cfg
        ladj = alpha * (luck[0] - luck[1])
        resid = margin - mu
        # + schedule terms: coefficients from earlier seasons' out-of-sample residuals, removed from
        #   the rating targets (each game's terms are known before that game) and added back
        b_s = ols(S.to_numpy(float)[tr], resid[tr])
        adj_s = S.to_numpy(float) @ b_s
        mu_s, _ = rate(g, blocks, hl=hl, carry=carry, k=k, cap=cap, per100=per100, adj=ladj + adj_s)
        mu_s = mu_s + adj_s
        sig = fit_sigma(mu_s[tr], y[tr])
        P["margin_luck_sched"][te] = norm.cdf(mu_s[te] / sig)
        info["margin_luck_sched"] = {"sigma": round(sig, 3), "beta": dict(zip(S.columns, np.round(b_s, 3).tolist()))}
        # + late-season standings terms (threshold by leave-one-season-out on train seasons)
        cut = choose_late_cut(g, S, resid, tr)
        SL = pd.concat([S, late_terms(g, cut)], axis=1)
        b_f = ols(SL.to_numpy(float)[tr], resid[tr])
        adj_f = SL.to_numpy(float) @ b_f
        mu_f, _ = rate(g, blocks, hl=hl, carry=carry, k=k, cap=cap, per100=per100, adj=ladj + adj_f)
        mu_f = mu_f + adj_f
        sig_f = fit_sigma(mu_f[tr], y[tr])
        P["margin_full"][te] = norm.cdf(mu_f[te] / sig_f)
        info["margin_full"] = {"late_cut": cut, "sigma": round(sig_f, 3),
                               "beta": dict(zip(SL.columns, np.round(b_f, 3).tolist()))}
        # O/D split version of margin_full (alpha_off / alpha_def chosen on train seasons)
        best = None
        for a_off, a_def in OD_GRID:
            mo = rate_od(g, blocks, hl=hl, carry=carry, k=k, cap=cap, luck=luck, a_off=a_off, a_def=a_def,
                         adj=adj_f) + adj_f
            so = fit_sigma(mo[tr], y[tr])
            llo = ll_vec(y[tr], norm.cdf(mo[tr] / so)).mean()
            if best is None or llo < best[0]:
                best = (llo, (a_off, a_def), mo, so)
        P["od_full"][te] = norm.cdf(best[2][te] / best[3])
        info["od_full"] = {"alpha_off": best[1][0], "alpha_def": best[1][1], "sigma": round(best[3], 3),
                           "train_logloss": round(best[0], 5)}
        # extension: + previous-game absence proxy (player box scores, 2020-21+ seasons only)
        trp = tr & g.has_players.to_numpy()
        xa = g.avail_prev_diff.to_numpy()
        r2 = margin - mu_f
        b_a = float(xa[trp] @ r2[trp] / (xa[trp] @ xa[trp]))
        mu_a = mu_f + b_a * xa
        sig_a = fit_sigma(mu_a[tr], y[tr])
        P["margin_full_avail"][te] = norm.cdf(mu_a[te] / sig_a)
        info["margin_full_avail"] = {"beta_avail": round(b_a, 4), "sigma": round(sig_a, 3)}
        chosen[s] = info
        print(f"[{s}] {cfg_dict(cfg)} late_cut={cut} od={best[1]} b_avail={b_a:.3f} ({time.time() - t0:.0f}s)",
              flush=True)

    # ---------------------------------------------------------------- evaluation (same games for all)
    ev = load_eval()
    gi = g.reset_index().merge(ev[["GAME_ID", "mkt_close", "mkt_open"]], on="GAME_ID", how="inner")
    E = gi["index"].to_numpy()
    assert len(E) == len(ev) == 5197
    fb = fallback_probs(g)
    preds_all = {**{k: v[E] for k, v in P.items()},
                 "fallback_logit": fb[E], "elo": g.elo_prob.to_numpy()[E],
                 "market_close": gi.mkt_close.to_numpy(), "market_open": gi.mkt_open.to_numpy()}
    yE = y[E]
    sE = seas[E]
    has_open = ~np.isnan(gi.mkt_open.to_numpy())
    for k, v in preds_all.items():
        if k != "market_open":
            assert not np.isnan(v).any(), k
    results = {"eval_set": {"games": int(len(E)), "seasons": TEST_SEASONS,
                            "definition": "research/upsets.py load(): every 2022-23..2025-26 game with an ESPN "
                                          "closing moneyline (reg. season, play-in, playoffs)",
                            "open_subset_games": int(has_open.sum()),
                            "open_subset_definition": "games with an ESPN opening moneyline (2023-24..2025-26)"},
               "metrics": {}, "metrics_open_subset": {}, "bootstrap_logloss_diff": {}, "chosen": chosen}
    for k, v in preds_all.items():
        if k == "market_open":
            continue
        results["metrics"][k] = {"pooled": metrics(yE, v),
                                 **{s: metrics(yE[sE == s], v[sE == s]) for s in TEST_SEASONS}}
    for k, v in preds_all.items():
        m = has_open
        results["metrics_open_subset"][k] = {"pooled": metrics(yE[m], v[m]),
                                             **{s: metrics(yE[m & (sE == s)], v[m & (sE == s)])
                                                for s in TEST_SEASONS[1:]}}
    # 2021-22: an extra walk-forward season (all its games; no market benchmark kept here)
    V = seas == VALID_SEASON
    vpreds = {**{k: v[V] for k, v in P.items()}, "fallback_logit": fb[V], "elo": g.elo_prob.to_numpy()[V]}
    results["validation_2021_22"] = {k: metrics(y[V], v) for k, v in vpreds.items()}
    best = "margin_full_avail"
    for k in ["margin_full", "margin_full_avail", "od_full", "margin_luck_sched"]:
        bt = {"vs_fallback": boot_ci(yE, preds_all[k], preds_all["fallback_logit"]),
              "vs_market_close": boot_ci(yE, preds_all[k], preds_all["market_close"]),
              "vs_market_open (open subset)": boot_ci(yE[has_open], preds_all[k][has_open],
                                                      preds_all["market_open"][has_open]),
              "vs_elo": boot_ci(yE, preds_all[k], preds_all["elo"])}
        bt["per_season_diff_vs_fallback"] = {
            s: round(float((ll_vec(yE, preds_all[k]) - ll_vec(yE, preds_all["fallback_logit"]))[sE == s].mean()), 5)
            for s in TEST_SEASONS}
        results["bootstrap_logloss_diff"][k] = bt
    results["bootstrap_logloss_diff"]["margin_full_avail_vs_margin_full"] = boot_ci(
        yE, preds_all["margin_full_avail"], preds_all["margin_full"])
    results["best_variant"] = best
    # where the gap to the market sits (season phase) and calibration of the best variant
    month = pd.to_datetime(gi.date).dt.month.to_numpy()
    reg = (gi.season_type == "Regular Season").to_numpy()
    phases = {"Oct-Dec": reg & np.isin(month, [10, 11, 12]), "Jan-Feb": reg & np.isin(month, [1, 2]),
              "Mar-Apr": reg & np.isin(month, [3, 4]), "play-in + playoffs": ~reg}
    results["by_phase_logloss"] = {
        ph: {"n": int(m.sum()), **{k: round(float(ll_vec(yE[m], preds_all[k][m]).mean()), 5)
                                   for k in [best, "margin_full", "fallback_logit", "market_close"]}}
        for ph, m in phases.items()}
    pb = preds_all[best]
    bins = np.clip((pb * 10).astype(int), 0, 9)
    results["calibration_best"] = [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int((bins == b).sum()),
                                    "mean_p": round(float(pb[bins == b].mean()), 4),
                                    "home_win_rate": round(float(yE[bins == b].mean()), 4)}
                                   for b in range(10) if (bins == b).any()]
    results["runtime_s"] = round(time.time() - t0, 1)

    out = gi[["GAME_ID", "season", "date", "home", "away", "home_win"]].copy()
    out["p"] = np.round(preds_all[best], 6)
    out.to_csv(HERE / "preds.csv", index=False)
    team_only = out.drop(columns="p").assign(p=np.round(preds_all["margin_full"], 6))
    team_only.to_csv(HERE / "preds_team_only.csv", index=False)
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    pd.DataFrame(grid_log).to_csv(HERE / "grid.csv", index=False)

    # console summary
    rows = []
    for k in preds_all:
        if k == "market_open":
            continue
        m = results["metrics"][k]
        rows.append({"model": k, **{f"ll_{s[2:4]}{s[5:]}": m[s]["logloss"] for s in TEST_SEASONS},
                     "ll_pooled": m["pooled"]["logloss"], "brier": m["pooled"]["brier"], "acc": m["pooled"]["acc"],
                     "ll_open_subset": results["metrics_open_subset"][k]["pooled"]["logloss"]})
    rows.append({"model": "market_open", "ll_open_subset": results["metrics_open_subset"]["market_open"]["pooled"]["logloss"]})
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))
    print("2021-22 validation:", {k: v["logloss"] for k, v in results["validation_2021_22"].items()})
    for k, bt in results["bootstrap_logloss_diff"].items():
        print(k, json.dumps(bt))
    print(f"[done] {time.time() - t0:.0f}s")
    return results


if __name__ == "__main__":
    main()
