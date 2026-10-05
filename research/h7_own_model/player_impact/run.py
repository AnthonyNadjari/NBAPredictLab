"""H7 / player impact: our own pre-game NBA win probabilities built from player ratings.

No betting odds are used as an input. Odds are loaded only to score the market as a baseline.

    python research/h7_own_model/player_impact/run.py          # about 2 minutes -> preds.csv, results.json
    python research/h7_own_model/player_impact/run.py --tune   # how 3 Kalman settings were chosen (2021-22 only)
    python research/h7_own_model/player_impact/run.py --check  # leakage checks on the rating states

Inputs (local, no network): research/data/h1_player_availability/player_logs_*.csv, team_logs.csv,
games_features_odds.csv (outcomes, schedule and team features only), upsets_dataset.csv (market
probabilities, used only as baselines and to define the evaluation set).

Pipeline (details and results in README.md):
  1. Player ratings, updated game by game, from games strictly before the game date:
       a. box-score and plus-minus rates per 100 on-court possessions, recency-weighted
          (exponential decay per appearance) and shrunk toward replacement level
          (fringe players of 2020-21, the burn-in season);
       b. a dynamic game-level adjusted plus-minus ("Kalman APM"): every player has a rating that
          drifts over time; after each day, the ratings of the players who appeared are updated
          from the game margins per 100 possessions (team strength = sum of 5 x minute share x
          rating). Full covariance, so teammates who always share the floor are separated only as
          far as the data allow.
  2. Expected lineup per team-game (availability variants):
       oracle         players who actually played, actual minutes          (upper bound, not deployable)
       oracle_roster  players who actually played, minutes projected       (= a perfect injury report)
       pregame        players seen in the team's last 10 games, minutes = mean of their last 5
                      appearances x P(plays | games missed in a row, minutes); P fitted on 2020-21 and
                      2021-22 (before every test season)
       pregame_hard   players seen in the last 5 team games, out if absent from both of the last 2
     Team composite = sum over the lineup of (5 x minute share) x player rates.
  3. Player rating = beta . rates, beta = ridge fit of home-minus-away composites on the game margin
     per 100 possessions. Team gap = beta . (X_home - X_away).
  4. Probability = logistic(gap, rest diff, back-to-back diff, games-in-5-days diff) + intercept
     (home court); the *_plus_team variants add the fallback's Elo, net-rating and form differences.
     Steps 3-4 are refitted on the 1st of every month on all earlier games since 2021-22.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.linear_model import LogisticRegression, Ridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "research" / "data"
PLOGS = DATA / "h1_player_availability"

TEST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
FIRST_TRAIN = "2021-22"          # 2020-21 = burn-in for player ratings (no earlier player logs)
FALLBACK_FEATS = ["elo_diff", "net_diff", "form_diff", "rest_diff", "b2b_diff", "g5_diff"]
SCHED = ["rest_diff", "b2b_diff", "g5_diff"]
TEAM_EXTRA = ["elo_diff", "net_diff", "form_diff"]

# ---- hyperparameters: never tuned on the test seasons (set a priori unless stated)
BOX = ["PTS", "FGA", "FTA", "FG3M", "FG3A", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
HL_BOX = 60.0        # half-life of box-score weights, in player appearances
HL_PM = 120.0        # half-life of plus-minus weights (noisier, longer memory)
K_BOX = 1500.0       # shrinkage toward replacement, in on-court possessions (~25 starter games)
K_PM = 4000.0        # same for on-court plus-minus
K_OO = 4000.0        # same for on/off (effective possessions)
# KAL: entry_offset, p0 and q_day were chosen on 2021-22 only (one-step-ahead margin error of the
# filter, grid in README), i.e. before any test season; the rest was set a priori.
KAL = {"prior_mean": -1.0,   # starting level (the level itself cancels out: 5 shares per team)
       "entry_offset": -4.0, # a newcomer enters this far below the current average established player
       "ref_rho": 0.05,      # daily smoothing of that average
       "burnin_end": "2021-01-31",  # first weeks of the data: everybody is new, no offset
       "p0": 4.0,            # prior variance of a new player (sd 2)
       "q_day": 0.012,       # in-season drift variance per day (sd ~1.4 over a season)
       "q_season": 0.5,      # extra variance at each new season (ageing, development)
       "sigma2": 150.0,      # game noise variance of the margin per 100 possessions
       "hca0": 2.5, "hca_p0": 1.0}
WIN = 10             # team games scanned for the expected rotation
N_MIN = 5            # appearances averaged for projected minutes
P_PLAY_FIT_END = "2022-10-01"   # P(plays) table fitted on candidates before the first test season
RIDGE_ALPHA = 10.0   # on standardized composites
N_BOOT = 2000
SEED = 7

RATE_COLS = [f"r_{c}" for c in BOX] + ["r_PM", "r_ONOFF"]
XCOLS = RATE_COLS + ["r_KAL"]
RATING_SETS = {"full": XCOLS, "box_pm": RATE_COLS, "kalman": ["r_KAL"]}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------- data
def load_players() -> pd.DataFrame:
    fs = sorted(PLOGS.glob("player_logs_*.csv"))
    p = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    p["GAME_ID"] = p.GAME_ID.astype(int)
    p = p.drop_duplicates(["PLAYER_ID", "GAME_ID"])
    p["date"] = pd.to_datetime(p.GAME_DATE)
    for c in ["MIN", "PLUS_MINUS"] + BOX:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0).astype(float)
    return p[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "date", "SEASON", "MIN", "PLUS_MINUS"] + BOX]


def load_team_games() -> pd.DataFrame:
    t = pd.read_csv(DATA / "team_logs.csv")
    t = t[t.SEASON >= "2020-21"].copy()
    t["GAME_ID"] = t.GAME_ID.astype(int)
    t = t.drop_duplicates(["TEAM_ID", "GAME_ID"])
    t["date"] = pd.to_datetime(t.GAME_DATE)
    t["poss_t"] = t.FGA - t.OREB + t.TOV + 0.44 * t.FTA
    t["poss"] = t.groupby("GAME_ID").poss_t.transform("mean")
    t["is_home"] = t.MATCHUP.str.contains(" vs. ")
    t = t.rename(columns={"MIN": "team_min", "PLUS_MINUS": "margin", "SEASON": "season"})
    return t[["TEAM_ID", "GAME_ID", "date", "season", "poss", "team_min", "margin", "is_home"]]


def load_games() -> pd.DataFrame:
    """One row per game: outcome, schedule features, Elo and the fallback features
    (games_features_odds.csv, built by research/features.py from earlier games only)."""
    g = pd.read_csv(DATA / "games_features_odds.csv")
    g["date"] = pd.to_datetime(g.date)
    keep = ["GAME_ID", "date", "season", "season_type", "home_id", "away_id", "home", "away", "home_win",
            "margin", "poss_h", "poss_a", "elo_prob"] + FALLBACK_FEATS
    g = g[keep].copy()
    g["poss"] = (g.poss_h + g.poss_a) / 2
    g["y100"] = 100 * g.margin / g.poss       # training target only (outcome of past games)
    return g


def load_market() -> pd.DataFrame:
    """Evaluation set: the shared 5,197-game protocol set (research/upsets.py), de-vigged ESPN lines."""
    u = pd.read_csv(DATA / "upsets_dataset.csv", usecols=["game_id", "mkt_open", "mkt_close"])
    return u.rename(columns={"game_id": "GAME_ID"})


# --------------------------------------------------------------------------- 1a. rate ratings
def player_states(p: pd.DataFrame, t: pd.DataFrame):
    """Post-appearance state of every player: shrunk per-100 rates after each of his games.
    Looked up later with 'last appearance strictly before the game date'."""
    q = p.merge(t[["TEAM_ID", "GAME_ID", "poss", "team_min", "margin"]], on=["TEAM_ID", "GAME_ID"], how="left")
    q["team_min"] = q.team_min.fillna(240.0)
    q["poss"] = q.poss.fillna(q.poss.median())
    q["margin"] = q.margin.fillna(0.0)
    q["on_poss"] = q.MIN / (q.team_min / 5.0) * q.poss
    q["off_poss"] = (q.poss - q.on_poss).clip(lower=0)
    q["off_pm"] = q.margin - q.PLUS_MINUS
    q = q.sort_values(["PLAYER_ID", "date"]).reset_index(drop=True)

    # replacement level = fringe players (< 12 min per appearance) of 2020-21, the burn-in season
    f = q[q.SEASON == "2020-21"]
    avg = f.groupby("PLAYER_ID").MIN.mean()
    f = f[f.PLAYER_ID.isin(avg[avg < 12].index)]
    prior_box = 100 * f[BOX].sum().to_numpy() / f.on_poss.sum()
    prior_pm = 100 * f.PLUS_MINUS.sum() / f.on_poss.sum()

    lam_b, lam_p = 0.5 ** (1 / HL_BOX), 0.5 ** (1 / HL_PM)
    xb = q[["on_poss"] + BOX].to_numpy(float)
    xp = q[["on_poss", "PLUS_MINUS", "off_pm", "off_poss"]].to_numpy(float)
    sb, sp = np.empty_like(xb), np.empty_like(xp)
    pid = q.PLAYER_ID.to_numpy()
    cut = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1, len(q)]
    for a, b in zip(cut[:-1], cut[1:]):        # S_k = lam * S_{k-1} + x_k, per player
        sb[a:b] = lfilter([1.0], [1.0, -lam_b], xb[a:b], axis=0)
        sp[a:b] = lfilter([1.0], [1.0, -lam_p], xp[a:b], axis=0)

    rates = 100 * (sb[:, 1:] + K_BOX * prior_box / 100) / (sb[:, [0]] + K_BOX)
    r_pm = 100 * (sp[:, 1] + K_PM * prior_pm / 100) / (sp[:, 0] + K_PM)
    on = 100 * sp[:, 1] / np.maximum(sp[:, 0], 1e-6)
    off = 100 * sp[:, 2] / np.maximum(sp[:, 3], 1e-6)
    neff = 1.0 / (1.0 / np.maximum(sp[:, 0], 1e-6) + 1.0 / np.maximum(sp[:, 3], 1e-6))
    r_oo = np.where((sp[:, 0] > 1) & (sp[:, 3] > 1), (on - off) * neff / (neff + K_OO), 0.0)

    st = pd.DataFrame(rates, columns=RATE_COLS[:len(BOX)])
    st["r_PM"], st["r_ONOFF"] = r_pm, r_oo
    st["PLAYER_ID"], st["date"], st["last_team"] = q.PLAYER_ID, q.date, q.TEAM_ID
    st["min_last5"] = q.groupby("PLAYER_ID").MIN.transform(lambda s: s.rolling(N_MIN, min_periods=1).mean())
    st["name"] = q.PLAYER_NAME
    prior = dict(zip(RATE_COLS, list(prior_box) + [prior_pm, 0.0]))
    return st, prior


# --------------------------------------------------------------------------- 1b. Kalman APM
class KalmanAPM:
    """Dynamic game-level adjusted plus-minus. State: home court + one rating per player.
    Observation for a game: margin/100 poss = hca + sum_home s_i r_i - sum_away s_j r_j + noise,
    s = 5 x actual minute share (known once the game is played). All games of a day are absorbed
    together after that day; R[d] is the state before the games of date d."""

    def __init__(self, p: pd.DataFrame, t: pd.DataFrame, cfg=KAL):
        q = p[p.MIN > 0].merge(t[["TEAM_ID", "GAME_ID", "is_home"]], on=["TEAM_ID", "GAME_ID"])
        q["s"] = 5 * q.MIN / q.groupby(["GAME_ID", "TEAM_ID"]).MIN.transform("sum")
        q["a"] = np.where(q.is_home, q.s, -q.s)
        pids = np.sort(q.PLAYER_ID.unique())
        self.k = pd.Series(np.arange(1, len(pids) + 1), index=pids)
        q["k"] = self.k[q.PLAYER_ID].to_numpy()
        h = t[t.is_home].copy()
        h["y"] = 100 * h.margin / h.poss
        h = h[h.GAME_ID.isin(q.GAME_ID)].sort_values(["date", "GAME_ID"])
        n = len(pids) + 1
        r = np.full(n, cfg["prior_mean"])
        r[0] = cfg["hca0"]
        P = np.diag(np.full(n, cfg["p0"]))
        P[0, 0] = cfg["hca_p0"]
        seen = np.zeros(n, bool)
        seen[0] = True
        self.dates = np.sort(h.date.unique())
        R = np.empty((len(self.dates) + 1, n), dtype=np.float32)
        ref = cfg["prior_mean"]          # running minutes-weighted mean rating of established players
        REF = np.empty(len(self.dates) + 1)
        qg = dict(tuple(q.groupby("GAME_ID")[["k", "a"]]))
        prev, prev_season = None, None
        self.onestep = []
        for di, (d, hd) in enumerate(h.groupby("date", sort=True)):
            season = hd.season.iloc[0]
            if prev is not None:
                idx = np.flatnonzero(seen)
                add = cfg["q_season"] if season != prev_season else cfg["q_day"] * (d - prev).days
                P[idx, idx] += add
            off = cfg["entry_offset"] if d > np.datetime64(cfg["burnin_end"]) else 0.0
            r[~seen] = ref + off                     # a newcomer enters below the current average
            R[di], REF[di] = r, ref + off
            gids = hd.GAME_ID.to_numpy()
            ks = [qg[gid].k.to_numpy() for gid in gids]
            J = np.unique(np.concatenate([[0]] + ks))
            pos = pd.Series(np.arange(len(J)), index=J)
            A = np.zeros((len(gids), len(J)))
            A[:, 0] = 1.0
            for m, gid in enumerate(gids):
                A[m, pos[qg[gid].k].to_numpy()] = qg[gid].a.to_numpy()
            PAt = P[:, J] @ A.T                                       # n x m
            S = A @ PAt[J] + cfg["sigma2"] * np.eye(len(gids))
            Kt = np.linalg.solve(S, PAt.T)                            # m x n
            pre = A @ r[J]                                            # one-step-ahead margin
            self.onestep.append(pd.DataFrame({"GAME_ID": gids, "pred": pre, "y": hd.y.to_numpy()}))
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
        self.onestep = pd.concat(self.onestep, ignore_index=True)
        self.R, self.REF = R, REF
        self.hca = float(r[0])

    def lookup(self, pid, dates) -> np.ndarray:
        """Rating of each player before the games of each date (entry value if never seen)."""
        di = np.searchsorted(self.dates, np.asarray(dates, dtype="datetime64[ns]"), side="left")
        k = self.k.reindex(pid).to_numpy()
        out = self.REF[di].copy()
        ok = ~np.isnan(k)
        out[ok] = self.R[di[ok], k[ok].astype(int)]
        return out


def attach_state(rows: pd.DataFrame, st: pd.DataFrame, prior: dict, kal: KalmanAPM) -> pd.DataFrame:
    """Ratings of each (player, date) from his games strictly before that date."""
    rows = rows.assign(date=rows.date.astype("datetime64[ns]")).sort_values("date").reset_index(drop=True)
    out = pd.merge_asof(rows, st.drop(columns=["name"]).sort_values("date"), on="date", by="PLAYER_ID",
                        allow_exact_matches=False, direction="backward")
    for c in RATE_COLS:
        out[c] = out[c].fillna(prior[c])
    out["r_KAL"] = kal.lookup(out.PLAYER_ID.to_numpy(), out.date.to_numpy())
    return out


# --------------------------------------------------------------------------- 2. expected lineups
def candidate_rows(p: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """For each team-game j: every player seen for the team in its previous WIN games
    (any season type; the window crosses into last season for the first games of a season).
    `played` / `min_now` (the game's own box score) are kept only to fit P(plays) on past
    seasons and for the oracle variants; they never enter a pre-game lineup."""
    out = []
    pp = p[["PLAYER_ID", "TEAM_ID", "GAME_ID", "MIN"]]
    for team, tg in t.sort_values(["date", "GAME_ID"]).groupby("TEAM_ID"):
        gids, dates = tg.GAME_ID.to_numpy(), tg.date.to_numpy()
        pt = pp[pp.TEAM_ID == team]
        players = pt.PLAYER_ID.unique()
        pidx = pd.Series(np.arange(len(players)), index=players)
        gidx = pd.Series(np.arange(len(gids)), index=gids)
        pt = pt[pt.GAME_ID.isin(gidx.index)]
        M = np.full((len(gids), len(players)), np.nan)
        M[gidx[pt.GAME_ID].to_numpy(), pidx[pt.PLAYER_ID].to_numpy()] = pt.MIN.to_numpy()
        P = ~np.isnan(M)
        for j in range(1, len(gids)):
            lo = max(0, j - WIN)
            W = P[lo:j]
            cand = np.flatnonzero(W.any(0))
            Wc, Mc = W[:, cand][::-1], np.nan_to_num(M[lo:j, cand][::-1])  # most recent first
            streak = np.where(Wc.any(0), Wc.argmax(0), Wc.shape[0])        # team games missed in a row
            take = Wc & (np.cumsum(Wc, 0) <= N_MIN)
            m_hat = (Mc * take).sum(0) / take.sum(0)
            W5, M5 = Wc[:5], Mc[:5]
            in5 = W5.any(0)
            m5 = np.where(in5, (M5 * W5).sum(0) / np.maximum(W5.sum(0), 1), 0.0)
            out.append(pd.DataFrame({
                "TEAM_ID": team, "GAME_ID": gids[j], "date": dates[j], "PLAYER_ID": players[cand],
                "m_hat": m_hat, "streak": streak, "in5": in5, "m5": m5, "miss2": ~Wc[:2].any(0),
                "played": P[j, cand], "min_now": np.nan_to_num(M[j, cand])}))
    return pd.concat(out, ignore_index=True)


def play_key(c: pd.DataFrame) -> np.ndarray:
    s, m = c.streak.to_numpy(), c.m_hat.to_numpy()
    sb = np.select([s == 0, s == 1, s == 2, s <= 4, s <= 9], [0, 1, 2, 3, 4], 5)
    return sb * 10 + np.select([m < 12, m < 24], [0, 1], 2)


def fit_p_play(cand: pd.DataFrame) -> pd.Series:
    """P(plays | team games missed in a row, projected minutes), from team-games before the
    first test season (2020-21 and 2021-22)."""
    c = cand[cand.date < pd.Timestamp(P_PLAY_FIT_END)]
    g = pd.DataFrame({"k": play_key(c), "y": c.played.astype(float)}).groupby("k").y.agg(["sum", "count"])
    return (g["sum"] + 1) / (g["count"] + 2)


def composites(rows: pd.DataFrame, wcol: str) -> pd.DataFrame:
    """Team composite X = sum_i (5 x minute share_i) x rates_i, one row per (GAME_ID, TEAM_ID)."""
    rows = rows[rows[wcol] > 0]
    s = 5 * rows[wcol] / rows.groupby(["GAME_ID", "TEAM_ID"])[wcol].transform("sum")
    X = rows[XCOLS].mul(s, axis=0)
    X["GAME_ID"], X["TEAM_ID"] = rows.GAME_ID.to_numpy(), rows.TEAM_ID.to_numpy()
    return X.groupby(["GAME_ID", "TEAM_ID"], as_index=False)[XCOLS].sum()


def game_diffs(X: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    h = g[["GAME_ID", "home_id"]].merge(X, left_on=["GAME_ID", "home_id"], right_on=["GAME_ID", "TEAM_ID"])
    a = g[["GAME_ID", "away_id"]].merge(X, left_on=["GAME_ID", "away_id"], right_on=["GAME_ID", "TEAM_ID"])
    d = h[["GAME_ID"] + XCOLS].merge(a[["GAME_ID"] + XCOLS], on="GAME_ID", suffixes=("_h", "_a"))
    out = pd.DataFrame({"GAME_ID": d.GAME_ID})
    for c in XCOLS:
        out[f"d_{c}"] = d[f"{c}_h"] - d[f"{c}_a"]
    return out


# --------------------------------------------------------------------------- 3-4. walk-forward models
def refit_dates(g: pd.DataFrame, season: str):
    d = g[g.season == season].date
    start, end = d.min(), d.max()
    months = pd.date_range(start.to_period("M").to_timestamp() + pd.offsets.MonthBegin(1), end, freq="MS")
    return [start] + list(months) + [end + pd.Timedelta(days=1)]


class GapModel:
    """Ridge of margin per 100 poss on standardized home-minus-away composites."""

    def fit(self, D, y):
        self.mu, self.sd = D.mean(0), D.std(0) + 1e-9
        self.m = Ridge(alpha=RIDGE_ALPHA).fit((D - self.mu) / self.sd, y)
        self.beta = self.m.coef_ / self.sd          # weights on raw per-100 rates = player rating formula
        return self

    def gap(self, D):
        return D @ self.beta                         # home minus away, points per 100 poss


def walk_forward(g, D, rating_cols, extra=(), collect=None):
    """Monthly refits on all games since FIRST_TRAIN before the refit date. Test-season predictions."""
    dcols = [f"d_{c}" for c in rating_cols]
    d = g.merge(D, on="GAME_ID", how="inner")
    pred = pd.Series(np.nan, index=d.GAME_ID.to_numpy())
    cols = ["gap"] + SCHED + list(extra)
    for s in TEST_SEASONS:
        ds = refit_dates(d, s)
        for lo, hi in zip(ds[:-1], ds[1:]):
            tr = d[(d.season >= FIRST_TRAIN) & (d.date < lo)]
            te = d[(d.season == s) & (d.date >= lo) & (d.date < hi)]
            if te.empty:
                continue
            gm = GapModel().fit(tr[dcols].to_numpy(), tr.y100.to_numpy())
            Xtr = tr[SCHED + list(extra)].assign(gap=gm.gap(tr[dcols].to_numpy()))[cols]
            Xte = te[SCHED + list(extra)].assign(gap=gm.gap(te[dcols].to_numpy()))[cols]
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, tr.home_win)
            pred.loc[te.GAME_ID.to_numpy()] = lr.predict_proba(Xte)[:, 1]
            if collect is not None:
                collect.append({"season": s, "refit": str(lo.date()), "n_train": int(len(tr)),
                                "beta": dict(zip(rating_cols, np.round(gm.beta, 4).tolist())),
                                "logit": dict(zip(cols, lr.coef_[0].round(4).tolist())),
                                "intercept": round(float(lr.intercept_[0]), 4)})
    return pred


def fallback_preds(g: pd.DataFrame) -> pd.Series:
    """research/backtest.py: logit on FEATS, trained on seasons 2019-20 .. S-1, once per season."""
    pred = pd.Series(np.nan, index=g.GAME_ID.to_numpy())
    for s in TEST_SEASONS:
        tr = g[(g.season < s) & (g.season >= "2019-20")]
        te = g[g.season == s]
        m = LogisticRegression(C=1.0, max_iter=2000).fit(tr[FALLBACK_FEATS], tr.home_win)
        pred.loc[te.GAME_ID.to_numpy()] = m.predict_proba(te[FALLBACK_FEATS])[:, 1]
    return pred


def team_logit_monthly(g: pd.DataFrame, feats) -> pd.Series:
    """Team-only control with the same monthly refits as the player models."""
    pred = pd.Series(np.nan, index=g.GAME_ID.to_numpy())
    for s in TEST_SEASONS:
        ds = refit_dates(g, s)
        for lo, hi in zip(ds[:-1], ds[1:]):
            tr = g[(g.season >= FIRST_TRAIN) & (g.date < lo)]
            te = g[(g.season == s) & (g.date >= lo) & (g.date < hi)]
            if te.empty:
                continue
            m = LogisticRegression(C=1.0, max_iter=2000).fit(tr[feats], tr.home_win)
            pred.loc[te.GAME_ID.to_numpy()] = m.predict_proba(te[feats])[:, 1]
    return pred


# --------------------------------------------------------------------------- evaluation
def ll_vec(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(y, p):
    return {"n": int(len(y)), "logloss": round(float(ll_vec(y, p).mean()), 5),
            "brier": round(float(((p - y) ** 2).mean()), 5), "acc": round(float(((p > 0.5) == (y == 1)).mean()), 4)}


def boot_ci(a, b, rng):
    """Paired bootstrap over games of mean(a - b), per-game log-loss (negative = a better)."""
    d = a - b
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    bs = d[idx].mean(1)
    return {"diff": round(float(d.mean()), 5), "ci95": [round(float(np.quantile(bs, 0.025)), 5),
                                                      round(float(np.quantile(bs, 0.975)), 5)]}


MODELS = {  # name: (lineup, rating set, extra team features)
    "pregame": ("pregame", "full", ()),
    "pregame_hard": ("pregame_hard", "full", ()),
    "oracle_roster": ("oracle_roster", "full", ()),
    "oracle": ("oracle", "full", ()),
    "pregame_box_pm_only": ("pregame", "box_pm", ()),
    "pregame_kalman_only": ("pregame", "kalman", ()),
    "pregame_plus_team": ("pregame", "full", TEAM_EXTRA),
    "pregame_kalman_plus_team": ("pregame", "kalman", TEAM_EXTRA),
    "oracle_roster_plus_team": ("oracle_roster", "full", TEAM_EXTRA),
}
BASELINES = ["fallback", "elo", "team_logit_monthly", "market_close", "market_open"]
# lowest pooled log-loss among the deployable (no same-game information) variants; picked on the
# test seasons themselves, see README (the deployable variants differ by <= 0.004)
BEST = "pregame_plus_team"


def tune_kalman():
    """How entry_offset, p0 and q_day were chosen: one-step-ahead error of the filter's own margin
    prediction on 2021-22 games, with data cut at 2022-10-01 (no test season involved). ~5 min."""
    import itertools
    p, t = load_players(), load_team_games()
    cut = pd.Timestamp(P_PLAY_FIT_END)
    p, t = p[p.date < cut], t[t.date < cut]
    val = set(t[t.season == "2021-22"].GAME_ID)
    rows = []
    for off, p0, qd in itertools.product([0.0, -1.0, -2.0, -3.0, -4.0, -5.0], [2.0, 4.0, 8.0], [0.004, 0.012]):
        o = KalmanAPM(p, t, dict(KAL, entry_offset=off, p0=p0, q_day=qd)).onestep
        o = o[o.GAME_ID.isin(val)]
        rows.append({"entry_offset": off, "p0": p0, "q_day": qd, "mse_2021_22": round(((o.y - o.pred) ** 2).mean(), 2)})
    print(pd.DataFrame(rows).sort_values("mse_2021_22").to_string(index=False))


def leak_check(d0="2024-01-15"):
    """Ratings looked up for date d0 must not move when the games of d0 (or later) change."""
    p, t = load_players(), load_team_games()
    d0 = pd.Timestamp(d0)
    full = KalmanAPM(p, t)
    pids = full.k.index.to_numpy()
    at = lambda k, d: k.lookup(pids, np.full(len(pids), np.datetime64(d, "ns")))
    trunc = KalmanAPM(p[p.date < d0], t[t.date < d0])
    t2 = t.copy()
    t2.loc[t2.date == d0, "margin"] += 50
    pert = KalmanAPM(p, t2)
    print("Kalman: full vs truncated-before-d0, max |diff| at d0:", float(np.abs(at(full, d0) - at(trunc, d0)).max()))
    print("Kalman: same-day margins +50, max |diff| at d0:", float(np.abs(at(full, d0) - at(pert, d0)).max()),
          "| the day after:", float(np.abs(at(full, d0 + pd.Timedelta(days=1)) - at(pert, d0 + pd.Timedelta(days=1))).max()))
    st, prior = player_states(p, t)
    p2 = p.copy()
    p2.loc[p2.date == d0, "PLUS_MINUS"] += 30
    p2.loc[p2.date == d0, "PTS"] += 30
    st2, _ = player_states(p2, t)
    rows = pd.DataFrame({"PLAYER_ID": pids, "date": d0})
    a, b = attach_state(rows, st, prior, full), attach_state(rows, st2, prior, full)
    print("rates: same-day PTS/+- changed, max |diff| at d0:",
          float(np.abs(a[RATE_COLS].to_numpy() - b[RATE_COLS].to_numpy()).max()))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if "--tune" in sys.argv:
        return tune_kalman()
    if "--check" in sys.argv:
        return leak_check()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    p, t, g, mk = load_players(), load_team_games(), load_games(), load_market()
    log(f"player-games {len(p):,} | team-games {len(t):,} | games {len(g):,}")

    st, prior = player_states(p, t)
    kal = KalmanAPM(p, t)
    log(f"player ratings built (Kalman home court at the end: {kal.hca:.2f} per 100)")

    cand = attach_state(candidate_rows(p, t), st, prior, kal)
    cand = cand[cand.last_team == cand.TEAM_ID].copy()      # traded / released since: not on the team
    tab = fit_p_play(cand)
    cand["e_min"] = pd.Series(play_key(cand)).map(tab).fillna(0.5).to_numpy() * cand.m_hat.to_numpy()
    log(f"candidate rows {len(cand):,}")

    orc = attach_state(p[p.MIN > 0][["PLAYER_ID", "TEAM_ID", "GAME_ID", "date", "MIN"]], st, prior, kal)
    orc = orc.merge(cand[["TEAM_ID", "GAME_ID", "PLAYER_ID", "m_hat"]], on=["TEAM_ID", "GAME_ID", "PLAYER_ID"],
                    how="left")
    orc["m_proj"] = orc.m_hat.fillna(orc.min_last5).fillna(10.0)   # new arrival: his recent minutes elsewhere

    gg = g[g.season >= "2020-21"]
    lineups = {"oracle": game_diffs(composites(orc, "MIN"), gg),
               "oracle_roster": game_diffs(composites(orc, "m_proj"), gg),
               "pregame": game_diffs(composites(cand, "e_min"), gg),
               "pregame_hard": game_diffs(composites(cand[cand.in5 & ~cand.miss2], "m5"), gg)}
    log("lineup composites built")

    preds, refits = {}, {}
    for name, (lu, rs, extra) in MODELS.items():
        refits[name] = []
        preds[name] = walk_forward(gg, lineups[lu], RATING_SETS[rs], extra, refits[name])
    preds["fallback"] = fallback_preds(g)
    preds["team_logit_monthly"] = team_logit_monthly(g, FALLBACK_FEATS)
    log("walk-forward done")

    # ---------------------------------------------------------------- evaluation on one game set
    ev = g[g.season.isin(TEST_SEASONS)].merge(mk, on="GAME_ID", how="inner")
    ev["elo"], ev["market_close"], ev["market_open"] = ev.elo_prob, ev.mkt_close, ev.mkt_open
    for k, s in preds.items():
        ev[k] = ev.GAME_ID.map(s)
    names = list(MODELS) + BASELINES
    n0 = len(ev)
    ev = ev.dropna(subset=[n for n in names if n != "market_open"]).reset_index(drop=True)
    log(f"evaluation games {len(ev):,} of {n0:,}")

    y = ev.home_win.to_numpy().astype(float)
    eo = ev[ev.market_open.notna()]
    yo = eo.home_win.to_numpy().astype(float)
    res = {"n_games": len(ev), "n_games_with_open": len(eo), "best_variant": BEST,
           "pooled": {}, "pooled_open_subset": {}, "seasons": {}, "bootstrap": {}}
    for n in names:
        if n == "market_open":
            res["pooled"][n] = {**metrics(yo, eo[n].to_numpy()), "note": "opening lines exist from 2023-24 only"}
        else:
            res["pooled"][n] = metrics(y, ev[n].to_numpy())
        res["pooled_open_subset"][n] = metrics(yo, eo[n].to_numpy())
    for s in TEST_SEASONS:
        e = ev[ev.season == s]
        res["seasons"][s] = {n: metrics(e.home_win.to_numpy().astype(float), e[n].to_numpy())
                             for n in names if e[n].notna().all()}
    L = {n: ll_vec(y, ev[n].to_numpy()) for n in names if n != "market_open"}
    Lo = {n: ll_vec(yo, eo[n].to_numpy()) for n in names}
    for n in list(MODELS) + ["elo", "team_logit_monthly", "market_close"]:
        b = {"vs_fallback": boot_ci(L[n], L["fallback"], rng)}
        if n != "market_close":
            b["vs_market_close"] = boot_ci(L[n], L["market_close"], rng)
        b["vs_market_open_open_subset"] = boot_ci(Lo[n], Lo["market_open"], rng)
        b["seasons_better_than_fallback"] = int(sum(
            res["seasons"][s][n]["logloss"] < res["seasons"][s]["fallback"]["logloss"] for s in TEST_SEASONS))
        res["bootstrap"][n] = b

    # where the gains are (season phase) and calibration slope (1 = calibrated, < 1 = overconfident)
    mo = ev.date.dt.month
    phase = np.select([ev.season_type != "Regular Season", mo.isin([10, 11]), mo.isin([3, 4])],
                      ["4_playoffs_playin", "1_oct_nov", "3_mar_apr"], "2_dec_feb")
    res["by_phase_logloss"] = {ph: {"n": int((phase == ph).sum()),
                                    **{n: round(float(L[n][phase == ph].mean()), 5) for n in L}}
                               for ph in sorted(set(phase))}
    lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
    res["calibration_slope"] = {n: round(float(LogisticRegression(C=1e6).fit(
        lg(ev[n].to_numpy()).reshape(-1, 1), y).coef_[0][0]), 3) for n in L}

    # ---------------------------------------------------------------- sanity: who the model rates highest
    last = refits["pregame"][-1]
    beta = np.array([last["beta"][c] for c in XCOLS])
    who = st[(st.date > pd.Timestamp("2026-02-01")) & (st.date < pd.Timestamp("2026-04-13"))]
    who = who.groupby("PLAYER_ID").tail(1)[["PLAYER_ID", "name"]].assign(date=pd.Timestamp("2026-04-13"))
    who = attach_state(who, st, prior, kal)
    who["rating"] = who[XCOLS].to_numpy() @ beta
    who["rating"] -= who.rating.median()
    res["top_players_end_2025_26"] = [[r.name, round(float(r.rating), 2), round(float(r.r_KAL), 2)]
                                      for r in who.nlargest(15, "rating").itertuples()]
    res["p_play_table"] = {f"streak_bucket{k // 10}_min_bucket{k % 10}": round(float(v), 3) for k, v in tab.items()}
    res["hyperparameters"] = {"HL_BOX": HL_BOX, "HL_PM": HL_PM, "K_BOX": K_BOX, "K_PM": K_PM, "K_OO": K_OO,
                              "KAL": KAL, "WIN": WIN, "N_MIN": N_MIN, "RIDGE_ALPHA": RIDGE_ALPHA,
                              "N_BOOT": N_BOOT, "SEED": SEED}
    res["last_refit"] = {n: refits[n][-1] for n in MODELS}

    out = ev[["GAME_ID", "season", "date", "home", "away", "home_win"]].copy()
    out["p"] = ev[BEST].round(5)
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    out.to_csv(HERE / "preds.csv", index=False)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "--dump":
        ev.to_csv(sys.argv[2], index=False)

    # ---------------------------------------------------------------- console summary
    pooled = pd.DataFrame({n: (v or {}) for n, v in res["pooled"].items()}).T
    print("\npooled, all", len(ev), "games:\n", pooled.to_string())
    print("\nopening-line subset (2023-24..2025-26):\n", pd.DataFrame(res["pooled_open_subset"]).T.to_string())
    for k in ("logloss", "acc"):
        print(f"\nper-season {k}:\n", pd.DataFrame({s: {n: v[k] for n, v in r.items()}
                                                    for s, r in res["seasons"].items()}).to_string())
    print("\npaired bootstrap, log-loss difference (model - reference), 95% CI:")
    for n, b in res["bootstrap"].items():
        f = lambda x: f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
        line = f"  {n:24s} vs fallback {f(b['vs_fallback'])}"
        if "vs_market_close" in b:
            line += f" | vs close {f(b['vs_market_close'])}"
        print(line + f" | vs open {f(b['vs_market_open_open_subset'])} | seasons < fallback "
                     f"{b['seasons_better_than_fallback']}/4")
    print("\ntop players end 2025-26 (rating, Kalman part):", res["top_players_end_2025_26"])
    print("last refit pregame:", res["last_refit"]["pregame"])
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
