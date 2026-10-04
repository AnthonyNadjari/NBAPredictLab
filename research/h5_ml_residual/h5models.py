"""H5 models: flexible learners that start FROM the market price and try to learn its residual.

All fitting functions take a training frame and return a predictor; every tuning choice (LightGBM
config + number of rounds, L2 strength) is made on an inner, time-based validation split INSIDE the
training seasons (the last training season, or the last 30% of games by date if only one season).
The outer test season is never seen during fitting or tuning.
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

PROBIT_TO_LOGIT = 1.702  # logit(Phi(z)) ~= 1.702 z


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ------------------------------------------------------------------------------------------- features
VENUE = [("team_home_win_pct", "team_road_win_pct"), ("team_home_ppg", "team_road_ppg"),
         ("team_home_point_diff", "team_road_point_diff"), ("team_home_fg_pct", "team_road_fg_pct")]


def per_team_stats(d):
    return [c[5:] for c in d.columns if c.startswith("home_") and f"away_{c[5:]}" in d.columns
            and c[5:] not in ("win", "pts")]


def season_day(d):
    start = pd.to_datetime(d.groupby("season").game_date.transform("min"))
    return (pd.to_datetime(d.game_date) - start).dt.days


def make_features(d, mode):
    """All pre-game features. mode='close': market inputs known at the close (closing ML, spread,
    total, open->close move). mode='open': only the opening ML -- the stored spread/total are closing
    numbers, so they are excluded to keep the opening model honest."""
    X = pd.DataFrame(index=d.index)
    stats = per_team_stats(d)
    for s in stats:
        X[f"home_{s}"], X[f"away_{s}"] = d[f"home_{s}"], d[f"away_{s}"]
        X[f"diff_{s}"] = d[f"home_{s}"] - d[f"away_{s}"]
    for h, a in VENUE:
        X[f"home_{h}"], X[f"away_{a}"] = d[f"home_{h}"], d[f"away_{a}"]
        X[f"diff_{h}"] = d[f"home_{h}"] - d[f"away_{a}"]
    for c in ("elo_diff", "elo_win_prob", "net_diff", "form_diff", "rest_diff", "b2b_diff", "g5_diff"):
        X[c] = d[c]
    X["month"] = d.month
    X["playoffs"] = d.playoffs
    X["playin"] = (d.season_type == "PlayIn").astype(int)
    X["neutral"] = d.neutral
    X["season_day"] = season_day(d)
    X["weekday"] = pd.to_datetime(d.game_date).dt.weekday
    if mode == "close":
        L = logit(d.p_close)
        X["mkt_logit"] = L
        X["vig"] = d.vig_close
        X["spread"] = d.spread
        X["total"] = d.total
        X["spread_missing"] = d.spread_missing
        X["spread_vs_ml"] = logit(norm.cdf(-d.spread / 12.5)) - L
        X["line_move"] = L - logit(d.p_open)  # NaN when no opening odds
        X.loc[d.p_open.isna(), "line_move"] = np.nan
    else:
        L = logit(d.p_open)
        X["mkt_logit"] = L
        X["vig"] = d.vig_open
    X["elo_gap"] = logit(d.elo_win_prob) - L
    return X


def offset_of(d, mode):
    return logit(d.p_close if mode == "close" else d.p_open)


def inner_split(tr):
    """Time-based inner validation inside the training seasons."""
    seasons = sorted(tr.season.unique())
    if len(seasons) >= 2:
        va = tr.season == seasons[-1]
    else:
        cut = tr.game_date.sort_values().iloc[int(0.7 * len(tr))]
        va = tr.game_date >= cut
    return ~va.to_numpy(), va.to_numpy()


# ------------------------------------------------------------------------------------------- LightGBM
LGB_BASE = dict(verbose=-1, seed=7, deterministic=True, force_row_wise=True, num_threads=4,
                boost_from_average=False, feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
                feature_fraction_seed=7, bagging_seed=7, data_random_seed=7)
LGB_CONFIGS = [
    dict(num_leaves=2, max_depth=1, min_data_in_leaf=300, learning_rate=0.02, lambda_l2=20.0),   # stumps
    dict(num_leaves=4, max_depth=2, min_data_in_leaf=200, learning_rate=0.01, lambda_l2=10.0),
    dict(num_leaves=8, max_depth=3, min_data_in_leaf=100, learning_rate=0.01, lambda_l2=10.0),
]
MAX_ROUNDS, PATIENCE = 3000, 200


def _lgb_train(params, X, y, off, rounds, Xva=None, yva=None, offva=None):
    dtr = lgb.Dataset(X, y, init_score=off, free_raw_data=False)
    if Xva is None:
        return lgb.train(params, dtr, num_boost_round=max(rounds, 1))
    dva = lgb.Dataset(Xva, yva, init_score=offva, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])


def fit_lgb(tr, X, y, off, objective="binary"):
    """Tune config + rounds on the inner split, then refit on all training rows.
    Returns (booster or None, info). None = zero rounds: the model IS the offset."""
    itr, iva = inner_split(tr)
    metric = "binary_logloss" if objective == "binary" else "l2"
    best = None
    for cfg in LGB_CONFIGS:
        p = dict(LGB_BASE, objective=objective, metric=metric, **cfg)
        b = _lgb_train(p, X[itr], y[itr], off[itr], MAX_ROUNDS, X[iva], y[iva], off[iva])
        score = b.best_score["valid_0"][metric]
        if best is None or score < best[0]:
            best = (score, cfg, b.best_iteration, b)
    # score of the pure offset on the inner validation set (zero rounds)
    if objective == "binary":
        base = float(ll_vec(y[iva], sigmoid(off[iva])).mean())
    else:
        base = float(((y[iva] - off[iva]) ** 2).mean())
    score, cfg, n_iter, inner_booster = best
    info = dict(cfg=cfg, n_iter=n_iter, inner_val=score, inner_base=base)
    p = dict(LGB_BASE, objective=objective, metric=metric, **cfg)
    final = _lgb_train(p, X, y, off, n_iter)
    return final, inner_booster, info


# ------------------------------------------------------------------------------------------- penalised logistic
def _fit_logreg(Z, y, off, lam, free_idx, pen_intercept=False):
    """min mean logloss(sigmoid(off + b0 + Z w)) + lam/2 * ||w_pen||^2; columns in free_idx unpenalised.
    pen_intercept=True also shrinks b0 to 0, so a large lam returns exactly the offset (the market)."""
    n, k = Z.shape
    pen = np.ones(k)
    pen[list(free_idx)] = 0.0

    def f(theta):
        b0, w = theta[0], theta[1:]
        eta = off + b0 + Z @ w
        p = sigmoid(eta)
        c0 = lam if pen_intercept else 0.0
        loss = np.mean(np.logaddexp(0, eta) - y * eta) + 0.5 * lam * np.sum(pen * w * w) + 0.5 * c0 * b0 * b0
        g = p - y
        grad = np.r_[g.mean() + c0 * b0, Z.T @ g / n + lam * pen * w]
        return loss, grad

    res = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B", options=dict(maxiter=2000))
    return res.x


class LogReg:
    """Logistic regression with an offset (the market logit) plus standardised covariates.
    free = names of covariates left unpenalised (e.g. the market logit itself = recalibration slope)."""
    LAMBDAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
    LAMBDAS_OFFSET = LAMBDAS + [3.0, 10.0, 30.0, 100.0]

    def __init__(self, free=(), offset_spec=False):
        """offset_spec=False: recalibration spec, logit p = b0 + (1+a) logit(mkt) + w.X with a, b0 free.
        offset_spec=True: logit p = logit(mkt) + b0 + w.X with EVERYTHING shrunk toward the market."""
        self.free = () if offset_spec else free
        self.pen0 = offset_spec
        self.lambdas = self.LAMBDAS_OFFSET if offset_spec else self.LAMBDAS

    def _prep(self, Z, fit=False):
        Z = Z.astype(float)
        if fit:
            self.cols = list(Z.columns)
            self.med = Z.median()
            Zf = Z.fillna(self.med)
            self.mu, self.sd = Zf.mean(), Zf.std().replace(0, 1).fillna(1)
        Z = Z[self.cols].fillna(self.med)
        return ((Z - self.mu) / self.sd).to_numpy()

    def fit(self, tr, Z, y, off):
        itr, iva = inner_split(tr)
        free_idx = [i for i, c in enumerate(Z.columns) if c in self.free]
        best = None
        for lam in self.lambdas:
            A = self._prep(Z[itr], fit=True)
            th = _fit_logreg(A, y[itr], off[itr], lam, free_idx, self.pen0)
            pv = sigmoid(off[iva] + th[0] + self._prep(Z[iva]) @ th[1:])
            s = ll_vec(y[iva], pv).mean()
            if best is None or s < best[0]:
                best = (s, lam)
        self.lam = best[1]
        self.inner_val = best[0]
        self.inner_base = float(ll_vec(y[iva], sigmoid(off[iva])).mean())
        A = self._prep(Z, fit=True)
        self.theta = _fit_logreg(A, y, off, self.lam, free_idx, self.pen0)
        return self

    def predict(self, Z, off):
        return sigmoid(off + self.theta[0] + self._prep(Z) @ self.theta[1:])

    def coefs(self):
        return pd.Series(self.theta[1:], index=self.cols)


def interaction_frame(d, X, extended=False, mode="close"):
    """Spec (c): market x rest, market x month, market x playoffs, market x favourite-at-home."""
    L = X.mkt_logit
    Z = pd.DataFrame(index=d.index)
    Z["mkt_logit"] = L  # unpenalised: recalibration slope
    early = d.month.isin([10, 11]).astype(int)
    late = (d.month.isin([3, 4]) & (d.playoffs == 0)).astype(int)
    fav_home = (L > 0).astype(int)
    rest = d.rest_diff.clip(-3, 3)
    mains = {"rest_diff": rest, "b2b_diff": d.b2b_diff, "early_season": early, "late_season": late,
             "playoffs": d.playoffs, "fav_home": fav_home}
    for k, v in mains.items():
        Z[k] = v
        Z[f"mkt_x_{k}"] = L * v
    if extended:
        Z["elo_gap"] = X.elo_gap
        for c in ("net_diff", "form_diff", "g5_diff", "diff_last10_net_rating", "diff_last5_net_rating",
                  "diff_streak", "diff_team_home_win_pct", "diff_last10_fg3_pct", "diff_season_win_pct",
                  "diff_last10_pace", "vig"):
            Z[c] = X[c]
        Z["min_games_played"] = np.minimum(d.home_games_played, d.away_games_played)
        if mode == "close":
            Z["spread_vs_ml"] = X.spread_vs_ml
            Z["line_move"] = X.line_move.fillna(0)
    return Z


class RidgeMove:
    """Ridge regression of the open->close line move (logit units) on pre-game features, alpha tuned
    on the inner split."""
    ALPHAS = [0.1, 1, 10, 100, 1000, 1e4, 1e5]

    def fit(self, tr, Z, t):
        from sklearn.linear_model import Ridge
        itr, iva = inner_split(tr)
        best = None
        for a in self.ALPHAS:
            self._stats(Z[itr])
            m = Ridge(alpha=a).fit(self._prep(Z[itr]), t[itr])
            s = float(((m.predict(self._prep(Z[iva])) - t[iva]) ** 2).mean())
            if best is None or s < best[0]:
                best = (s, a)
        self.alpha = best[1]
        self._stats(Z)
        self.m = Ridge(alpha=self.alpha).fit(self._prep(Z), t)
        return self

    def _stats(self, Z):
        self.med = Z.median()
        Zf = Z.fillna(self.med)
        self.mu, self.sd = Zf.mean(), Zf.std().replace(0, 1).fillna(1)

    def _prep(self, Z):
        return ((Z.fillna(self.med) - self.mu) / self.sd).to_numpy()

    def predict(self, Z):
        return self.m.predict(self._prep(Z))
