"""Models and evaluation for H2 (walk-forward by season, paired bootstrap, betting view).

Every model is   logit P(home win) = a + b * logit(market) + w . X
with a, b never penalised (so the market is only re-calibrated) and w optionally L2 / (smooth) L1
penalised with the strength chosen by inner cross-validation on the training seasons only.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ll_vec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ----------------------------------------------------------------------------- feature sets
def _d(df, col, scale=1.0):
    """away minus home (positive = away team more tired -> should help the home team)."""
    return (df[f"away_{col}"] - df[f"home_{col}"]) / scale


SINGLE = {
    "travel_km_diff": lambda d: _d(d, "travel_km", 1000),
    "tz_abs_diff": lambda d: d.away_tz_shift.abs() - d.home_tz_shift.abs(),
    "tz_east_west": lambda d: pd.DataFrame({
        "east": d.away_tz_shift.clip(lower=0) - d.home_tz_shift.clip(lower=0),
        "west": (-d.away_tz_shift).clip(lower=0) - (-d.home_tz_shift).clip(lower=0)}),
    "visiting_altitude": lambda d: _d(d, "visiting_altitude"),
    "after_altitude": lambda d: _d(d, "after_altitude"),
    "road_trip_len": lambda d: _d(d, "road_trip_len"),
    "home_stand_len": lambda d: d.home_home_stand_len - d.away_home_stand_len,
    "three_in_four": lambda d: _d(d, "three_in_four"),
    "four_in_six": lambda d: _d(d, "four_in_six"),
    "b2b_with_travel": lambda d: _d(d, "b2b_travel"),
    "games_last7": lambda d: _d(d, "games_last7"),
    "travel_7d": lambda d: _d(d, "travel_7d", 1000),
    "days_since_home": lambda d: _d(d, "days_since_home"),
    "home_return_from_trip": lambda d: d.home_return_from_trip.clip(upper=6),
    "body_clock_diff": lambda d: _d(d, "body_clock"),
}

PER_TEAM = ["travel_km", "tz_east", "tz_west", "visiting_altitude", "after_altitude", "road_trip_len",
            "home_stand_len", "three_in_four", "four_in_six", "b2b_travel", "b2b", "rest_days",
            "games_last7", "travel_7d", "days_since_home", "return_from_trip", "body_clock"]


def combined_matrix(d):
    out = {}
    for s in ("home", "away"):
        for c in PER_TEAM:
            if c == "tz_east":
                v = d[f"{s}_tz_shift"].clip(lower=0)
            elif c == "tz_west":
                v = (-d[f"{s}_tz_shift"]).clip(lower=0)
            else:
                v = d[f"{s}_{c}"]
            out[f"{s}_{c}"] = v.astype(float)
    return pd.DataFrame(out, index=d.index)


def feature_matrix(d, name):
    if name in SINGLE:
        x = SINGLE[name](d)
        return (x.to_frame(name) if isinstance(x, pd.Series) else x).astype(float).to_numpy()
    return combined_matrix(d).to_numpy()


# ----------------------------------------------------------------------------- plain inference helpers
class _Fit:
    def __init__(self, params, se):
        self.params, self.tvalues = params, params / se


def logit_mle(X, y):
    """Unpenalised logistic regression with intercept; Fisher-information standard errors."""
    Z = np.column_stack([np.ones(len(y)), X])

    def f(b):
        eta = Z @ b
        return -(y * eta - np.logaddexp(0, eta)).sum(), Z.T @ (expit(eta) - y)

    b = minimize(f, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 5000}).x
    p = expit(Z @ b)
    cov = np.linalg.inv(Z.T @ (Z * (p * (1 - p))[:, None]))
    return _Fit(b, np.sqrt(np.diag(cov)))


def ols_hc1(X, y):
    """OLS with intercept and HC1 heteroskedasticity-robust standard errors."""
    Z = np.column_stack([np.ones(len(y)), X])
    n, k = Z.shape
    ZtZi = np.linalg.inv(Z.T @ Z)
    b = ZtZi @ Z.T @ y
    e = y - Z @ b
    cov = ZtZi @ (Z.T @ (Z * (e ** 2)[:, None])) @ ZtZi * n / (n - k)
    return _Fit(b, np.sqrt(np.diag(cov)))


# ----------------------------------------------------------------------------- penalised logit
def fit_logit(m_logit, X, y, lam=0.0, kind="l2", offset=False):
    """Return coefficient vector [a, b, w...]; X already standardised.
    offset=False: a, b free (market re-calibrated) -- the protocol's 'logit(market) + features'.
    offset=True : a=0, b=1 fixed (market is an offset), only the feature weights w are fitted."""
    n = len(y)
    if offset:
        Z = np.asarray(X, float).reshape(n, -1)
        pen = np.full(Z.shape[1], lam)
        base = np.asarray(m_logit, float)
    else:
        Z = np.column_stack([np.ones(n), m_logit, X])
        pen = np.r_[0.0, 0.0, np.full(X.shape[1], lam)]
        base = np.zeros(n)

    def f(beta):
        eta = base + Z @ beta
        nll = -(y * eta - np.logaddexp(0, eta)).mean()
        g = Z.T @ (expit(eta) - y) / n
        if kind == "l2":
            return nll + (pen * beta ** 2).sum(), g + 2 * pen * beta
        sq = np.sqrt(beta ** 2 + 1e-6)  # smooth L1
        return nll + (pen * sq).sum(), g + pen * beta / sq

    if Z.shape[1] == 0:
        return np.r_[0.0, 1.0]
    x0 = np.zeros(Z.shape[1]) if offset else np.r_[0.0, 1.0, np.zeros(X.shape[1])]
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return np.r_[0.0, 1.0, r.x] if offset else r.x


def predict_logit(beta, m_logit, X):
    return expit(beta[0] + beta[1] * m_logit + X @ beta[2:])


def _standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def inner_folds(seasons_tr, dates_tr):
    """Leave-one-season-out if >= 2 training seasons, else 5 contiguous date blocks."""
    us = np.unique(seasons_tr)
    if len(us) >= 2:
        return [seasons_tr == s for s in us]
    order = np.argsort(dates_tr, kind="stable")
    blocks = np.array_split(order, 5)
    masks = []
    for b in blocks:
        m = np.zeros(len(dates_tr), bool)
        m[b] = True
        masks.append(m)
    return masks


LAMS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 10.0]


def fit_penalised_cv(m_tr, X_tr, y_tr, seasons_tr, dates_tr, kind, offset=False):
    folds = inner_folds(seasons_tr, dates_tr)
    cv = []
    for lam in LAMS:
        tot = 0.0
        for va in folds:
            Xa, Xb = _standardise(X_tr[~va], X_tr[va])
            beta = fit_logit(m_tr[~va], Xa, y_tr[~va], lam, kind, offset)
            tot += ll_vec(y_tr[va], predict_logit(beta, m_tr[va], Xb)).sum()
        cv.append(tot / len(y_tr))
    lam = LAMS[int(np.argmin(cv))]
    return lam


def fit_gbm_cv(m_tr, X_tr, y_tr, seasons_tr, dates_tr):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.01, num_leaves=4, min_data_in_leaf=150,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, seed=7, num_threads=4)
    grid = [0, 25, 50, 100, 200, 400]
    folds = inner_folds(seasons_tr, dates_tr)
    cv = np.zeros(len(grid))
    for va in folds:
        ds = lgb.Dataset(X_tr[~va], y_tr[~va], init_score=m_tr[~va], free_raw_data=False)
        bst = lgb.train(params, ds, num_boost_round=max(grid))
        for j, k in enumerate(grid):
            raw = m_tr[va] + (bst.predict(X_tr[va], num_iteration=k, raw_score=True) if k else 0)
            cv[j] += ll_vec(y_tr[va], expit(raw)).sum()
    k = grid[int(np.argmin(cv))]
    return params, k


# ----------------------------------------------------------------------------- walk-forward
def walk_forward(d, mkt_col, test_seasons, variants, offset=False):
    """Return {variant: per-game prediction Series indexed like test rows} + chosen hyper-params."""
    preds, hp = {v: [] for v in variants}, {v: [] for v in variants}
    test_idx = []
    for S in test_seasons:
        tr = d[(d.season < S) & d[mkt_col].notna()]
        te = d[(d.season == S) & d[mkt_col].notna()]
        test_idx.append(te.index)
        y_tr = tr.home_win.to_numpy(float)
        m_tr, m_te = logit(tr[mkt_col]), logit(te[mkt_col])
        for v in variants:
            if v == "market_recal":
                beta = fit_logit(m_tr, np.zeros((len(tr), 0)), y_tr)
                p = predict_logit(beta, m_te, np.zeros((len(te), 0)))
                hp[v].append(f"{S}: a={beta[0]:+.3f} b={beta[1]:.3f}")
            elif v in SINGLE:
                Xtr, Xte = _standardise(feature_matrix(tr, v), feature_matrix(te, v))
                beta = fit_logit(m_tr, Xtr, y_tr, offset=offset)
                p = predict_logit(beta, m_te, Xte)
                hp[v].append(f"{S}: w/sd={np.round(beta[2:], 4).tolist()}")
            elif v in ("combined_l2", "combined_l1"):
                kind = v[-2:]
                Xtr_raw, Xte_raw = feature_matrix(tr, v), feature_matrix(te, v)
                lam = fit_penalised_cv(m_tr, Xtr_raw, y_tr, tr.season.to_numpy(), tr.game_date.to_numpy(), kind,
                                       offset)
                Xtr, Xte = _standardise(Xtr_raw, Xte_raw)
                beta = fit_logit(m_tr, Xtr, y_tr, lam, kind, offset)
                p = predict_logit(beta, m_te, Xte)
                hp[v].append(f"{S}: lambda={lam} |w|max={np.abs(beta[2:]).max():.4f}")
            elif v == "combined_gbm":
                import lightgbm as lgb
                Xtr, Xte = feature_matrix(tr, v), feature_matrix(te, v)
                params, k = fit_gbm_cv(m_tr, Xtr, y_tr, tr.season.to_numpy(), tr.game_date.to_numpy())
                if k == 0:
                    p = expit(m_te)
                else:
                    bst = lgb.train(params, lgb.Dataset(Xtr, y_tr, init_score=m_tr), num_boost_round=k)
                    p = expit(m_te + bst.predict(Xte, raw_score=True))
                hp[v].append(f"{S}: rounds={k}")
            preds[v].append(pd.Series(p, index=te.index))
    return {v: pd.concat(preds[v]) for v in variants}, hp, test_idx


# ----------------------------------------------------------------------------- statistics
def boot_mean(x, n_boot=4000, seed=0, chunk=500):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    out = []
    for _ in range(n_boot // chunk):
        idx = rng.integers(0, len(x), size=(chunk, len(x)))
        out.append(x[idx].mean(1))
    return np.concatenate(out)


def ci(samples, level):
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def bet_view(d, p_home, dec_home, dec_away, thresholds=(0.0, 0.02, 0.04), n_boot=4000, seed=1):
    """Flat 1-unit bets where model prob exceeds the vig-inclusive implied prob (1/decimal) + thr."""
    rows = []
    y = d.home_win.to_numpy(float)
    ph = np.asarray(p_home, float)
    dh, da = np.asarray(dec_home, float), np.asarray(dec_away, float)
    ok = np.isfinite(dh) & np.isfinite(da)
    e_h, e_a = ph - 1 / dh, (1 - ph) - 1 / da
    for thr in thresholds:
        bet_h = ok & (e_h > thr) & (e_h >= e_a)
        bet_a = ok & (e_a > thr) & (e_a > e_h)
        prof = np.r_[np.where(y[bet_h] == 1, dh[bet_h] - 1, -1.0), np.where(y[bet_a] == 0, da[bet_a] - 1, -1.0)]
        if len(prof) == 0:
            rows.append({"thr": thr, "bets": 0, "roi": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        bs = boot_mean(prof, n_boot, seed, chunk=min(500, n_boot))
        lo, hi = ci(bs, 0.95)
        rows.append({"thr": thr, "bets": len(prof), "roi": prof.mean(), "lo": lo, "hi": hi,
                     "home_share": bet_h.sum() / len(prof)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- spread residual (ATS)
def walk_forward_ats(d, test_seasons, variants, lam_grid=(0.0, 1.0, 10.0, 100.0, 1000.0, 1e4)):
    """Predict the home cover margin (margin + closing spread; 0 = market's expectation) from the
    features with (ridge) OLS fitted on earlier seasons. Returns per-variant predictions."""
    out = {v: [] for v in variants}
    for S in test_seasons:
        tr = d[(d.season < S) & d.spread.notna()]
        te = d[(d.season == S) & d.spread.notna()]
        ytr = (tr.margin + tr.spread).to_numpy(float)
        for v in variants:
            Xtr, Xte = _standardise(feature_matrix(tr, v), feature_matrix(te, v))
            if v in SINGLE:
                lam = 0.0
            else:  # ridge strength by inner folds
                folds = inner_folds(tr.season.to_numpy(), tr.game_date.to_numpy())
                cv = []
                for lg in lam_grid:
                    e = 0.0
                    for va in folds:
                        A, B = _standardise(feature_matrix(tr[~va], v), feature_matrix(tr[va], v))
                        w = _ridge(A, ytr[~va], lg)
                        e += ((ytr[va] - (w[0] + B @ w[1:])) ** 2).sum()
                    cv.append(e)
                lam = lam_grid[int(np.argmin(cv))]
            w = _ridge(Xtr, ytr, lam)
            out[v].append(pd.Series(w[0] + Xte @ w[1:], index=te.index))
    return {v: pd.concat(p) for v, p in out.items()}


def _ridge(X, y, lam):
    Z = np.column_stack([np.ones(len(y)), X])
    P = max(lam, 1e-6) * np.eye(Z.shape[1])  # tiny floor: constant columns after standardising
    P[0, 0] = 0.0
    return np.linalg.solve(Z.T @ Z + P, Z.T @ y)
