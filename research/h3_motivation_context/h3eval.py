"""Models, game-level features and statistics for H3.

Every model is   logit P(home win) = [a + b*logit(mkt)] or [logit(mkt)]  +  w . X
  spec A ("recal", the protocol's 'logistic regression on logit(market) + features'): a, b free, unpenalised
  spec B ("offset"): a = 0, b = 1 fixed; only w is fitted -> games where every feature is 0 get exactly the
                     market price, so a sparse context feature can only change the games it describes.
Combined models: L2 penalty, strength by inner leave-one-season-out CV on the training seasons only.
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
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ----------------------------------------------------------------------------- game-level features
RS_SIDE = ["elim_post", "seed_locked", "tank", "in_race", "lw_no_stakes", "last_game", "post_margin", "race_close",
           "lottery"]
RS_SINGLE = ["elim_diff", "locked_diff", "nostakes_diff", "tank_diff", "race_diff", "lw_nostakes_diff"]
PO_SINGLE = ["po_fav", "po_slope", "zigzag", "elim_po", "game7", "series_lead", "down02"]
PO_ALL = PO_SINGLE + ["prev_margin"]


def game_features(d, mkt_col):
    """All candidate columns, 0 where they do not apply. Home-minus-away convention for team flags."""
    rs = (d.season_type == "Regular Season").to_numpy(float)
    post = 1 - rs
    po = (d.season_type == "Playoffs").to_numpy(float)
    m = logit(d[mkt_col])
    f = {}
    lw = d.home_last_week.fillna(0).to_numpy(float)
    for side in ("home", "away"):
        for c in ("elim_post", "seed_locked", "tank", "in_race", "last_game", "no_stakes"):
            f[f"{side}_{c}"] = rs * d[f"{side}_{c}"].fillna(0).to_numpy(float)
        f[f"{side}_lw_no_stakes"] = lw * f[f"{side}_no_stakes"]
        # continuous late-season pressure (only with <= 25 games left): games above(+)/below(-) the
        # postseason line, closeness to the nearest still-live seed boundary, lottery position
        late = rs * (d[f"{side}_rem"].fillna(99).to_numpy(float) <= 25)
        post_b = np.where(d.season.isin(["2018-19", "2019-20"]), d[f"{side}_margin_8"], d[f"{side}_margin_10"])
        f[f"{side}_post_margin"] = late * np.clip(np.nan_to_num(post_b.astype(float)), -8, 8) / 8
        best, worst = d[f"{side}_best_seed"].to_numpy(float), d[f"{side}_worst_seed"].to_numpy(float)
        near = np.full(len(d), 99.0)
        for b in (1, 4, 6, 8, 10):
            live = (best <= b) & (worst > b)
            near = np.where(live, np.minimum(near, np.abs(d[f"{side}_margin_{b}"].to_numpy(float))), near)
        f[f"{side}_race_close"] = late * np.clip(3 - np.nan_to_num(near, nan=99.0), 0, 3) / 3
        f[f"{side}_lottery"] = late * np.clip(7 - d[f"{side}_league_bottom_rank"].fillna(30).to_numpy(float), 0, 6) / 6
    f["elim_diff"] = f["home_elim_post"] - f["away_elim_post"]
    f["locked_diff"] = f["home_seed_locked"] - f["away_seed_locked"]
    f["nostakes_diff"] = f["home_no_stakes"] - f["away_no_stakes"]
    f["tank_diff"] = f["home_tank"] - f["away_tank"]
    f["race_diff"] = f["home_in_race"] - f["away_in_race"]
    f["lw_nostakes_diff"] = f["home_lw_no_stakes"] - f["away_lw_no_stakes"]
    f["po_fav"] = post * np.sign(m)
    f["po_slope"] = post * m
    g = lambda c: d[c].fillna(0).to_numpy(float)  # noqa: E731
    f["zigzag"] = po * g("home_lost_prev")
    f["elim_po"] = post * (g("home_elim") - g("away_elim"))
    f["game7"] = po * g("game7")
    f["series_lead"] = po * (g("home_series_w") - g("away_series_w"))
    f["down02"] = po * g("home_down02")
    f["prev_margin"] = po * g("prev_margin_home") / 10
    return pd.DataFrame(f, index=d.index)


RS_COMB = [f"{s}_{c}" for s in ("home", "away") for c in RS_SIDE]
VARIANTS = {**{v: [v] for v in RS_SINGLE}, "rs_combined_l2": RS_COMB,
            **{v: [v] for v in PO_SINGLE}, "po_combined_l2": PO_ALL,
            "all_combined_l2": RS_COMB + PO_ALL, "all_combined_gbm": RS_COMB + PO_ALL}
FAMILY = {**{v: "regular season" for v in RS_SINGLE + ["rs_combined_l2"]},
          **{v: "playoffs" for v in PO_SINGLE + ["po_combined_l2"]},
          "all_combined_l2": "all", "all_combined_gbm": "all"}


# ----------------------------------------------------------------------------- fitting
def fit_logit(m, X, y, lam=0.0, offset=True):
    n = len(y)
    if offset:
        Z, base, pen = X, m, np.full(X.shape[1], lam)
        x0 = np.zeros(X.shape[1])
    else:
        Z, base = np.column_stack([np.ones(n), m, X]), np.zeros(n)
        pen = np.r_[0.0, 0.0, np.full(X.shape[1], lam)]
        x0 = np.r_[0.0, 1.0, np.zeros(X.shape[1])]
    if Z.shape[1] == 0:
        return np.r_[0.0, 1.0]

    def f(b):
        eta = base + Z @ b
        nll = -(y * eta - np.logaddexp(0, eta)).mean()
        return nll + (pen * b ** 2).sum(), Z.T @ (expit(eta) - y) / n + 2 * pen * b

    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
    return np.r_[0.0, 1.0, r.x] if offset else r.x


def predict(beta, m, X):
    return expit(beta[0] + beta[1] * m + X @ beta[2:])


def _scale(Xtr, Xte):
    """Scale by the training SD of the non-zero entries' magnitude (keeps exact zeros at zero, so a
    sparse feature never moves games it does not describe)."""
    sd = np.array([np.std(c[c != 0]) if (c != 0).sum() > 1 else 1.0 for c in Xtr.T])
    mag = np.array([np.abs(c[c != 0]).mean() if (c != 0).any() else 1.0 for c in Xtr.T])
    s = np.where(sd > 1e-9, sd, np.where(mag > 0, mag, 1.0))
    return Xtr / s, Xte / s


LAMS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]


def cv_lambda(m, X, y, seasons, offset):
    us = np.unique(seasons)
    if len(us) < 2:
        return 1e-3
    cv = []
    for lam in LAMS:
        tot = 0.0
        for s in us:
            va = seasons == s
            A, B = _scale(X[~va], X[va])
            beta = fit_logit(m[~va], A, y[~va], lam, offset)
            tot += ll_vec(y[va], predict(beta, m[va], B)).sum()
        cv.append(tot)
    return LAMS[int(np.argmin(cv))]


GBM_PARAMS = dict(objective="binary", learning_rate=0.01, num_leaves=4, min_data_in_leaf=60,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, seed=7, num_threads=4)
GBM_GRID = [0, 25, 50, 100, 200, 400]


def gbm_fit_predict(m_tr, X_tr, y_tr, seasons, m_te, X_te):
    import lightgbm as lgb
    us = np.unique(seasons)
    cv = np.zeros(len(GBM_GRID))
    for s in us if len(us) >= 2 else []:
        va = seasons == s
        bst = lgb.train(GBM_PARAMS, lgb.Dataset(X_tr[~va], y_tr[~va], init_score=m_tr[~va]),
                        num_boost_round=max(GBM_GRID))
        for j, k in enumerate(GBM_GRID):
            raw = m_tr[va] + (bst.predict(X_tr[va], num_iteration=k, raw_score=True) if k else 0)
            cv[j] += ll_vec(y_tr[va], expit(raw)).sum()
    k = GBM_GRID[int(np.argmin(cv))] if len(us) >= 2 else 0
    if k == 0:
        return expit(m_te), k
    bst = lgb.train(GBM_PARAMS, lgb.Dataset(X_tr, y_tr, init_score=m_tr), num_boost_round=k)
    return expit(m_te + bst.predict(X_te, raw_score=True)), k


def walk_forward(d, mkt_col, test_seasons, offset, first_train=None, variants=None):
    """Per-variant out-of-sample predictions for the test seasons, plus fitted weights per season."""
    variants = variants or [v for v in VARIANTS if offset or not v.endswith("gbm")]
    d = d[d[mkt_col].notna()]
    F = game_features(d, mkt_col)
    preds = {v: [] for v in ["market_recal"] + variants}
    info = {v: [] for v in preds}
    for S in test_seasons:
        trm = (d.season < S) & ((d.season >= first_train) if first_train else True)
        tr, te = d[trm], d[d.season == S]
        y = tr.home_win.to_numpy(float)
        m_tr, m_te = logit(tr[mkt_col]), logit(te[mkt_col])
        if not offset:
            beta = fit_logit(m_tr, np.zeros((len(tr), 0)), y, offset=False)
            preds["market_recal"].append(pd.Series(predict(beta, m_te, np.zeros((len(te), 0))), te.index))
            info["market_recal"].append(f"{S}: a={beta[0]:+.3f} b={beta[1]:.3f}")
        for v in variants:
            cols = VARIANTS[v]
            Xtr_raw, Xte_raw = F.loc[tr.index, cols].to_numpy(), F.loc[te.index, cols].to_numpy()
            if v.endswith("gbm"):
                p, k = gbm_fit_predict(m_tr, Xtr_raw, y, tr.season.to_numpy(), m_te, Xte_raw)
                info[v].append(f"{S}: rounds={k}")
            else:
                lam = cv_lambda(m_tr, Xtr_raw, y, tr.season.to_numpy(), offset) if len(cols) > 1 else 0.0
                Xtr, Xte = _scale(Xtr_raw, Xte_raw)
                beta = fit_logit(m_tr, Xtr, y, lam, offset)
                p = predict(beta, m_te, Xte)
                w = beta[2:]
                info[v].append(f"{S}: lam={lam} w={np.round(w, 3).tolist() if len(w) <= 3 else np.round(w, 2).tolist()}")
            preds[v].append(pd.Series(p, te.index))
    out = {v: pd.concat(p) for v, p in preds.items() if p}
    return out, info, F


# ----------------------------------------------------------------------------- statistics
def boot_mean(x, n_boot=4000, seed=0, chunk=500):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    out = []
    for _ in range(int(np.ceil(n_boot / chunk))):
        idx = rng.integers(0, len(x), size=(chunk, len(x)))
        out.append(x[idx].mean(1))
    return np.concatenate(out)[:n_boot]


def boot_cluster_mean(x, clusters, n_boot=4000, seed=0):
    """Bootstrap of the mean of x resampling whole clusters (e.g. playoff series)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    codes, uniq = pd.factorize(pd.Series(clusters).astype(str))
    sums = np.bincount(codes, weights=x)
    cnts = np.bincount(codes).astype(float)
    k = len(sums)
    idx = rng.integers(0, k, size=(n_boot, k))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def ci(samples, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def bet_profits(y_home, p_home, dec_home, dec_away, thr):
    """Flat 1u at vig-inclusive odds when model prob - 1/decimal > thr (better side if both)."""
    y, ph = np.asarray(y_home, float), np.asarray(p_home, float)
    dh, da = np.asarray(dec_home, float), np.asarray(dec_away, float)
    ok = np.isfinite(dh) & np.isfinite(da) & np.isfinite(ph)
    eh, ea = ph - 1 / dh, (1 - ph) - 1 / da
    bh = ok & (eh > thr) & (eh >= ea)
    ba = ok & (ea > thr) & (ea > eh)
    prof = np.full(len(y), np.nan)
    prof[bh] = np.where(y[bh] == 1, dh[bh] - 1, -1.0)
    prof[ba] = np.where(y[ba] == 0, da[ba] - 1, -1.0)
    return prof  # NaN = no bet


def side_profit(y_home, side, dec_home, dec_away):
    """Profit of 1u on the side given by `side` (+1 home, -1 away)."""
    y = np.asarray(y_home, float)
    side = np.asarray(side, float)
    won = np.where(side > 0, y == 1, y == 0)
    dec = np.where(side > 0, dec_home, dec_away)
    return np.where(won, dec - 1, -1.0)
