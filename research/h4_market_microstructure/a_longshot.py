"""(a) Favourite-longshot bias: calibration by price band, alternative de-vig methods,
walk-forward Platt / isotonic recalibration, ROI by price band at the closing price."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from common import (TEST_SEASONS, am_to_dec, am_to_p, compare, bet_roi, logit, pnl_roi,
                    walk_forward)


# ------------------------------------------------------------------ de-vig methods (no fitting)
def devig_power(qh, qa):
    lo, hi = np.ones_like(qh), np.full_like(qh, 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        s = qh ** k + qa ** k
        lo, hi = np.where(s > 1, k, lo), np.where(s > 1, hi, k)
    k = (lo + hi) / 2
    return qh ** k / (qh ** k + qa ** k)


def devig_shin(qh, qa):
    Q = qh + qa

    def probs(z):
        f = lambda q: (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / Q) - z) / (2 * (1 - z))
        return f(qh), f(qa)

    lo, hi = np.zeros_like(qh), np.full_like(qh, 0.4)
    for _ in range(60):
        z = (lo + hi) / 2
        ph, pa = probs(z)
        s = ph + pa
        lo, hi = np.where(s > 1, z, lo), np.where(s > 1, hi, z)
    ph, pa = probs((lo + hi) / 2)
    return ph / (ph + pa)


def devig_additive(qh, qa):
    return np.clip(qh - (qh + qa - 1) / 2, 1e-4, 1 - 1e-4)


# ------------------------------------------------------------------ tables
def calibration_table(g):
    fav_home = g.p_close >= .5
    fp = np.where(fav_home, g.p_close, 1 - g.p_close)
    fw = np.where(fav_home, g.home_win, 1 - g.home_win)
    t = pd.DataFrame({"fav_p": fp, "fav_won": fw, "season": g.season})
    t["band"] = pd.cut(t.fav_p, [.5, .55, .6, .65, .7, .75, .8, .85, .9, 1.0], include_lowest=True)
    r = t.groupby("band", observed=True).agg(n=("fav_won", "size"), priced=("fav_p", "mean"),
                                             won=("fav_won", "mean"))
    r["gap_pts"] = 100 * (r.won - r.priced)
    r["z"] = (r.won - r.priced) / np.sqrt(r.priced * (1 - r.priced) / r.n)
    return r.reset_index().assign(band=lambda x: x.band.astype(str))


def roi_by_band(g, seasons=None):
    """Bet every team (home and away rows) in a de-vigged price band at the closing ML."""
    if seasons is not None:
        g = g[g.season.isin(seasons)]
    p = np.r_[g.p_close, 1 - g.p_close]
    dec = np.r_[am_to_dec(g.h_close), am_to_dec(g.a_close)]
    won = np.r_[g.home_win, 1 - g.home_win]
    ss = np.r_[g.season, g.season]
    pnl = np.where(won == 1, dec - 1, -1.0)
    bands = [(0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, 1.01)]
    rows = []
    for lo, hi in bands:
        m = (p >= lo) & (p < hi) & np.isfinite(dec)
        r = pnl_roi(pnl[m], ss[m], label=f"all teams priced {lo:.2f}-{min(hi, 1):.2f}")
        r["win_rate"], r["priced"] = float(won[m].mean()), float(p[m].mean())
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def run(g):
    out = {"logloss": [], "bets": [], "tables": {}}
    out["tables"]["a_calibration_close"] = calibration_table(g)
    out["tables"]["a_roi_by_band_all_seasons"] = roi_by_band(g)
    out["tables"]["a_roi_by_band_test_seasons"] = roi_by_band(g, TEST_SEASONS)

    t = g[g.season.isin(TEST_SEASONS)]
    qh, qa = am_to_p(g.h_close), am_to_p(g.a_close)
    for name, fn in [("devig_additive", devig_additive), ("devig_power", devig_power),
                     ("devig_shin", devig_shin)]:
        p = pd.Series(fn(qh, qa), index=g.index)
        out["logloss"].append({"part": "a", **compare(t, p[t.index].to_numpy(),
                                                       t.p_close.to_numpy(), name)})

    platt = walk_forward(g, None)
    out["platt"] = platt
    out["logloss"].append({"part": "a", **compare(t, platt[t.index].to_numpy(),
                                                   t.p_close.to_numpy(), "platt_recalibration")})
    # Platt + asymmetric term (FLB may differ for home vs away favourites)
    asym = walk_forward(g, lambda d: np.abs(logit(d.p_close)))
    out["logloss"].append({"part": "a", **compare(t, asym[t.index].to_numpy(),
                                                   t.p_close.to_numpy(), "platt_plus_|logit|")})
    # does the size of the overround carry information (books widen margins when unsure)?
    vg = walk_forward(g, lambda d: np.c_[d.vig_close * 100, d.vig_close * 100 * logit(d.p_close)])
    r = compare(t, vg[t.index].to_numpy(), t.p_close.to_numpy(), "platt_plus_vig_interaction")
    r["delta_vs_platt"] = compare(t, vg[t.index].to_numpy(), platt[t.index].to_numpy(), "x")["delta"]
    out["logloss"].append({"part": "a", **r})
    # isotonic, walk-forward
    iso = pd.Series(np.nan, index=g.index)
    for s in TEST_SEASONS:
        tr, te = g.season < s, g.season == s
        m = IsotonicRegression(y_min=.01, y_max=.99, out_of_bounds="clip").fit(g.p_close[tr], g.home_win[tr])
        iso[te] = m.predict(g.p_close[te])
    out["logloss"].append({"part": "a", **compare(t, iso[t.index].to_numpy(),
                                                   t.p_close.to_numpy(), "isotonic_recalibration")})
    # fitted Platt coefficients per fold (interpretation: slope > 1 => favourites underpriced)
    from sklearn.linear_model import LogisticRegression
    coefs = []
    for s in TEST_SEASONS:
        tr = g.season < s
        m = LogisticRegression(C=1e4, max_iter=5000).fit(logit(g.p_close[tr]).reshape(-1, 1), g.home_win[tr])
        coefs.append({"test_season": s, "intercept": m.intercept_[0], "slope": m.coef_[0, 0]})
    out["tables"]["a_platt_coefs"] = pd.DataFrame(coefs)

    for name, p in [("platt", platt), ("isotonic", iso), ("shin", pd.Series(devig_shin(qh, qa), index=g.index))]:
        for th in (0.0, 0.02, 0.04):
            out["bets"].append({"part": "a", **bet_roi(t, p[t.index].to_numpy(), th, label=f"{name} edge>{th}")})
    # simple FLB rules at the closing ML, test seasons
    fav_home = t.p_close >= .5
    for lo in (.75, .85):
        m_h = fav_home & (t.p_close >= lo)
        m_a = ~fav_home & (1 - t.p_close >= lo)
        pnl = np.r_[np.where(t.home_win[m_h] == 1, am_to_dec(t.h_close[m_h]) - 1, -1.0),
                    np.where(t.home_win[m_a] == 0, am_to_dec(t.a_close[m_a]) - 1, -1.0)]
        ss = np.r_[t.season[m_h], t.season[m_a]]
        out["bets"].append({"part": "a", **pnl_roi(pnl, ss, label=f"all favourites >= {lo:.0%}")})
    return out
