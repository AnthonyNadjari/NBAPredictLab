"""(c) Moneyline vs spread inconsistency.

The spread is mapped to a win probability walk-forward (logistic on the juice-adjusted
spread, optionally interacted with the total: margin sd grows with scoring level). The
question is whether the ML-vs-spread gap predicts outcomes beyond the closing moneyline.
"""
import numpy as np
import pandas as pd

from common import TEST_SEASONS, am_to_p, bet_roi, compare, logit, sigmoid, walk_forward

K_PTS = 12.5 * np.sqrt(2 * np.pi)  # points per unit of cover probability around the spread


def add_spread_cols(g, books):
    g = g.copy()
    # juice of the main book's spread (same book as the ML), from the full re-fetch
    j = pd.DataFrame(columns=["event_id", "book", "h_so", "a_so"])
    if books is not None and "h_so" in books.columns:
        j = books[["event_id", "book", "h_so", "a_so"]]
    g = g.merge(j, on=["event_id", "book"], how="left")
    ph, pa = am_to_p(g.h_so), am_to_p(g.a_so)
    pc = ph / (ph + pa)
    ok = np.isfinite(pc) & (ph + pa > 1.0) & (ph + pa < 1.12)
    g["p_cover_home"] = np.where(ok, pc, 0.5)
    g["juice_known"] = ok
    g["spread_eff"] = g.spread - (g.p_cover_home - 0.5) * K_PTS
    g["tot_c"] = (g.total - 225) / 10
    return g


def run(g, books):
    out = {"logloss": [], "bets": [], "tables": {}}
    g = add_spread_cols(g, books)
    g = g[g.spread.notna() & g.total.notna()].reset_index(drop=True)
    t = g[g.season.isin(TEST_SEASONS)]
    out["tables"]["c_juice_coverage"] = g.groupby("season").juice_known.mean().reset_index()

    # spread-only price (fitted mapping), compared with the ML close
    sp_only = pd.Series(np.nan, index=g.index)
    from sklearn.linear_model import LogisticRegression
    for s in TEST_SEASONS:
        tr, te = g.season < s, g.season == s
        X = np.c_[-g.spread_eff, -g.spread_eff * g.tot_c]
        m = LogisticRegression(C=1e4, max_iter=5000).fit(X[tr], g.home_win[tr])
        sp_only[te] = m.predict_proba(X[te])[:, 1]
    out["logloss"].append({"part": "c", **compare(t, sp_only[t.index].to_numpy(), t.p_close.to_numpy(),
                                                   "spread_only_price (fitted map)")})
    g["gap"] = logit(sp_only.fillna(0.5)) - logit(g.p_close)
    variants = {
        "ml+spread_raw": lambda d: -d.spread,
        "ml+spread_eff": lambda d: -d.spread_eff,
        "ml+spread_eff+spread_x_total": lambda d: np.c_[-d.spread_eff, -d.spread_eff * d.tot_c],
        "ml+spread_eff_x_|ml|": lambda d: np.c_[-d.spread_eff, -d.spread_eff * np.abs(logit(d.p_close))],
    }
    platt = walk_forward(g, None)
    preds = {}
    for name, fx in variants.items():
        p = walk_forward(g, fx)
        preds[name] = p
        r = compare(t, p[t.index].to_numpy(), t.p_close.to_numpy(), name)
        r["delta_vs_platt"] = compare(t, p[t.index].to_numpy(), platt[t.index].to_numpy(), name)["delta"]
        out["logloss"].append({"part": "c", **r})
    for th in (0.0, 0.02, 0.04):
        out["bets"].append({"part": "c", **bet_roi(t, preds["ml+spread_eff+spread_x_total"][t.index].to_numpy(),
                                                     th, label=f"ML+spread model edge>{th}")})
    # direct rule: the side whose ML is cheaper than its spread-implied probability
    tt = t.copy()
    tt["gap"] = logit(sp_only[t.index]) - logit(t.p_close)
    for th in (0.10, 0.20, 0.30):
        # gap > th: spread says home stronger than ML does -> bet home ML (and vice versa)
        p_rule = np.where(tt.gap > th, 0.999, np.where(tt.gap < -th, 0.001, np.nan))
        out["bets"].append({"part": "c", **bet_roi(tt, p_rule, -1.0,
                                                     label=f"bet ML side favoured by spread, |gap|>{th} logit")})
    q = tt.assign(gap_bin=pd.qcut(tt.gap, 5))
    q["fav_home"] = q.p_close >= .5
    out["tables"]["c_gap_quintiles"] = q.groupby("gap_bin", observed=True).agg(
        n=("home_win", "size"), mean_gap=("gap", "mean"), p_ml=("p_close", "mean"),
        p_spread=("gap", lambda s: float(sigmoid(logit(tt.loc[s.index, "p_close"]) + s).mean())),
        home_won=("home_win", "mean")).reset_index().assign(gap_bin=lambda x: x.gap_bin.astype(str))
    return out
