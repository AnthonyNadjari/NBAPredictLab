"""Robustness checks for H2 (called from run.py; not part of the survival bar).

1. Power: inject a KNOWN effect beta (logit per SD) of a feature into simulated outcomes drawn from the
   closing line, rerun the same walk-forward (offset spec) and count how often the effect is detected
   (95% CI excluding 0 and >=3/4 seasons negative). Tells us whether "no edge" means "none" or "too small
   to see".
2. Look-ahead fit: fit the combined offset model IN-SAMPLE on the test seasons themselves
   (deliberate leakage) and bet with it at closing odds, then repeat with the feature rows randomly
   permuted across games (which destroys any real signal). If the real in-sample "edge" is no bigger
   than the permuted ones, the in-sample edge is pure overfitting.
3. Regular season only: rerun the offset spec without play-in / playoffs (playoff travel is a
   different regime: two cities, 2-2-1-1-1).
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from h2eval import (SINGLE, _standardise, bet_view, feature_matrix, fit_logit, ll_vec, logit, predict_logit,
                    walk_forward)

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


def power(d, features=("travel_km_diff", "tz_abs_diff", "home_return_from_trip", "three_in_four"),
          betas=(0.05, 0.10, 0.15), n_sim=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    m = logit(d.mkt_close)
    for f in features:
        X = feature_matrix(d, f)
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
        if X.shape[1] > 1:  # combined: spread the effect evenly over a random direction of unit SD
            w = rng.normal(size=X.shape[1])
            z = X @ w
            x_eff = (z - z.mean()) / z.std()
        else:
            x_eff = X[:, 0]
        for b in betas:
            hits, deltas = 0, []
            for _ in range(n_sim):
                ds = d.copy()
                p_true = expit(m + b * x_eff)
                ds["home_win"] = (rng.random(len(ds)) < p_true).astype(float)
                pr, _, _ = walk_forward(ds, "mkt_close", SEASONS, [f], offset=True)
                te = ds.loc[pr[f].index]
                delta = ll_vec(te.home_win.to_numpy(), pr[f].to_numpy()) - ll_vec(te.home_win.to_numpy(),
                                                                                    te.mkt_close.to_numpy())
                se = delta.std(ddof=1) / np.sqrt(len(delta))
                per = [delta[(te.season == S).to_numpy()].mean() for S in SEASONS]
                hits += int(delta.mean() + 1.96 * se < 0 and sum(x < 0 for x in per) >= 3)
                deltas.append(delta.mean())
            # implied probability shift of a 1-SD feature at a 50/50 game
            rows.append({"feature": f, "beta_per_sd": b, "pp_at_50pct_per_sd": 100 * (expit(b) - 0.5),
                         "power": hits / n_sim, "mean_delta": float(np.mean(deltas))})
            print(f"  power {f:22s} beta={b:.2f} (~{100 * (expit(b) - .5):.1f} pp/SD): "
                  f"detected {hits}/{n_sim}, mean delta {np.mean(deltas):+.5f}", flush=True)
    return pd.DataFrame(rows)


def lookahead_upper_bound(d, n_perm=40, seed=0):
    te = d[d.season.isin(SEASONS)]
    y = te.home_win.to_numpy(float)
    m = logit(te.mkt_close)
    Xs, _ = _standardise(feature_matrix(te, "combined_l2"), feature_matrix(te, "combined_l2"))
    base_ll = ll_vec(y, te.mkt_close.to_numpy()).mean()

    def fit_eval(X):
        beta = fit_logit(m, X, y, lam=0.0, offset=True)
        p = predict_logit(beta, m, X)
        b = bet_view(te, p, te.dec_home_close, te.dec_away_close, thresholds=(0.0, 0.02, 0.04), n_boot=500)
        return float(ll_vec(y, p).mean() - base_ll), b

    d_ll, b = fit_eval(Xs)
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        dl, bp = fit_eval(Xs[rng.permutation(len(Xs))])
        perm.append({"d_ll": dl, **{f"roi_{t:.2f}": r for t, r in zip(bp.thr, bp.roi)}})
    perm = pd.DataFrame(perm)
    print(f"  REAL features, in-sample: log-loss delta {d_ll:+.5f}")
    print(b.round(3).to_string(index=False))
    print(f"  PERMUTED features ({n_perm}x), in-sample: log-loss delta mean {perm.d_ll.mean():+.5f} "
          f"(min {perm.d_ll.min():+.5f}); share of permutations at least as good as real: "
          f"{(perm.d_ll <= d_ll).mean():.2f}")
    for c in [c for c in perm.columns if c.startswith("roi")]:
        real = float(b.roi[np.isclose(b.thr, float(c[4:]))].iloc[0])
        print(f"    {c}: permuted mean {perm[c].mean():+.3f}, 90th pct {perm[c].quantile(.9):+.3f}; real {real:+.3f}"
              f" -> share of permutations >= real: {(perm[c] >= real).mean():.2f}")
    return d_ll, b, perm


def regular_season_only(d):
    r = d[d.season_type == "Regular Season"]
    variants = list(SINGLE) + ["combined_l2", "combined_l1"]
    pr, _, _ = walk_forward(r, "mkt_close", SEASONS, variants, offset=True)
    te = r.loc[pr[variants[0]].index]
    y = te.home_win.to_numpy(float)
    base = ll_vec(y, te.mkt_close.to_numpy())
    rows = []
    for v in variants:
        delta = ll_vec(y, pr[v].loc[te.index].to_numpy()) - base
        se = delta.std(ddof=1) / np.sqrt(len(delta))
        rows.append({"variant": v, "delta": delta.mean(), "ci95_lo": delta.mean() - 1.96 * se,
                     "ci95_hi": delta.mean() + 1.96 * se,
                     **{S: delta[(te.season == S).to_numpy()].mean() for S in SEASONS}})
    out = pd.DataFrame(rows)
    print(f"  regular season only, offset spec, {len(te)} games, market LL {base.mean():.5f}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    return out
