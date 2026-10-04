"""H2 - travel & schedule fatigue beyond simple rest days. End-to-end:  python research/h2_travel_schedule/run.py

1. build schedule features (team logs + arena coordinates) and the evaluation set (cached)
2. descriptive: does each feature predict the result / the margin on its own, and beyond the price?
3. walk-forward log-loss vs closing line (test 2022-23..2025-26) and opening line (test 2024-25, 2025-26)
   two specifications:  A = logit(p) = a + b*logit(mkt) + w.X   (protocol; market re-calibrated)
                        B = logit(p) = logit(mkt) + w.X         (market as fixed offset)
4. paired bootstrap CIs (95% and Bonferroni over all variants of both specs), per-season deltas
5. betting view: flat bets at the vig-inclusive closing (and opening) moneyline
6. supplementary: spread-residual (ATS) walk-forward, more statistical power than win/loss
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from h2data import build  # noqa: E402
from h2eval import (SINGLE, bet_view, boot_mean, ci, feature_matrix, ll_vec, logit, logit_mle,  # noqa: E402
                    ols_hc1, walk_forward, walk_forward_ats)

warnings.filterwarnings("ignore")
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
CACHE = HERE.parents[1] / "research/data/h2_travel_schedule"
pd.set_option("display.width", 250)

SPECS = {"A_recal": (False, list(SINGLE) + ["combined_l2", "combined_l1"]),
         "B_offset": (True, list(SINGLE) + ["combined_l2", "combined_l1", "combined_gbm"])}
N_VAR = sum(len(v) for _, v in SPECS.values())
BONF = 1 - 0.05 / N_VAR
CLOSE_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
OPEN_SEASONS = ["2024-25", "2025-26"]


def _std(X):
    return (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)


def _names(v, k):
    return [v] if k == 1 else [f"{v}[{s}]" for s in ("east", "west")]


def descriptive(d):
    """In-sample, pooled 2021-22..2025-26: per-SD effect of each single feature
    (a) alone on home win, (b) on top of logit(closing), (c) on margin, (d) on spread residual."""
    rows = []
    y = d.home_win.to_numpy(float)
    m = logit(d.mkt_close)
    hs = d.spread.notna().to_numpy()
    cover = (d.margin + d.spread).to_numpy(float)[hs]
    for v in SINGLE:
        raw = feature_matrix(d, v)
        X = _std(raw)
        r_alone, r_mkt = logit_mle(X, y), logit_mle(np.column_stack([m, X]), y)
        r_margin, r_cover = ols_hc1(X, d.margin.to_numpy(float)), ols_hc1(X[hs], cover)
        for j, nm in enumerate(_names(v, X.shape[1])):
            rows.append({"feature": nm, "nonzero_share": float((np.abs(raw[:, j]) > 1e-9).mean()),
                         "alone_coef_sd": r_alone.params[1 + j], "alone_z": r_alone.tvalues[1 + j],
                         "beyond_close_coef_sd": r_mkt.params[2 + j], "beyond_close_z": r_mkt.tvalues[2 + j],
                         "margin_pts_sd": r_margin.params[1 + j], "margin_t": r_margin.tvalues[1 + j],
                         "ats_pts_sd": r_cover.params[1 + j], "ats_t": r_cover.tvalues[1 + j]})
    return pd.DataFrame(rows)


def line_move(d):
    """Does the market move (open -> close, logit) in the direction of the feature? (2023-24+)"""
    s = d[d.mkt_open.notna()]
    mv = np.asarray(logit(s.mkt_close) - logit(s.mkt_open), float)
    rows = []
    for v in SINGLE:
        X = _std(feature_matrix(s, v))
        r = ols_hc1(X, mv)
        for j, nm in enumerate(_names(v, X.shape[1])):
            rows.append({"feature": nm, "move_logit_per_sd": r.params[1 + j], "t": r.tvalues[1 + j]})
    return pd.DataFrame(rows)


def score(te, mkt_col, p, test_seasons, need):
    y = te.home_win.to_numpy(float)
    delta = ll_vec(y, p) - ll_vec(y, te[mkt_col].to_numpy())
    bs = boot_mean(delta, 4000, seed=11)
    lo95, hi95 = ci(bs, 0.95)
    lob, hib = ci(bs, BONF)
    r = {"logloss": float(ll_vec(y, p).mean()), "delta": float(delta.mean()), "ci95_lo": lo95, "ci95_hi": hi95,
         "ciBonf_lo": lob, "ciBonf_hi": hib, "z": float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta))))}
    for S in test_seasons:
        r[S] = float(delta[(te.season == S).to_numpy()].mean())
    r["n_neg_seasons"] = int(sum(r[S] < 0 for S in test_seasons))
    if len(test_seasons) == 4:
        r["delta_2023_26"] = float(delta[(te.season >= "2023-24").to_numpy()].mean())
    r["survives"] = bool(hib < 0 and r["n_neg_seasons"] >= need)
    return r


def evaluate(d, mkt_col, test_seasons, label):
    t0 = time.time()
    need = 3 if len(test_seasons) == 4 else len(test_seasons)
    rows, preds, hps = [], {}, {}
    for spec, (offset, variants) in SPECS.items():
        vs = (["market_recal"] if spec == "A_recal" else []) + variants
        pr, hp, _ = walk_forward(d, mkt_col, test_seasons, vs, offset=offset)
        for v in vs:
            preds[(spec, v)] = pr[v]
            hps[(spec, v)] = hp[v]
    te = d.loc[preds[("A_recal", "market_recal")].index]
    base = ll_vec(te.home_win.to_numpy(float), te[mkt_col].to_numpy())
    for (spec, v), p in preds.items():
        r = score(te, mkt_col, p.loc[te.index].to_numpy(), test_seasons, need)
        if v == "market_recal":
            r["survives"] = None
        rows.append({"spec": spec, "variant": v, **r})
    res = pd.DataFrame(rows)
    print(f"\n=== {label}: market log-loss {base.mean():.5f} on {len(te)} games; per season: "
          + ", ".join(f"{S} {base[(te.season == S).to_numpy()].mean():.4f}" for S in test_seasons)
          + f"  [{time.time() - t0:.0f}s]")
    show = res.copy()
    for c in show.columns:
        if show[c].dtype == float and c != "z":
            show[c] = show[c].map(lambda x: f"{x:.5f}" if abs(x) > 0.1 else f"{x:+.5f}")
    show["z"] = res.z.round(2)
    print(show.to_string(index=False))
    print("selected hyper-parameters / weights (per SD) by test season:")
    for k in [("A_recal", "market_recal"), ("A_recal", "combined_l2"), ("A_recal", "combined_l1"),
              ("B_offset", "combined_l2"), ("B_offset", "combined_l1"), ("B_offset", "combined_gbm"),
              ("B_offset", "travel_km_diff"), ("B_offset", "tz_east_west"), ("B_offset", "home_return_from_trip")]:
        print(f"  {k[0]}/{k[1]}: {hps[k]}")
    rv = res[res.variant != "market_recal"]
    best = rv.loc[rv.delta.idxmin()]
    print(f"[{label}] {len(rv)} variants | survivors: {rv[rv.survives == True][['spec', 'variant']].values.tolist()} | "  # noqa: E712
          f"best {best.spec}/{best.variant} {best.delta:+.5f} (95% {best.ci95_lo:+.5f},{best.ci95_hi:+.5f}; "
          f"Bonf {best.ciBonf_lo:+.5f},{best.ciBonf_hi:+.5f}) | median {rv.delta.median():+.5f} | "
          f"variants with 95% CI < 0: {int((rv.ci95_hi < 0).sum())}")
    return res, preds, te


def betting(te, preds, model, odds):
    rows = []
    for (spec, v), p in preds.items():
        p = p.loc[te.index].to_numpy()
        for book, (dh, da) in odds.items():
            b = bet_view(te, p, te[dh], te[da])
            rows.append(b.assign(model=model, spec=spec, variant=v, odds=book))
    out = pd.concat(rows, ignore_index=True)
    for book in odds:
        o = out[out.odds == book]
        cell = o.apply(lambda r: "-" if r.bets == 0 else f"{r.roi:+.3f} [{r.lo:+.3f},{r.hi:+.3f}] n={int(r.bets)}",
                       axis=1)
        print(f"\n=== Betting: {model}-line models, flat 1u at {book.upper()} odds when "
              f"model prob > vig-inclusive implied prob + thr ===")
        print(o.assign(cell=cell).pivot_table(index=["spec", "variant"], columns="thr", values="cell",
                                              aggfunc="first").to_string())
        ok = o[o.bets >= 50]
        print(f"  rules with >=50 bets: {len(ok)} | ROI>0: {int((ok.roi > 0).sum())} | 95% CI lo>0: "
              f"{int((ok.lo > 0).sum())} | median ROI {ok.roi.median():+.3f}")
    return out


def ats(d):
    variants = list(SINGLE) + ["combined_l2"]
    pr = walk_forward_ats(d, CLOSE_SEASONS, variants)
    te = d.loc[pr[variants[0]].index]
    y = (te.margin + te.spread).to_numpy(float)
    bonf_ats = 1 - 0.05 / len(variants)
    rows = []
    for v in variants:
        p = pr[v].loc[te.index].to_numpy()
        delta = (y - p) ** 2 - y ** 2  # vs market's expectation of 0 cover margin
        bs = boot_mean(delta, 4000, seed=5)
        r = {"variant": v, "d_mse": delta.mean(), "ci95": ci(bs, 0.95), "ciBonf": ci(bs, bonf_ats)}
        for S in CLOSE_SEASONS:
            r[S] = delta[(te.season == S).to_numpy()].mean()
        for thr in (1.0, 2.0):
            pick = np.abs(p) >= thr
            res = np.sign(p[pick]) * y[pick]
            res = res[res != 0]  # pushes
            win = (res > 0).astype(float)
            prof = np.where(win == 1, 100 / 110, -1.0)  # standard -110 pricing (assumed)
            pb = boot_mean(prof, 2000, seed=3) if len(prof) else np.array([np.nan])
            r[f"thr{thr:.0f}_n"], r[f"thr{thr:.0f}_hit"] = len(win), win.mean() if len(win) else np.nan
            r[f"thr{thr:.0f}_roi"], r[f"thr{thr:.0f}_roi_ci"] = (prof.mean() if len(prof) else np.nan,
                                                               tuple(np.round(ci(pb, 0.95), 3)))
        rows.append(r)
    out = pd.DataFrame(rows)
    print(f"\n=== Supplementary: spread-residual (ATS) walk-forward, {len(te)} games; delta MSE (pts^2) vs "
          f"market's 0 (negative = better); picks at assumed -110 ===")
    sh = out.copy()
    for c in ["d_mse"] + CLOSE_SEASONS:
        sh[c] = sh[c].map(lambda x: f"{x:+.3f}")
    for c in ["ci95", "ciBonf"]:
        sh[c] = sh[c].map(lambda t: f"[{t[0]:+.3f},{t[1]:+.3f}]")
    print(sh.round(3).to_string(index=False))
    return out


def main():
    d = build()
    d = d[d.home_win.notna()].copy()
    print(f"games: {len(d)} | by season: {d.season.value_counts().sort_index().to_dict()}")
    num = [c for c in d.columns if c.startswith(("home_", "away_")) and d[c].dtype != object]
    assert d[num].isna().sum().sum() == 0, d[num].isna().sum()[lambda s: s > 0]

    print("\n=== Descriptive (in-sample, 2021-22..2025-26 pooled; per 1 SD of the feature) ===")
    desc = descriptive(d)
    print(desc.round(3).to_string(index=False))
    desc.to_csv(OUT / "descriptive.csv", index=False)
    print("\n=== Does the line move (open->close, logit) with the feature? (2023-24+) ===")
    lm = line_move(d)
    print(lm.round(4).to_string(index=False))
    lm.to_csv(OUT / "line_move.csv", index=False)

    print(f"\n{N_VAR} feature variants per baseline (spec A {len(SPECS['A_recal'][1])}, "
          f"spec B {len(SPECS['B_offset'][1])}) -> Bonferroni CI level {BONF:.4%}")
    rc, pc, tec = evaluate(d, "mkt_close", CLOSE_SEASONS, "CLOSING line baseline")
    rc.to_csv(OUT / "walkforward_close.csv", index=False)
    ro, po, teo = evaluate(d, "mkt_open", OPEN_SEASONS, "OPENING line baseline")
    ro.to_csv(OUT / "walkforward_open.csv", index=False)

    bc = betting(tec, pc, "closing", {"close": ("dec_home_close", "dec_away_close")})
    bo = betting(teo, po, "opening", {"open": ("dec_home_open", "dec_away_open"),
                                      "close": ("dec_home_close", "dec_away_close")})
    pd.concat([bc, bo]).to_csv(OUT / "betting.csv", index=False)
    pd.DataFrame({f"{s}/{v}": p for (s, v), p in pc.items()}).join(
        tec[["game_date", "season", "home", "away", "home_win", "mkt_close"]]).to_csv(CACHE / "preds_close.csv")

    a = ats(d)
    a.to_csv(OUT / "ats_walkforward.csv", index=False)

    import supplementary as sup
    print("\n=== Robustness 1: regular season only (offset spec, closing line) ===")
    sup.regular_season_only(d).to_csv(OUT / "regular_season_only.csv", index=False)
    print("\n=== Robustness 2: LOOK-AHEAD in-sample fit (deliberate leak) vs permutation null ===")
    d_ll, b, perm = sup.lookahead_upper_bound(d)
    b.assign(insample_delta_ll=d_ll).to_csv(OUT / "lookahead_real.csv", index=False)
    perm.to_csv(OUT / "lookahead_permuted.csv", index=False)
    print("\n=== Robustness 3: power simulation (outcomes simulated from closing line + known effect) ===")
    sup.power(d).to_csv(OUT / "power.csv", index=False)


if __name__ == "__main__":
    main()
