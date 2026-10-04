"""H5: can a flexible ML model learn residual structure the betting market misses?

python research/h5_ml_residual/run.py            # everything (about 6 minutes, no network)
python research/h5_ml_residual/run.py --no-power # skip the synthetic-signal power check

Walk-forward by season. For each test season S, every model is fitted (and tuned on an inner
time-based split) on seasons < S only, then predicts S. The market logit is an input/offset of every
candidate, so the question is "does anything add information BEYOND the price".
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h5data  # noqa: E402
from h5models import (PROBIT_TO_LOGIT, LogReg, RidgeMove, fit_lgb, interaction_frame, ll_vec, logit,  # noqa: E402
                      make_features, offset_of, sigmoid, inner_split)

RES = HERE / "results"
RES.mkdir(exist_ok=True)
CLOSE_TEST = ["2022-23", "2023-24", "2024-25", "2025-26"]
OPEN_TEST = ["2024-25", "2025-26"]  # opening odds start in 2023-24
MARKET_COLS = ["mkt_logit", "vig", "spread", "total", "spread_missing", "spread_vs_ml", "line_move", "elo_gap"]
B_LL, B_BET = 10000, 10000
THRESHOLDS = [0.0, 0.02, 0.04, 0.06]


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


# ------------------------------------------------------------------------------------------- one fold
def fold_predictions(d, tr_mask, te_mask, mode, models, fold_log, imp_log, coef_log, tag):
    tr, te = d[tr_mask], d[te_mask]
    X = make_features(d, mode)
    Xtr, Xte = X[tr_mask], X[te_mask]
    y = d.home_win.to_numpy(float)
    ytr = y[tr_mask]
    off = offset_of(d, mode)
    otr, ote = off[tr_mask], off[te_mask]
    out = {}
    season = te.season.iloc[0]

    def log(model, **kw):
        fold_log.append(dict(tag=tag, mode=mode, season=season, model=model, n_train=int(tr_mask.sum()), **kw))

    if "A_lgb" in models:
        b, _, info = fit_lgb(tr, Xtr, ytr, otr)
        out["A_lgb"] = sigmoid(ote + b.predict(Xte, raw_score=True))
        log("A_lgb", n_iter=info["n_iter"], leaves=info["cfg"]["num_leaves"],
            inner_gain=info["inner_val"] - info["inner_base"])
        imp_log.append(pd.DataFrame(dict(tag=tag, mode=mode, season=season, model="A_lgb", feature=b.feature_name(),
                                         gain=b.feature_importance("gain"), splits=b.feature_importance("split"))))
    if "R1_lgb_no_market" in models:
        keep = [c for c in X.columns if c not in MARKET_COLS]
        z = np.zeros(len(d))
        b, _, info = fit_lgb(tr, Xtr[keep], ytr, z[tr_mask] + logit(ytr.mean()))
        out["R1_lgb_no_market"] = sigmoid(logit(ytr.mean()) + b.predict(Xte[keep], raw_score=True))
        log("R1_lgb_no_market", n_iter=info["n_iter"], leaves=info["cfg"]["num_leaves"],
            inner_gain=info["inner_val"] - info["inner_base"])
    if "R0_platt" in models:
        Z = pd.DataFrame({"mkt_logit": X.mkt_logit})
        m = LogReg(free=["mkt_logit"]).fit(tr, Z[tr_mask], ytr, otr)
        out["R0_platt"] = m.predict(Z[te_mask], ote)
        log("R0_platt", lam=m.lam, inner_gain=m.inner_val - m.inner_base)
    for name, ext, offspec in (("C1_logit_inter", False, False), ("C2_logit_inter_plus", True, False),
                               ("C1o_logit_inter_offset", False, True), ("C2o_logit_inter_plus_offset", True, True)):
        if name in models:
            Z = interaction_frame(d, X, extended=ext, mode=mode)
            m = LogReg(free=["mkt_logit"], offset_spec=offspec).fit(tr, Z[tr_mask], ytr, otr)
            out[name] = m.predict(Z[te_mask], ote)
            log(name, lam=m.lam, inner_gain=m.inner_val - m.inner_base)
            c = m.coefs() / m.sd  # back to raw units (per unit of the covariate)
            coef_log.append(pd.DataFrame(dict(tag=tag, mode=mode, season=season, model=name, term=c.index,
                                              coef_std=m.coefs().values, coef_raw=c.values)))
    if "D1_margin_shift" in models or "D2_margin_link" in models:
        # expected home margin implied by the market: the closing spread, or for the opening model the
        # opening ML mapped to points with the spread~probit(p_close) slope estimated on training rows
        m_tr = d.margin.to_numpy(float)[tr_mask]
        if mode == "close":
            mu = -d.spread.to_numpy(float)
        else:
            pr = norm.ppf(np.clip(d.p_close.to_numpy(float), 1e-4, 1 - 1e-4))
            ok = tr_mask & (d.spread_missing.to_numpy() == 0)
            k_s = np.polyfit(pr[ok], -d.spread.to_numpy(float)[ok], 1)[0]
            mu = k_s * norm.ppf(np.clip(d.p_open.to_numpy(float), 1e-4, 1 - 1e-4))
        mu_tr, mu_te = mu[tr_mask], mu[te_mask]
        b, b_inner, info = fit_lgb(tr, Xtr, m_tr, mu_tr, objective="regression")
        sigma = float(np.std(m_tr - mu_tr))
        adj_te = b.predict(Xte, raw_score=True)
        out["D1_margin_shift"] = sigmoid(ote + PROBIT_TO_LOGIT * adj_te / sigma)
        # D2: pure margin model -> P(win) through a logistic link fitted on the INNER validation rows
        itr, iva = inner_split(tr)
        z_va = (mu_tr[iva] + b_inner.predict(Xtr[iva], raw_score=True, num_iteration=b_inner.best_iteration)) / sigma
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z_va[:, None], ytr[iva])
        z_te = (mu_te + adj_te) / sigma
        out["D2_margin_link"] = lr.predict_proba(z_te[:, None])[:, 1]
        log("D_margin", n_iter=info["n_iter"], leaves=info["cfg"]["num_leaves"],
            inner_gain=info["inner_val"] - info["inner_base"], sigma=sigma, adj_sd=float(np.std(adj_te)))
        imp_log.append(pd.DataFrame(dict(tag=tag, mode=mode, season=season, model="D_margin", feature=b.feature_name(),
                                         gain=b.feature_importance("gain"), splits=b.feature_importance("split"))))
    if mode == "open" and ("F_move_lgb" in models or "F_move_ridge" in models):
        # predict the open->close move (a far less noisy target than the result) from pre-game info,
        # then publish logit(open) + predicted move. The close is only used as a TRAINING target.
        mv = logit(d.p_close.to_numpy(float)) - off
        mv_tr = mv[tr_mask]
        b, _, info = fit_lgb(tr, Xtr, mv_tr, np.full(len(mv_tr), mv_tr.mean()), objective="regression")
        pm = mv_tr.mean() + b.predict(Xte, raw_score=True)
        out["F_move_lgb"] = sigmoid(ote + pm)
        r = RidgeMove().fit(tr, Xtr, mv_tr)
        pr = r.predict(Xte)
        out["F_move_ridge"] = sigmoid(ote + pr)
        mv_te = mv[te_mask]
        r2 = lambda p: 1 - ((mv_te - p) ** 2).mean() / ((mv_te - mv_tr.mean()) ** 2).mean()
        log("F_move", n_iter=info["n_iter"], leaves=info["cfg"]["num_leaves"], lam=r.alpha,
            r2_move_lgb=r2(pm), r2_move_ridge=r2(pr), corr_move_lgb=float(np.corrcoef(pm, mv_te)[0, 1]))
        imp_log.append(pd.DataFrame(dict(tag=tag, mode=mode, season=season, model="F_move_lgb",
                                         feature=b.feature_name(), gain=b.feature_importance("gain"),
                                         splits=b.feature_importance("split"))))
    if "G1_screen_top1" in models or "G5_screen_top5" in models:
        # honest data mining: rank every feature by |corr(feature, y - p_market)| on the TRAINING rows only,
        # keep the top k, fit an offset logistic (shrunk toward the market) on them
        res_tr = ytr - sigmoid(otr)
        Xf = Xtr.astype(float)
        Xf = Xf.fillna(Xf.median())
        sd = Xf.std()
        cols = [c for c in Xf.columns if sd[c] > 0 and c != "mkt_logit"]
        cor = Xf[cols].apply(lambda s: np.corrcoef(s.to_numpy(), res_tr)[0, 1]).abs().sort_values(ascending=False)
        for name, k in (("G1_screen_top1", 1), ("G5_screen_top5", 5)):
            if name not in models:
                continue
            top = list(cor.index[:k])
            m = LogReg(offset_spec=True).fit(tr, X.loc[tr_mask, top], ytr, otr)
            out[name] = m.predict(X.loc[te_mask, top], ote)
            log(name, lam=m.lam, inner_gain=m.inner_val - m.inner_base, picked=",".join(top))
    if "E_ensemble" in models:
        parts = [k for k in ("A_lgb", "C2_logit_inter_plus", "D1_margin_shift") if k in out]
        out["E_ensemble"] = sigmoid(np.mean([logit(out[k]) for k in parts], axis=0))
    return pd.DataFrame(out, index=te.index)


def walk_forward(d, mode, tests, models, train_from=None, tag="main"):
    fold_log, imp_log, coef_log, preds = [], [], [], []
    has = d.p_close.notna() if mode == "close" else d.p_open.notna()
    for s in tests:
        t0 = time.time()
        tr_mask = (d.season < s) & has
        if train_from:
            tr_mask &= d.season >= train_from
        te_mask = (d.season == s) & has
        p = fold_predictions(d, tr_mask.to_numpy(), te_mask.to_numpy(), mode, models, fold_log, imp_log,
                             coef_log, tag)
        preds.append(p)
        print(f"  [{tag}/{mode}] {s}: train {tr_mask.sum()} test {te_mask.sum()} ({time.time() - t0:.0f}s)")
    P = pd.concat(preds)
    return P, pd.DataFrame(fold_log), (pd.concat(imp_log) if imp_log else None), \
        (pd.concat(coef_log) if coef_log else None)


# ------------------------------------------------------------------------------------------- evaluation
def boot_means(x, B, seed=0):
    rng = np.random.default_rng(seed)
    n, out = len(x), np.empty(B)
    for i in range(0, B, 250):
        k = min(250, B - i)
        out[i:i + k] = x[rng.integers(0, n, (k, n))].mean(1)
    return out


def evaluate(d, P, base_col, K, label):
    """Per-game paired log-loss differences vs the raw market price."""
    idx = P.index
    y = d.loc[idx, "home_win"].to_numpy(float)
    base = d.loc[idx, base_col].to_numpy(float)
    ll_b = ll_vec(y, base)
    seasons = d.loc[idx, "season"].to_numpy()
    rows, per = [], []
    for m in P.columns:
        diff = ll_vec(y, P[m].to_numpy()) - ll_b
        bs = boot_means(diff, B_LL, seed=1)  # same seed -> same resamples for every model (paired)
        a = 0.05 / K
        sd = {s: diff[seasons == s].mean() for s in sorted(set(seasons))}
        n_neg = sum(v < 0 for v in sd.values())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        blo, bhi = np.percentile(bs, [100 * a / 2, 100 * (1 - a / 2)])
        need = 3 if len(sd) == 4 else len(sd)
        rows.append(dict(baseline=label, model=m, n=len(diff), logloss=ll_vec(y, P[m].to_numpy()).mean(),
                         base_logloss=ll_b.mean(), delta=diff.mean(), ci95_lo=lo, ci95_hi=hi,
                         bonf_lo=blo, bonf_hi=bhi, p_boot=float((bs >= 0).mean()), seasons_neg=n_neg,
                         passes=bool(bhi < 0 and n_neg >= need), passes_95=bool(hi < 0 and n_neg >= need)))
        for s, v in sd.items():
            per.append(dict(baseline=label, model=m, season=s, delta=v,
                            base_logloss=ll_b[seasons == s].mean(), n=int((seasons == s).sum())))
    return pd.DataFrame(rows), pd.DataFrame(per)


def betting(d, P, odds, label):
    """Flat 1u on the side whose model probability exceeds the vig-inclusive implied price by >= thr."""
    idx = P.index
    y = d.loc[idx, "home_win"].to_numpy(float)
    dh, da = d.loc[idx, f"dec_h_{odds}"].to_numpy(float), d.loc[idx, f"dec_a_{odds}"].to_numpy(float)
    ok = np.isfinite(dh) & np.isfinite(da)
    rows = []
    for m in P.columns:
        p = P[m].to_numpy()
        eh, ea = p - 1 / dh, (1 - p) - 1 / da
        home_side = eh >= ea
        edge = np.where(home_side, eh, ea)
        win = np.where(home_side, y == 1, y == 0)
        dec = np.where(home_side, dh, da)
        prof = np.where(win, dec - 1, -1.0)
        for thr in THRESHOLDS:
            sel = ok & (edge > thr)
            n = int(sel.sum())
            if n < 20:
                rows.append(dict(baseline=label, odds=odds, model=m, thr=thr, bets=n))
                continue
            x = prof[sel]
            bs = boot_means(x, B_BET, seed=2)
            rows.append(dict(baseline=label, odds=odds, model=m, thr=thr, bets=n, roi=x.mean(),
                             ci95_lo=np.percentile(bs, 2.5), ci95_hi=np.percentile(bs, 97.5),
                             hit=win[sel].mean(), avg_dec=dec[sel].mean(),
                             fav_share=float((dec[sel] < 2).mean())))
    return pd.DataFrame(rows)


def importance_table(imp, model):
    t = imp[imp.model == model].groupby(["feature"]).agg(gain=("gain", "sum"), splits=("splits", "sum"))
    tot = t.gain.sum()
    t["gain_share"] = t.gain / tot if tot > 0 else 0.0
    return t.sort_values("gain", ascending=False)


# ------------------------------------------------------------------------------------------- power check
def power_check(d, effects=(0.0, 0.1, 0.2), draws=2):
    """Synthetic test of the pipeline: outcomes are re-drawn from sigmoid(logit(close) + s(X)) where s is a
    KNOWN residual the market 'misses' (nonlinear: a rest/back-to-back term plus a favourite-longshot
    term). If the pipeline cannot find an effect of a given size, a real one of that size could hide."""
    X = make_features(d, "close")
    L = logit(d.p_close.to_numpy(float))
    rest = np.clip(d.rest_diff.to_numpy(float), -3, 3) / 3
    b2b = d.b2b_diff.to_numpy(float)
    shape = 0.5 * b2b + 0.5 * rest - 0.4 * np.tanh(L / 2) * (np.abs(L) > 1.5)
    shape = shape / np.std(shape)
    rows = []
    for eff in effects:
        for r in range(draws):
            rng = np.random.default_rng(100 + r)
            p_true = sigmoid(L + eff * shape)
            dd = d.copy()
            dd["home_win"] = (rng.random(len(d)) < p_true).astype(float)
            P, *_ = walk_forward(dd, "close", CLOSE_TEST, ["A_lgb", "C2o_logit_inter_plus_offset"], tag=f"power{eff}")
            idx = P.index
            y = dd.loc[idx, "home_win"].to_numpy()
            llb = ll_vec(y, dd.loc[idx, "p_close"].to_numpy())
            oracle = (ll_vec(y, p_true[idx]) - llb).mean()
            for m in P.columns:
                diff = ll_vec(y, P[m].to_numpy()) - llb
                bs = boot_means(diff, 2000, seed=3)
                rows.append(dict(effect_sd_logit=eff, draw=r, model=m, delta=diff.mean(), oracle_delta=oracle,
                                 ci95_hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- main
def main(power=True):
    sys.stdout = Tee(RES / "run_log.txt")
    t0 = time.time()
    d = h5data.build()
    print(f"dataset: {len(d)} games, seasons {d.season.min()}..{d.season.max()}")
    print(d.groupby("season").agg(n=("home_win", "size"), open=("p_open", "count"), src=("odds_src", "first")))

    close_models = ["A_lgb", "C1_logit_inter", "C2_logit_inter_plus", "C1o_logit_inter_offset",
                    "C2o_logit_inter_plus_offset", "D1_margin_shift", "D2_margin_link", "E_ensemble",
                    "G1_screen_top1", "G5_screen_top5", "R0_platt", "R1_lgb_no_market"]
    open_models = close_models + ["F_move_lgb", "F_move_ridge"]

    print("\n=== walk-forward vs CLOSING line (train = all earlier seasons with odds, from 2018-19) ===")
    Pc, fc, ic, cc = walk_forward(d, "close", CLOSE_TEST, close_models)
    print("\n=== robustness: CLOSING line, training from 2021-22 only ===")
    Pr, fr, ir, cr = walk_forward(d, "close", CLOSE_TEST, ["A_lgb", "C1_logit_inter", "C2_logit_inter_plus",
                                                            "D1_margin_shift", "E_ensemble"],
                                  train_from="2021-22", tag="from2021")
    Pr.columns = [f"{c}@2021+" for c in Pr.columns]
    print("\n=== walk-forward vs OPENING line (train = 2023-24 onward) ===")
    Po, fo, io, co = walk_forward(d, "open", OPEN_TEST, open_models)

    K_close = Pc.shape[1] + Pr.shape[1]
    K_open = Po.shape[1]
    ev_c, ps_c = evaluate(d, pd.concat([Pc, Pr], axis=1), "p_close", K_close, "close")
    ev_o, ps_o = evaluate(d, Po, "p_open", K_open, "open")
    # opening-line models judged against the CLOSING line too (what an early publisher gives up)
    ev_oc, ps_oc = evaluate(d, Po, "p_close", K_open, "open_models_vs_close")
    pd.set_option("display.width", 200)
    cols = ["model", "n", "logloss", "base_logloss", "delta", "ci95_lo", "ci95_hi", "bonf_lo", "bonf_hi",
            "seasons_neg", "passes"]
    print(f"\n=== RESULT vs CLOSING line ({K_close} variants, Bonferroni level {1 - 0.05 / K_close:.4f}) ===")
    print(ev_c[cols].round(5).to_string(index=False))
    print("\nper-season delta vs close (negative = better than market):")
    print(ps_c.pivot(index="model", columns="season", values="delta").round(5).to_string())
    print(f"\n=== RESULT vs OPENING line ({K_open} variants; test {OPEN_TEST}, 2 of 2 seasons needed) ===")
    print(ev_o[cols].round(5).to_string(index=False))
    print(ps_o.pivot(index="model", columns="season", values="delta").round(5).to_string())
    print("\nopening-line models vs the CLOSING line, same games:")
    print(ev_oc[["model", "n", "logloss", "base_logloss", "delta", "ci95_lo", "ci95_hi"]].round(5).to_string(index=False))

    cand_c = ev_c[~ev_c.model.str.startswith("R")]
    print(f"\nclose: best variant {cand_c.loc[cand_c.delta.idxmin(), 'model']} delta {cand_c.delta.min():+.5f}, "
          f"median variant delta {cand_c.delta.median():+.5f}")
    cand_o = ev_o[~ev_o.model.str.startswith("R")]
    print(f"open : best variant {cand_o.loc[cand_o.delta.idxmin(), 'model']} delta {cand_o.delta.min():+.5f}, "
          f"median variant delta {cand_o.delta.median():+.5f}")

    folds = pd.concat([fc, fr, fo])
    print("\n=== fold details (n_iter = boosting rounds chosen by inner early stopping; inner_gain < 0 = the "
          "inner validation season improved on the market) ===")
    print(folds.round(5).to_string(index=False))

    bet = pd.concat([betting(d, pd.concat([Pc, Pr], axis=1), "close", "close"),
                     betting(d, Po, "open", "open"), betting(d, Po, "close", "open_models_at_close_odds")])
    nb = int(bet.roi.notna().sum())
    print(f"\n=== betting: flat 1u at vig-inclusive odds when model prob - implied > thr ({nb} rules) ===")
    print(bet.round(4).to_string(index=False))
    sig = bet[bet.ci95_lo > 0]
    print(f"rules with 95% CI above 0: {len(sig)} of {nb}")

    for m, I in (("A_lgb", ic), ("D_margin", ic)):
        t = importance_table(I, m)
        print(f"\n=== {m} (close) gain importance, summed over the 4 folds; top 20 ===")
        print(t.head(20).round(4).to_string())
        t.to_csv(RES / f"importance_close_{m}.csv")
    for m in ("A_lgb", "F_move_lgb"):
        t = importance_table(io, m)
        print(f"\n=== {m} (open) gain importance; top 20 ===")
        print(t.head(20).round(4).to_string())
        t.to_csv(RES / f"importance_open_{m}.csv")
    coefs = pd.concat([cc, co])
    last = coefs[coefs.season == coefs.season.max()]
    print("\n=== logistic (c) coefficients, last fold (standardised units) ===")
    print(last.pivot_table(index="term", columns=["mode", "model"], values="coef_std").round(4).to_string())

    ev_c.to_csv(RES / "walkforward_close.csv", index=False)
    ev_o.to_csv(RES / "walkforward_open.csv", index=False)
    ev_oc.to_csv(RES / "open_models_vs_close.csv", index=False)
    pd.concat([ps_c, ps_o, ps_oc]).to_csv(RES / "per_season.csv", index=False)
    folds.to_csv(RES / "folds.csv", index=False)
    bet.to_csv(RES / "betting.csv", index=False)
    coefs.to_csv(RES / "logreg_coefs.csv", index=False)
    keep = ["season", "game_date", "home", "away", "home_win", "p_close", "p_open"]
    pd.concat([d.loc[Pc.index, keep], Pc, Pr], axis=1).to_csv(RES / "predictions_close.csv", index=False)
    pd.concat([d.loc[Po.index, keep], Po], axis=1).to_csv(RES / "predictions_open.csv", index=False)

    if power:
        print("\n=== power check: synthetic residual signal injected on top of the closing line ===")
        pw = power_check(d)
        pw.to_csv(RES / "power.csv", index=False)
        print(pw.groupby(["effect_sd_logit", "model"]).agg(delta=("delta", "mean"), oracle=("oracle_delta", "mean"),
                                                           ci95_hi=("ci95_hi", "mean"),
                                                           detected=("ci95_hi", lambda s: (s < 0).mean()))
              .round(5).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")
    # diagnostics (own logs): where the opening-line gain comes from; did residual structure fade?
    import decay
    import open_diagnostics
    open_diagnostics.main()
    decay.main()


if __name__ == "__main__":
    main(power="--no-power" not in sys.argv)
