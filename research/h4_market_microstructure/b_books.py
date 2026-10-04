"""(b) Cross-book dispersion. Multi-book data exists for 2021-22..2023-24 only (2024-25 and
2025-26 carry a single book in the ESPN feed), so walk-forward test seasons are 2022-23 and
2023-24; the 3-of-4-seasons rule cannot be met by construction.

- per-book sharpness (log-loss on the games each book prices), consensus vs main vs outlier
- walk-forward: does consensus / outlier deviation / dispersion add to the main closing line?
- line shopping: best available price across books vs the consensus fair probability
- leak check: does a book's deviation from the main close predict the result too well
  (a sign that its stored 'current' line was captured in-play)?
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import compare, ll_vec, logit, pnl_roi, sigmoid, walk_forward, bet_roi

EXCLUDE = ("MGM",)  # two sides recorded inconsistently: 33% of 2021-22 rows have negative vig
MULTI = ["2021-22", "2022-23", "2023-24"]
TEST = ["2022-23", "2023-24"]


def game_consensus(g, b):
    b = b[~b.book.isin(EXCLUDE)].copy()
    b["lp"] = logit(b.p)
    agg = b.groupby("event_id").agg(n_books=("p", "size"), p_med=("p", "median"),
                                    p_mean=("p", "mean"), lp_mean=("lp", "mean"),
                                    lp_std=("lp", "std"), p_std=("p", "std"),
                                    best_h=("h_dec", "max"), best_a=("a_dec", "max"))
    b = b.merge(agg[["p_med"]], left_on="event_id", right_index=True)
    b["dev_med"] = b.lp - logit(b.p_med)
    i = b.groupby("event_id").dev_med.apply(lambda s: s.abs().idxmax())
    out = b.loc[i.values, ["event_id", "book", "p", "dev_med"]].rename(
        columns={"book": "outlier_book", "p": "p_outlier", "dev_med": "outlier_dev"})
    agg = agg.merge(out, left_index=True, right_on="event_id")
    # best-price book per side
    bh = b.loc[b.groupby("event_id").h_dec.idxmax(), ["event_id", "book"]].rename(columns={"book": "best_h_book"})
    ba = b.loc[b.groupby("event_id").a_dec.idxmax(), ["event_id", "book"]].rename(columns={"book": "best_a_book"})
    agg = agg.merge(bh, on="event_id").merge(ba, on="event_id")
    m = g.merge(agg, on="event_id", how="inner")
    m = m[m.season.isin(MULTI) & (m.n_books >= 4)].reset_index(drop=True)
    return m, b


def leak_check(g, b):
    """Per book-season: LR of home_win on [logit(main close), deviation]. A deviation
    coefficient >> 1 (in logit units) would indicate in-play contamination."""
    d = b.merge(g[["event_id", "season", "home_win", "p_close"]], on="event_id")
    rows = []
    for (bk, s), x in d.groupby(["book", "season"]):
        if len(x) < 300 or x.dev_main.abs().max() < 1e-9:
            continue
        X = np.c_[logit(x.p_close), x.dev_main]
        m = LogisticRegression(C=1e4, max_iter=5000).fit(X, x.home_win)
        # t-stat via observed information
        p = m.predict_proba(X)[:, 1]
        Xi = np.c_[np.ones(len(X)), X]
        cov = np.linalg.pinv(Xi.T @ (Xi * (p * (1 - p))[:, None]))
        rows.append({"book": bk, "season": s, "n": len(x), "coef_dev": m.coef_[0, 1],
                     "se": np.sqrt(cov[2, 2]), "mad_dev": x.dev_main.abs().mean(),
                     "ll_book": ll_vec(x.home_win, x.p).mean(), "ll_main": ll_vec(x.home_win, x.p_close).mean()})
    r = pd.DataFrame(rows)
    r["z"] = r.coef_dev / r.se
    return r


def run(g, books):
    out = {"logloss": [], "bets": [], "tables": {}}
    m, b = game_consensus(g, books)
    out["tables"]["b_leak_check"] = leak_check(g, b)
    out["tables"]["b_coverage"] = m.groupby("season").agg(games=("event_id", "size"),
                                                         books_mean=("n_books", "mean"),
                                                         disp_lp_std=("lp_std", "mean")).reset_index()

    # descriptive log-loss: main vs consensus vs outlier vs each book (all multi-book seasons)
    rows = []
    for name, p in [("main_close", m.p_close), ("consensus_median", m.p_med),
                    ("consensus_mean", m.p_mean), ("consensus_logit_mean", sigmoid(m.lp_mean)),
                    ("outlier_book", m.p_outlier)]:
        for s in MULTI + ["ALL"]:
            x = m if s == "ALL" else m[m.season == s]
            pp = p if s == "ALL" else p[x.index]
            rows.append({"price": name, "season": s, "n": len(x), "logloss": ll_vec(x.home_win, pp).mean()})
    out["tables"]["b_price_logloss"] = pd.DataFrame(rows).pivot(index="price", columns="season", values="logloss").reset_index()
    # each book head-to-head vs consensus on the same games
    bb = b.drop(columns=["p_med"]).merge(m[["event_id", "season", "home_win", "p_med", "p_close"]], on="event_id")
    sh = bb.groupby("book").apply(lambda x: pd.Series({
        "n": len(x), "ll_book": ll_vec(x.home_win, x.p).mean(),
        "ll_consensus_same_games": ll_vec(x.home_win, x.p_med).mean(),
        "vig": x.vig.mean()})).reset_index()
    sh["book_minus_consensus"] = sh.ll_book - sh.ll_consensus_same_games
    out["tables"]["b_book_sharpness"] = sh.sort_values("book_minus_consensus")

    # outlier vs consensus directly, test seasons (no fitting)
    t = m[m.season.isin(TEST)]
    out["logloss"].append({"part": "b", **compare(t, t.p_med.to_numpy(), t.p_close.to_numpy(),
                                                   "consensus_median_as_price")})
    out["logloss"].append({"part": "b", **compare(t, t.p_outlier.to_numpy(), t.p_close.to_numpy(),
                                                   "outlier_book_as_price")})
    variants = {
        "main+consensus_gap": lambda d: logit(d.p_med) - logit(d.p_close),
        "main+outlier_dev": lambda d: d.outlier_dev,
        "main+dispersion_shrink": lambda d: d.lp_std.fillna(0) * logit(d.p_close),
        "main+consensus_gap+dispersion": lambda d: np.c_[logit(d.p_med) - logit(d.p_close),
                                                         d.lp_std.fillna(0) * logit(d.p_close)],
    }
    platt = walk_forward(m, None, test_seasons=TEST)
    preds = {}
    for name, fx in variants.items():
        p = walk_forward(m, fx, test_seasons=TEST)
        preds[name] = p
        r = compare(t, p[t.index].to_numpy(), t.p_close.to_numpy(), name)
        r["delta_vs_platt"] = compare(t, p[t.index].to_numpy(), platt[t.index].to_numpy(), name)["delta"]
        out["logloss"].append({"part": "b", **r})

    # ----------------------------------------------------------------- line shopping
    # rule: bet a side when consensus fair prob x best decimal price - 1 > threshold
    for seasons, tag in [(TEST, "test"), (MULTI, "all3")]:
        x = m[m.season.isin(seasons)]
        ev_h = x.p_med * x.best_h - 1
        ev_a = (1 - x.p_med) * x.best_a - 1
        for th in (0.0, 0.01, 0.02, 0.03, 0.05):
            bh = (ev_h > th) & (ev_h >= ev_a)
            ba = (ev_a > th) & ~bh
            pnl = np.r_[np.where(x.home_win[bh] == 1, x.best_h[bh] - 1, -1.0),
                        np.where(x.home_win[ba] == 0, x.best_a[ba] - 1, -1.0)]
            ss = np.r_[x.season[bh], x.season[ba]]
            r = pnl_roi(pnl, ss, label=f"best price EV>{th:.0%} vs consensus [{tag}]")
            evs = np.r_[ev_h[bh], ev_a[ba]]
            r["mean_ev_claimed"] = float(evs.mean()) if len(pnl) else np.nan
            if len(pnl):  # if the consensus were the truth, realised ROI would equal the claimed EV
                bt = np.random.default_rng(1).choice(pnl - evs, (2000, len(pnl))).mean(1)
                r["realised_minus_claimed"] = float((pnl - evs).mean())
                r["rmc_lo"], r["rmc_hi"] = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
            out["bets"].append({"part": "b", **r})
        # same rule restricted to games whose best prices are not a two-book arbitrage
        # (an arb at the close = the two quotes were almost surely not live at the same time)
        sync = (1 / x.best_h + 1 / x.best_a - 1) >= 0.005
        bh = (ev_h > 0.02) & (ev_h >= ev_a) & sync
        ba = (ev_a > 0.02) & ~bh & sync
        pnl = np.r_[np.where(x.home_win[bh] == 1, x.best_h[bh] - 1, -1.0),
                    np.where(x.home_win[ba] == 0, x.best_a[ba] - 1, -1.0)]
        r = pnl_roi(pnl, np.r_[x.season[bh], x.season[ba]],
                    label=f"best price EV>2% vs consensus, no-arb games only [{tag}]")
        r["mean_ev_claimed"] = float(np.r_[ev_h[bh], ev_a[ba]].mean()) if len(pnl) else np.nan
        out["bets"].append({"part": "b", **r})
        # all favourites / all dogs at best price
        fav_h = x.p_med >= .5
        for side, sel_h, sel_a in [("favourites", fav_h, ~fav_h), ("underdogs", ~fav_h, fav_h)]:
            pnl = np.r_[np.where(x.home_win[sel_h] == 1, x.best_h[sel_h] - 1, -1.0),
                        np.where(x.home_win[sel_a] == 0, x.best_a[sel_a] - 1, -1.0)]
            ss = np.r_[x.season[sel_h], x.season[sel_a]]
            out["bets"].append({"part": "b", **pnl_roi(pnl, ss, label=f"all {side} at best price [{tag}]")})
    # model-based (walk-forward consensus-gap model) against best price
    p = preds["main+consensus_gap"]
    tt = t.assign(hb=t.best_h, ab=t.best_a)
    for th in (0.0, 0.02):
        hd, ad = tt.hb.to_numpy(), tt.ab.to_numpy()
        pm = p[tt.index].to_numpy()
        eh, ea = pm * hd - 1, (1 - pm) * ad - 1
        bh = (eh > th) & (eh >= ea)
        ba = (ea > th) & ~bh
        y = tt.home_win.to_numpy()
        pnl = np.r_[np.where(y[bh] == 1, hd[bh] - 1, -1.0), np.where(y[ba] == 0, ad[ba] - 1, -1.0)]
        ss = np.r_[tt.season.to_numpy()[bh], tt.season.to_numpy()[ba]]
        out["bets"].append({"part": "b", **pnl_roi(pnl, ss, label=f"consensus-gap model EV>{th:.0%} at best price [test]")})
    # how much vig is left after shopping, and how often the best prices form an 'arbitrage'
    m["best_overround"] = 1 / m.best_h + 1 / m.best_a - 1
    out["tables"]["b_best_price_overround"] = m.groupby("season").agg(
        main_vig=("vig_close", "mean"), best_price_overround=("best_overround", "mean"),
        share_arbitrage=("best_overround", lambda s: (s < 0).mean())).reset_index()
    # which books supply the best price (stale-line warning)
    out["tables"]["b_best_price_books"] = (pd.concat([m.best_h_book, m.best_a_book]).value_counts(normalize=True)
                                           .rename("share_of_best_prices").reset_index())
    out["consensus_frame"] = m
    return out
