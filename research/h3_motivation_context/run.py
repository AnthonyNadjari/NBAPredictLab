"""H3 - motivation & competitive context vs the betting market.  End-to-end:

    python research/h3_motivation_context/fetch_older_odds.py   # once, ~25 min, cached (ESPN 2018-19..2020-21)
    python research/h3_motivation_context/run.py                # ~5 min, no network

1. standings / series-state features as of the morning of each game (h3feat.py), evaluation set (h3data.py)
2. descriptive slices (pre-listed, Bonferroni over all of them): does the side of interest win more/less
   than the de-vigged closing price, cover the spread, return money at the vig-inclusive close?
   Playoff slices use a series-clustered bootstrap (games of one series are not independent).
3. the 'playoff favourites underperform' hint (found on 2022-23..2025-26): independent re-test on
   2018-19..2021-22, per season, calibration slope, ATS, ROI of fading favourites
4. walk-forward log-loss vs closing line (test 2022-23..2025-26, trained on all earlier seasons with odds)
   and vs opening line (test 2024-25, 2025-26; opening odds exist from 2023-24); specs A (recal) and B (offset)
5. betting view of every model at the vig-inclusive closing (and opening) moneyline
6. line movement open -> close in the direction of the context features (is the info priced by the close?)
7. robustness: training window starting 2021-22 only (comparable with the other hypotheses)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from h3data import build  # noqa: E402
from h3eval import (FAMILY, PO_ALL, RS_SINGLE, PO_SINGLE, VARIANTS, bet_profits, boot_cluster_mean,  # noqa: E402
                    boot_mean, ci, game_features, ll_vec, logit, side_profit, walk_forward)

warnings.filterwarnings("ignore")
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
CACHE = HERE.parents[1] / "research/data/h3_motivation_context"
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
CLOSE_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
OPEN_SEASONS = ["2024-25", "2025-26"]
N_BOOT = 4000


def series_id(d):
    a = np.where(d.home < d.away, d.home, d.away)
    b = np.where(d.home < d.away, d.away, d.home)
    return d.season + "_" + d.season_type + "_" + a + "_" + b


# ----------------------------------------------------------------------------- 2. descriptive slices
def slice_defs(d):
    rs = (d.season_type == "Regular Season").to_numpy()
    po = (d.season_type == "Playoffs").to_numpy()
    pi = (d.season_type == "PlayIn").to_numpy()
    p = d.mkt_close.to_numpy()
    fav = np.where(p >= 0.5, 1, -1)
    H = lambda c: d[f"home_{c}"].fillna(0).to_numpy() == 1  # noqa: E731
    A = lambda c: d[f"away_{c}"].fillna(0).to_numpy() == 1  # noqa: E731
    lw = d.home_last_week.fillna(0).to_numpy() == 1
    g = lambda c: d[c].fillna(0).to_numpy()  # noqa: E731
    one = lambda mh, ma: (mh & ~ma, ma & ~mh)  # noqa: E731

    def side_of(mh, ma):  # +1 where home is the team of interest, -1 where away
        return np.where(mh, 1, np.where(ma, -1, 0))

    S = []
    # regular season: side = the team of interest
    ns_h, ns_a = H("no_stakes"), A("no_stakes")
    h1, a1 = one(~ns_h, ~ns_a)  # exactly one side still has stakes
    S.append(("RS", "team with stakes vs no-stakes opponent", rs & (h1 | a1), side_of(h1, a1)))
    S.append(("RS", "  same, last week of season", rs & lw & (h1 | a1), side_of(h1, a1)))
    h1, a1 = one(H("elim_post"), A("elim_post"))
    S.append(("RS", "eliminated team vs non-eliminated", rs & (h1 | a1), side_of(h1, a1)))
    h1, a1 = one(H("seed_locked"), A("seed_locked"))
    S.append(("RS", "seed-locked playoff team vs not locked", rs & (h1 | a1), side_of(h1, a1)))
    S.append(("RS", "  same, last week of season", rs & lw & (h1 | a1), side_of(h1, a1)))
    h1, a1 = one(H("tank"), A("tank"))
    S.append(("RS", "tank team (bottom-6, <=25 left) vs non-tank", rs & (h1 | a1), side_of(h1, a1)))
    h1, a1 = one(H("in_race"), A("in_race"))
    S.append(("RS", "team in a seeding race vs opp not in race", rs & (h1 | a1), side_of(h1, a1)))
    h1, a1 = H("in_race") & A("no_stakes"), A("in_race") & H("no_stakes")
    S.append(("RS", "team in race vs no-stakes opponent", rs & (h1 | a1), side_of(h1, a1)))
    h1, a1 = H("last_game") & H("no_stakes"), A("last_game") & A("no_stakes")
    h1, a1 = h1 & ~a1, a1 & ~h1
    S.append(("RS", "no-stakes team in its final game", rs & (h1 | a1), side_of(h1, a1)))
    S.append(("RS", "both teams in a race (side = home)", rs & H("in_race") & A("in_race"), np.ones(len(d))))
    S.append(("RS", "both teams no stakes (side = home)", rs & ns_h & ns_a, np.ones(len(d))))
    S.append(("RS", "last week, favourite", rs & lw, fav))
    # playoffs / play-in
    S.append(("PO", "playoff favourite", po, fav))
    S.append(("PO", "  home favourite", po & (fav == 1), fav))
    S.append(("PO", "  road favourite", po & (fav == -1), fav))
    S.append(("PO", "  big favourite (>=70%)", po & (np.maximum(p, 1 - p) >= 0.7), fav))
    S.append(("PO", "  round 1", po & (g("round") == 1), fav))
    S.append(("PO", "  rounds 2-4", po & (g("round") >= 2), fav))
    for k in range(1, 8):
        S.append(("PO", f"  game {k}, favourite", po & (g("game_no") == k), fav))
    S.append(("PO", "play-in favourite", pi, fav))
    zz = g("home_lost_prev")
    S.append(("PO", "zig-zag: loser of previous series game", po & (zz != 0), zz))
    S.append(("PO", "  zig-zag, previous loser is favourite", po & (zz != 0) & (zz == fav), zz))
    S.append(("PO", "  zig-zag, previous loser is underdog", po & (zz != 0) & (zz != fav), zz))
    S.append(("PO", "  lost previous game by 15+", po & (zz != 0) & (np.abs(g("prev_margin_home")) >= 15), zz))
    he, ae = g("home_elim") == 1, g("away_elim") == 1
    S.append(("PO", "team facing elimination (not game 7)", po & (he ^ ae), np.where(he, 1, -1)))
    S.append(("PO", "  facing elimination at home", po & he & ~ae, np.ones(len(d))))
    S.append(("PO", "  facing elimination on the road", po & ae & ~he, -np.ones(len(d))))
    S.append(("PO", "game 7 home team", po & (g("game7") == 1), np.ones(len(d))))
    S.append(("PO", "game 3 home team down 0-2", po & (g("home_down02") == 1), np.ones(len(d))))
    S.append(("PO", "game 3 home team at 1-1", po & (g("game_no") == 3) & (g("home_series_w") == 1),
              np.ones(len(d))))
    S.append(("PO", "home team up 2-0 (game 3)", po & (g("home_down02") == -1), np.ones(len(d))))
    S.append(("PO", "game 1 home team", po & (g("game_no") == 1), np.ones(len(d))))
    return S


def slice_stats(d, mask, side, clusters=None, seed=0):
    s = d[mask]
    sd = np.asarray(side)[mask].astype(float)
    if len(s) == 0:
        return {"n": 0}
    p_side = np.where(sd > 0, s.mkt_close, 1 - s.mkt_close)
    won = np.where(sd > 0, s.home_win, 1 - s.home_win).astype(float)
    gap = won - p_side
    z = gap.sum() / np.sqrt((p_side * (1 - p_side)).sum())
    r = {"n": len(s), "priced": p_side.mean(), "won": won.mean(), "gap_pts": 100 * gap.mean(), "z": z}
    if clusters is not None:
        bs = boot_cluster_mean(gap, np.asarray(clusters)[mask], 2000, seed)
        r["gap_ci_cluster"] = tuple(np.round(100 * np.array(ci(bs)), 1))
    cov = sd * (s.margin + s.spread).to_numpy(float)
    okc = np.isfinite(cov) & (cov != 0)
    if okc.sum() >= 20:
        r["ats_n"], r["ats_cover"] = int(okc.sum()), float((cov[okc] > 0).mean())
        r["ats_pts"] = float(np.nanmean(cov[np.isfinite(cov)]))
        r["ats_t"] = float(r["ats_pts"] / (np.nanstd(cov[np.isfinite(cov)], ddof=1) / np.sqrt(np.isfinite(cov).sum())))
    prof = side_profit(s.home_win, sd, s.dec_home_close.to_numpy(float), s.dec_away_close.to_numpy(float))
    okp = np.isfinite(prof)
    if okp.any():
        bs = (boot_cluster_mean(prof[okp], np.asarray(clusters)[mask][okp], 2000, seed + 1)
              if clusters is not None else boot_mean(prof[okp], 2000, seed + 1))
        r["roi"], r["roi_ci"] = prof[okp].mean(), tuple(np.round(ci(bs), 3))
    return r


def descriptive(d):
    S = slice_defs(d)
    cl = series_id(d).to_numpy()
    periods = {"2018-22 (hold-out)": d.season < "2022-23", "2022-26 (eval)": d.season >= "2022-23",
               "all 2018-26": np.ones(len(d), bool)}
    rows = []
    for fam, name, mask, side in S:
        for per, pm in periods.items():
            m = np.asarray(mask, bool) & np.asarray(pm, bool) & d.mkt_close.notna().to_numpy()
            r = slice_stats(d, m, side, clusters=cl if fam == "PO" else None)
            rows.append({"family": fam, "slice": name, "period": per, **r})
    out = pd.DataFrame(rows)
    n_slices = len(S)
    zcrit = float(__import__("scipy.stats").stats.norm.ppf(1 - 0.025 / n_slices))
    print(f"\n=== 2. Descriptive slices: {n_slices} pre-listed slices; side of interest vs de-vigged CLOSING price"
          f" (Bonferroni |z| > {zcrit:.2f}); gap/ats in points, ROI at vig-inclusive close ===")
    sh = out.copy()
    for c in ("priced", "won"):
        sh[c] = sh[c].map(lambda x: f"{x:.3f}")
    for c in ("gap_pts", "z", "ats_pts", "ats_t"):
        sh[c] = sh[c].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "")
    sh["ats_cover"] = sh.ats_cover.map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    sh["roi"] = sh.roi.map(lambda x: f"{x:+.3f}" if pd.notna(x) else "")
    for per in periods:
        print(f"\n--- {per}")
        cols = ["family", "slice", "n", "priced", "won", "gap_pts", "z", "gap_ci_cluster", "ats_n", "ats_cover",
                "ats_pts", "ats_t", "roi", "roi_ci"]
        print(sh[sh.period == per][cols].to_string(index=False))
    allp = out[out.period == "all 2018-26"]
    hits = allp[allp.z.abs() > zcrit]
    print(f"\nslices with |z| > {zcrit:.2f} (Bonferroni over {n_slices}) on all seasons: "
          f"{hits[['slice', 'n', 'gap_pts', 'z']].values.tolist()}")
    print(f"largest |z| (all seasons): {allp.loc[allp.z.abs().idxmax(), ['slice', 'n', 'gap_pts', 'z']].tolist()}")
    out.attrs["zcrit"], out.attrs["n_slices"] = zcrit, n_slices
    return out


# ----------------------------------------------------------------------------- 3. playoff favourite deep dive
def logit_fit(x, y):
    """y ~ a + b x; returns (a, b, se_a, se_b) with Fisher SEs."""
    from scipy.optimize import minimize
    from scipy.special import expit
    Z = np.column_stack([np.ones(len(y)), x])

    def f(b):
        e = Z @ b
        return -(y * e - np.logaddexp(0, e)).sum(), Z.T @ (expit(e) - y)

    b = minimize(f, np.array([0.0, 1.0]), jac=True, method="L-BFGS-B").x
    p = expit(Z @ b)
    cov = np.linalg.inv(Z.T @ (Z * (p * (1 - p))[:, None]))
    return b[0], b[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])


def playoff_favourites(d):
    po = d[(d.season_type == "Playoffs") & d.mkt_close.notna()].copy()
    po["fav"] = np.where(po.mkt_close >= 0.5, 1, -1)
    po["fav_p"] = np.maximum(po.mkt_close, 1 - po.mkt_close)
    po["fav_won"] = np.where(po.fav == 1, po.home_win, 1 - po.home_win)
    po["sid"] = series_id(po)
    po["dog_profit"] = side_profit(po.home_win, -po.fav, po.dec_home_close.to_numpy(float),
                                   po.dec_away_close.to_numpy(float))
    po["fav_cover"] = po.fav * (po.margin + po.spread)
    # spread-implied win probability of the favourite (normal margin, SD fitted on regular-season residuals)
    from scipy.stats import norm
    rsd = d[(d.season_type == "Regular Season")]
    sd_m = float(np.nanstd((rsd.margin + rsd.spread).to_numpy(float)))
    po["fav_p_spread"] = np.where(po.fav == 1, norm.cdf(-po.spread / sd_m), 1 - norm.cdf(-po.spread / sd_m))
    rows = []
    groups = [(s, po.season == s) for s in sorted(po.season.unique())]
    groups += [("2018-22 hold-out", po.season < "2022-23"), ("2022-26 (hint sample)", po.season >= "2022-23"),
               ("all 2018-26", np.ones(len(po), bool)), ("all excl. 2019-20 bubble", po.season != "2019-20")]
    for name, m in groups:
        s = po[np.asarray(m, bool)]
        gap = s.fav_won - s.fav_p
        z = gap.sum() / np.sqrt((s.fav_p * (1 - s.fav_p)).sum())
        bs = boot_cluster_mean(gap.to_numpy(), s.sid.to_numpy(), N_BOOT, 3)
        a, b, sea, seb = logit_fit(logit(s.mkt_close), s.home_win.to_numpy(float))
        prof = s.dog_profit.dropna()
        bsr = boot_cluster_mean(prof.to_numpy(), s.loc[prof.index, "sid"].to_numpy(), N_BOOT, 4) if len(prof) else [np.nan]
        cv = s.fav_cover.dropna()
        cv = cv[cv != 0]
        rows.append({"sample": name, "n": len(s), "series": s.sid.nunique(), "fav_priced": s.fav_p.mean(),
                     "fav_p_spread": s.fav_p_spread.mean(),
                     "fav_won": s.fav_won.mean(), "gap_pts": 100 * gap.mean(), "z": z,
                     "gap_ci95_cluster": tuple(np.round(100 * np.array(ci(bs)), 1)),
                     "calib_slope": b, "slope_se": seb, "slope_z_vs_1": (b - 1) / seb,
                     "fav_ats_cover": (cv > 0).mean(), "fav_ats_pts": s.fav_cover.mean(),
                     "dog_roi_close": prof.mean(), "dog_roi_ci": tuple(np.round(ci(bsr), 3)),
                     "odds_src": s.odds_src.mode().iloc[0]})
    out = pd.DataFrame(rows)
    # regular-season reference slope
    rs = d[(d.season_type == "Regular Season") & d.mkt_close.notna()]
    a, b, _, seb = logit_fit(logit(rs.mkt_close), rs.home_win.to_numpy(float))
    print("\n=== 3. Playoff favourites: favourite win rate vs de-vigged close; calibration slope (y ~ a + b*logit p;"
          " b<1 = favourites overpriced); ATS; ROI of betting every underdog at the vig-inclusive close ===")
    print(out.round(4).to_string(index=False))
    print(f"regular-season reference (all seasons, n={len(rs)}): calibration slope {b:.3f} (se {seb:.3f}); "
          f"fav_p_spread uses margin SD {sd_m:.2f} around the closing spread")
    # power: SE of the gap in the hold-out
    ho = po[po.season < "2022-23"]
    se = 100 * np.sqrt((ho.fav_p * (1 - ho.fav_p)).sum()) / len(ho)
    print(f"hold-out power: SE of the favourite gap = {se:.2f} pts -> a true -4.7 pt effect would give z ~ {-4.7 / se:.2f}"
          f" (power to reach z<-1.96 ~ {__import__('scipy.stats').stats.norm.cdf((4.7 - 1.96 * se) / se):.0%})")
    return out


# ----------------------------------------------------------------------------- 3b. folk rules as bets
def rule_bets(d):
    """Pre-registered folk betting rules. The DIRECTION of each rule is fixed by the folk theory, not chosen
    from the data. Moneyline at the vig-inclusive close; ATS at an assumed -110 (spread prices are not
    stored). Playoff CIs resample whole series."""
    rs = (d.season_type == "Regular Season").to_numpy()
    po = (d.season_type == "Playoffs").to_numpy()
    p = d.mkt_close.to_numpy()
    fav = np.where(p >= 0.5, 1, -1)
    g = lambda c: d[c].fillna(0).to_numpy()  # noqa: E731
    H = lambda c: g(f"home_{c}") == 1  # noqa: E731
    A = lambda c: g(f"away_{c}") == 1  # noqa: E731
    lw = H("last_week")
    zz = g("home_lost_prev")
    he, ae = g("home_elim") == 1, g("away_elim") == 1
    sh, sa = ~H("no_stakes") & A("no_stakes"), ~A("no_stakes") & H("no_stakes")
    lh, la = H("seed_locked") & ~A("seed_locked"), A("seed_locked") & ~H("seed_locked")
    th, ta = H("tank") & ~A("tank"), A("tank") & ~H("tank")
    rh, ra = H("in_race") & A("no_stakes"), A("in_race") & H("no_stakes")
    rules = [
        ("PO", "bet every playoff underdog", po, -fav),
        ("PO", "zig-zag: bet loser of previous series game", po & (zz != 0), zz),
        ("PO", "bet team facing elimination (not game 7)", po & (he ^ ae), np.where(he, 1, -1)),
        ("PO", "bet home team down 0-2 in game 3", po & (g("home_down02") == 1), np.ones(len(d))),
        ("RS", "last week: bet team with stakes vs no-stakes opp", rs & lw & (sh | sa), np.where(sh, 1, -1)),
        ("RS", "last week: bet against seed-locked team", rs & lw & (lh | la), np.where(lh, -1, 1)),
        ("RS", "bet against tank team (bottom-6, <=25 left)", rs & (th | ta), np.where(th, -1, 1)),
        ("RS", "bet team in race vs no-stakes opponent", rs & (rh | ra), np.where(rh, 1, -1)),
    ]
    bonf = 1 - 0.05 / len(rules)
    cl = series_id(d).to_numpy()
    periods = {"2018-22 hold-out": (d.season < "2022-23").to_numpy(), "2022-26 eval": (d.season >= "2022-23").to_numpy(),
               "all": np.ones(len(d), bool)}
    rows = []
    for fam, name, mask, side in rules:
        side = np.asarray(side, float)
        for per, pm in periods.items():
            m = mask & pm & np.isfinite(d.dec_home_close.to_numpy(float))
            if m.sum() == 0:
                continue
            s = d[m]
            prof = side_profit(s.home_win, side[m], s.dec_home_close.to_numpy(float), s.dec_away_close.to_numpy(float))
            bs = boot_cluster_mean(prof, cl[m], N_BOOT, 7) if fam == "PO" else boot_mean(prof, N_BOOT, 7)
            cov = side[m] * (s.margin + s.spread).to_numpy(float)
            okc = np.isfinite(cov)
            ats = np.where(cov[okc] > 0, 100 / 110, np.where(cov[okc] < 0, -1.0, 0.0))
            bsa = (boot_cluster_mean(ats, cl[m][okc], N_BOOT, 8) if fam == "PO" else boot_mean(ats, N_BOOT, 8))
            rows.append({"family": fam, "rule": name, "period": per, "bets": int(m.sum()),
                         "win_rate": float(np.mean(prof > 0)), "ml_roi": prof.mean(),
                         "ml_ci95": tuple(np.round(ci(bs), 3)), "ml_ci_bonf": tuple(np.round(ci(bs, bonf), 3)),
                         "ats_bets": int(okc.sum()), "ats_cover": float((cov[okc] > 0).sum() / max((cov[okc] != 0).sum(), 1)),
                         "ats_roi_110": ats.mean(), "ats_ci95": tuple(np.round(ci(bsa), 3)),
                         "ats_ci_bonf": tuple(np.round(ci(bsa, bonf), 3))})
    out = pd.DataFrame(rows)
    print(f"\n=== 3b. Pre-registered folk rules as flat 1u bets ({len(rules)} rules; Bonferroni level {bonf:.2%}). "
          "ML at the vig-inclusive close; ATS at an assumed -110 ===")
    print(out.round(4).to_string(index=False))
    a = out[out.period == "all"]
    print(f"rules with ML 95% CI > 0 (all seasons): {int(sum(c[0] > 0 for c in a.ml_ci95))}; Bonferroni: "
          f"{int(sum(c[0] > 0 for c in a.ml_ci_bonf))} | ATS 95% CI > 0: {int(sum(c[0] > 0 for c in a.ats_ci95))}; "
          f"Bonferroni: {int(sum(c[0] > 0 for c in a.ats_ci_bonf))}")
    return out


# ----------------------------------------------------------------------------- 4. walk-forward
def evaluate(d, mkt_col, test_seasons, label, first_train=None, n_var_total=None):
    t0 = time.time()
    need = 3 if len(test_seasons) == 4 else len(test_seasons)
    preds, infos, F = {}, {}, None
    for spec, offset in (("B_offset", True), ("A_recal", False)):
        pr, info, F = walk_forward(d, mkt_col, test_seasons, offset, first_train)
        for v, p in pr.items():
            preds[(spec, v)] = p
            infos[(spec, v)] = info[v]
    idx = preds[("A_recal", "market_recal")].index
    te = d.loc[idx]
    y = te.home_win.to_numpy(float)
    base = ll_vec(y, te[mkt_col].to_numpy())
    n_var = n_var_total or (len(preds) - 1)
    bonf = 1 - 0.05 / n_var
    rows = []
    for (spec, v), p in preds.items():
        p = p.loc[idx].to_numpy()
        delta = ll_vec(y, p) - base
        bs = boot_mean(delta, N_BOOT, seed=11)
        r = {"spec": spec, "variant": v, "family": FAMILY.get(v, "-"), "logloss": ll_vec(y, p).mean(),
             "delta": delta.mean(), "ci95": ci(bs), "ci_bonf": ci(bs, bonf)}
        for S in test_seasons:
            r[S] = delta[(te.season == S).to_numpy()].mean()
        r["n_neg"] = int(sum(r[S] < 0 for S in test_seasons))
        if v in VARIANTS:
            act = (F.loc[idx, VARIANTS[v]].to_numpy() != 0).any(1)
            r["n_active"] = int(act.sum())
            r["delta_active"] = delta[act].mean() if act.any() else np.nan
            bsa = boot_mean(delta[act], 2000, seed=12) if act.sum() > 1 else [np.nan]
            r["ci95_active"] = ci(bsa)
        r["survives"] = (bool(r["ci_bonf"][1] < 0 and r["n_neg"] >= need) if v != "market_recal" else None)
        rows.append(r)
    res = pd.DataFrame(rows)
    print(f"\n=== 4. {label}: market log-loss {base.mean():.5f} on {len(te)} games; per season "
          + ", ".join(f"{S} {base[(te.season == S).to_numpy()].mean():.5f}" for S in test_seasons)
          + f"; Bonferroni level {bonf:.4%} over {n_var} variants  [{time.time() - t0:.0f}s]")
    sh = res.copy()
    for c in ["delta", "delta_active"] + test_seasons:
        sh[c] = sh[c].map(lambda x: f"{x:+.5f}" if pd.notna(x) else "")
    for c in ["ci95", "ci_bonf", "ci95_active"]:
        sh[c] = sh[c].map(lambda t: f"[{t[0]:+.5f},{t[1]:+.5f}]" if isinstance(t, tuple) else "")
    sh["logloss"] = sh.logloss.map(lambda x: f"{x:.5f}")
    print(sh.drop(columns=["family"]).to_string(index=False))
    rv = res[res.variant != "market_recal"]
    best = rv.loc[rv.delta.idxmin()]
    print(f"[{label}] {len(rv)} variants | survivors: {rv[rv.survives == True][['spec', 'variant']].values.tolist()} | "  # noqa: E712
          f"best {best.spec}/{best.variant} {best.delta:+.5f} 95% [{best.ci95[0]:+.5f},{best.ci95[1]:+.5f}] "
          f"({best.n_neg}/{len(test_seasons)} seasons neg) | median {rv.delta.median():+.5f} | "
          f"95% CI < 0: {int(sum(c[1] < 0 for c in rv.ci95))}")
    print("fitted weights (per scaled unit) by test season, selected variants:")
    for k in [("B_offset", v) for v in ["po_fav", "po_slope", "zigzag", "elim_po", "nostakes_diff", "tank_diff",
                                        "locked_diff", "race_diff", "all_combined_l2", "all_combined_gbm"]] + \
             [("A_recal", "market_recal")]:
        if k in infos:
            print(f"  {k[0]}/{k[1]}: {infos[k]}")
    return res, preds, te


def betting(te, preds, label, odds_sets, thresholds=(0.0, 0.02, 0.04, 0.06)):
    rows = []
    y = te.home_win.to_numpy(float)
    for (spec, v), p in preds.items():
        p = p.loc[te.index].to_numpy()
        for book, (dh, da) in odds_sets.items():
            for thr in thresholds:
                prof = bet_profits(y, p, te[dh].to_numpy(float), te[da].to_numpy(float), thr)
                pr = prof[np.isfinite(prof)]
                r = {"model": label, "spec": spec, "variant": v, "odds": book, "thr": thr, "bets": len(pr)}
                if len(pr):
                    bs = boot_mean(pr, 2000, seed=21)
                    r.update(roi=pr.mean(), lo=ci(bs)[0], hi=ci(bs)[1])
                rows.append(r)
    out = pd.DataFrame(rows)
    for book in odds_sets:
        o = out[(out.odds == book) & (out.bets >= 30)]
        print(f"\n=== 5. Betting ({label}-line models at {book} odds, vig-inclusive; rules with >= 30 bets) ===")
        if len(o):
            print(o.sort_values("roi", ascending=False).head(12).round(4).to_string(index=False))
            print(f"  rules: {len(o)} | ROI>0: {int((o.roi > 0).sum())} | 95% CI lo>0: {int((o.lo > 0).sum())} | "
                  f"median ROI {o.roi.median():+.3f}")
    return out


# ----------------------------------------------------------------------------- 6. line movement
def line_move(d):
    s = d[d.mkt_open.notna() & d.mkt_close.notna()]
    F = game_features(s, "mkt_open")
    mv = logit(s.mkt_close) - logit(s.mkt_open)
    rows = []
    for c in RS_SINGLE + ["zigzag", "elim_po", "series_lead", "down02", "game7"]:
        x = F[c].to_numpy()
        act = x != 0
        if act.sum() < 10:
            continue
        Z = np.column_stack([np.ones(len(x)), x])
        b = np.linalg.lstsq(Z, mv, rcond=None)[0]
        e = mv - Z @ b
        ZtZi = np.linalg.inv(Z.T @ Z)
        cov = ZtZi @ (Z.T @ (Z * (e ** 2)[:, None])) @ ZtZi * len(x) / (len(x) - 2)
        rows.append({"feature": c, "n_active": int(act.sum()), "move_logit_per_unit": b[1], "t": b[1] / np.sqrt(cov[1, 1]),
                     "move_pp_at_50": 100 * b[1] / 4})
    out = pd.DataFrame(rows)
    print("\n=== 6. Line movement open -> close (logit, home perspective) on the context features, 2023-24..2025-26 ===")
    print(out.round(4).to_string(index=False))
    return out


def main():
    t0 = time.time()
    d = build()
    print(f"dataset: {len(d)} games, seasons {sorted(d.season.unique())}")
    print(d.groupby("season").agg(n=("home_win", "size"), close=("mkt_close", "count"), open=("mkt_open", "count"),
                                  playoffs=("season_type", lambda s: (s == "Playoffs").sum()),
                                  odds=("odds_src", lambda s: s.mode().iloc[0])).to_string())
    desc = descriptive(d)
    desc.to_csv(OUT / "slices.csv", index=False)
    pf = playoff_favourites(d)
    pf.to_csv(OUT / "playoff_favourites.csv", index=False)
    rb = rule_bets(d)
    rb.to_csv(OUT / "rule_bets.csv", index=False)

    n_var = 2 * len(VARIANTS) - 1  # spec B all variants + spec A without gbm
    res_c, preds_c, te_c = evaluate(d, "mkt_close", CLOSE_SEASONS, "Closing line", n_var_total=n_var)
    res_c.to_csv(OUT / "walkforward_close.csv", index=False)
    pd.DataFrame({f"{s}|{v}": p for (s, v), p in preds_c.items()}).assign(
        home_win=te_c.home_win, mkt_close=te_c.mkt_close).to_csv(CACHE / "preds_close.csv")
    res_o, preds_o, te_o = evaluate(d, "mkt_open", OPEN_SEASONS, "Opening line", n_var_total=n_var)
    res_o.to_csv(OUT / "walkforward_open.csv", index=False)

    b1 = betting(te_c, preds_c, "close", {"close": ("dec_home_close", "dec_away_close")})
    b2 = betting(te_o, preds_o, "open", {"open": ("dec_home_open", "dec_away_open"),
                                         "close": ("dec_home_close", "dec_away_close")})
    pd.concat([b1, b2]).to_csv(OUT / "betting_models.csv", index=False)

    lm = line_move(d)
    lm.to_csv(OUT / "line_move.csv", index=False)

    # 7. robustness: training only from 2021-22 (as in the other hypotheses)
    res_r, _, _ = evaluate(d, "mkt_close", CLOSE_SEASONS, "Closing line, training from 2021-22 only",
                           first_train="2021-22", n_var_total=n_var)
    res_r.to_csv(OUT / "walkforward_close_train2021.csv", index=False)
    # regular-season-only and playoffs-only views of the closing-line results (where the features act)
    for name, m in (("regular season", te_c.season_type == "Regular Season"),
                    ("playoffs+play-in", te_c.season_type != "Regular Season")):
        y = te_c.home_win.to_numpy(float)[m.to_numpy()]
        base = ll_vec(y, te_c.mkt_close.to_numpy()[m.to_numpy()])
        rows = []
        fam = "regular season" if name == "regular season" else "playoffs"
        for (spec, v), p in preds_c.items():
            if spec != "B_offset" or FAMILY.get(v) not in (fam, "all"):
                continue
            dl = ll_vec(y, p.loc[te_c.index].to_numpy()[m.to_numpy()]) - base
            bs = boot_mean(dl, 2000, seed=31)
            rows.append({"variant": v, "delta": dl.mean(), "ci95": tuple(np.round(ci(bs), 5)),
                         **{S: dl[(te_c.season[m] == S).to_numpy()].mean() for S in CLOSE_SEASONS}})
        print(f"\n=== 4b. Closing line, spec B, restricted to {name} ({m.sum()} games; market ll {base.mean():.5f}) ===")
        print(pd.DataFrame(rows).round(5).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s")


class _Tee:
    def __init__(self, *fs):
        self.fs = fs

    def write(self, x):
        for f in self.fs:
            f.write(x)

    def flush(self):
        for f in self.fs:
            f.flush()


if __name__ == "__main__":
    with open(OUT / "run_log.txt", "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(sys.__stdout__, fh)
        try:
            main()
        finally:
            sys.stdout = sys.__stdout__
