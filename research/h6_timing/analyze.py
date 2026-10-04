"""H6 timing analysis: how accurate is the market price as a function of time before tip?

Sections (printed + saved to research/h6_timing/results/):
  A coverage            matched Kalshi x ESPN games, quote availability per snapshot
  B validation          Kalshi pre-tip price vs ESPN main-book close / open on the same games
  C horizon curve       log-loss / Brier / accuracy of the Kalshi price h hours before tip
  D publish policies    what the bot would publish if it ran at UTC time X (09:00 now)
  E sportsbook bounds   ESPN open vs close (2024-25, 2025-26 and all test seasons)
  F when prices move    remaining squared move to the close, by UTC clock time (evening games)
  G protocol test       does the Kalshi price add information beyond the ESPN closing line?
                        (walk-forward + betting at the actual ESPN closing moneyline)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLOCK_UTC, HOURS, RES, am_to_dec, am_to_p, boot_diff, dataset, espn_games,  # noqa: E402
                    ll_vec, logit, metrics)

NB = 2000
SUMMARY = {}


def power_devig(h_ml, a_ml):
    """De-vig by finding k with ph^k + pa^k = 1 (power method; removes more vig from the longshot)."""
    from scipy.optimize import brentq
    ph, pa = am_to_p(h_ml), am_to_p(a_ml)
    out = np.full(len(ph), np.nan)
    for i, (x, y) in enumerate(zip(ph, pa)):
        if np.isnan(x) or np.isnan(y) or x + y <= 1:
            out[i] = x / (x + y) if not (np.isnan(x) or np.isnan(y)) else np.nan
            continue
        k = brentq(lambda k: x ** k + y ** k - 1, 1, 10)
        out[i] = x ** k
    return out


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string())


def section_a(d):
    t = d.groupby(["season", "season_type"]).agg(games=("event_id", "size")).reset_index()
    cols = ["k_prev_utc09", "k_utc09", "k_utc17", "k_utc21", "k_h24", "k_h12", "k_h4", "k_h1", "k_close"]
    cov = pd.DataFrame({c: [d[c].notna().mean()] for c in cols}, index=["share with valid quote"]).T
    show("A. Coverage: matched Kalshi x ESPN games", t)
    show("A. Share of games with a valid Kalshi quote (spread <= 10c)", cov.round(3))
    SUMMARY["n_games"] = int(len(d))
    SUMMARY["n_by_season"] = d.groupby("season").size().to_dict()
    return t


def section_b(d):
    rows = []
    s = d[d.k_close.notna() & d.mkt_close.notna()]
    for name, col in (("Kalshi pre-tip (T-1min)", "k_close"), ("ESPN close (main book)", "mkt_close")):
        rows.append({"price": name, **metrics(s.home_win, s[col])})
    so = s[s.mkt_open.notna()]
    rows.append({"price": "ESPN open (same games, subset)", **metrics(so.home_win, so.mkt_open)})
    rows.append({"price": "Kalshi pre-tip (same subset)", **metrics(so.home_win, so.k_close)})
    t = pd.DataFrame(rows)
    m, lo, hi, _ = boot_diff(ll_vec(s.home_win, s.k_close), ll_vec(s.home_win, s.mkt_close), NB)
    corr = np.corrcoef(logit(s.k_close), logit(s.mkt_close))[0, 1]
    mad = np.abs(s.k_close - s.mkt_close).mean()
    show("B. Validation: Kalshi vs ESPN on the same games", t.round(5))
    print(f"Kalshi close - ESPN close log-loss: {m:+.5f} [{lo:+.5f}, {hi:+.5f}]; corr(logit) {corr:.3f}; "
          f"mean |diff| {mad * 100:.2f} pts")
    SUMMARY["kalshi_vs_espn_close"] = {"delta": m, "ci": [lo, hi], "corr_logit": corr, "mean_abs_diff_pts": mad * 100}
    # robustness: ESPN close de-vigged with the power method instead of multiplicative normalisation
    pw = power_devig(s.home_ml_close, s.away_ml_close)
    m2, lo2, hi2, _ = boot_diff(ll_vec(s.home_win, s.k_close), ll_vec(s.home_win, pw), NB)
    big = (np.abs(s.k_close - s.mkt_close) > 0.15).sum()
    print(f"vs ESPN close power-de-vigged: {m2:+.5f} [{lo2:+.5f}, {hi2:+.5f}] | games with |Kalshi-ESPN| > 15 pts "
          f"(possible in-play contamination): {big}")
    SUMMARY["kalshi_vs_espn_close_power"] = {"delta": m2, "ci": [lo2, hi2], "n_big_gaps": int(big)}
    t.to_csv(RES / "b_validation.csv", index=False)
    # when were ESPN's 'open' and 'close' captured? distance to the Kalshi price at each horizon
    snaps = [("D-1 09:00 UTC", "k_prev_utc09")] + [(f"T-{h:g}h", f"k_h{h:g}") for h in HOURS] + [("T-1min", "k_close")]
    cols = [c for _, c in snaps]
    s2 = d[d[cols].notna().all(axis=1) & d.mkt_open.notna() & d.mkt_close.notna()]
    rows = []
    for label, c in snaps:
        rows.append({"kalshi_snapshot": label,
                     "espn_open_mean_abs_diff_pts": 100 * (s2.mkt_open - s2[c]).abs().mean(),
                     "espn_close_mean_abs_diff_pts": 100 * (s2.mkt_close - s2[c]).abs().mean()})
    t2 = pd.DataFrame(rows)
    show(f"B2. Which Kalshi snapshot does ESPN's open / close resemble most? (n={len(s2)})", t2.round(3))
    t2.to_csv(RES / "b2_espn_snapshot_timing.csv", index=False)
    return t


def section_c(d):
    snaps = [("D-1 09:00 UTC", "k_prev_utc09")] + [(f"T-{h:g}h", f"k_h{h:g}") for h in HOURS] + [("T-1min", "k_close")]
    out = []
    for panel_name, cols_req in (("all with quote", None), ("balanced <=24h", [c for _, c in snaps[3:]])):
        for label, col in snaps:
            if cols_req is not None and col not in cols_req:
                continue
            s = d[d[col].notna() & d.k_close.notna()]
            if cols_req is not None:
                s = s[s[cols_req].notna().all(axis=1)]
            if len(s) < 50:
                continue
            mt = metrics(s.home_win, s[col])
            m, lo, hi, _ = boot_diff(ll_vec(s.home_win, s[col]), ll_vec(s.home_win, s.k_close), NB)
            out.append({"panel": panel_name, "snapshot": label, **mt, "d_ll_vs_close": m, "ci_lo": lo, "ci_hi": hi,
                        "flip_vs_close": ((s[col] > .5) != (s.k_close > .5)).mean()})
    t = pd.DataFrame(out)
    show("C. Horizon curve: Kalshi price h hours before scheduled tip (lower log-loss = better)", t.round(4))
    t.to_csv(RES / "c_horizon_curve.csv", index=False)
    # per season, balanced panel
    ps = []
    bal = d[d[[c for _, c in snaps[3:]]].notna().all(axis=1)]
    for season, s in bal.groupby("season"):
        for label, col in [("T-24h", "k_h24"), ("T-12h", "k_h12"), ("T-4h", "k_h4"), ("T-1h", "k_h1"), ("T-1min", "k_close")]:
            ps.append({"season": season, "snapshot": label, **metrics(s.home_win, s[col])})
    show("C. Horizon curve by season (balanced panel)", pd.DataFrame(ps).round(4))
    pd.DataFrame(ps).to_csv(RES / "c_horizon_by_season.csv", index=False)
    return t


LEAD_MIN = 30  # a run's price is only usable for a game if the run finishes >= 30 min before tip


def policy_prices(d, runs, lead_min=LEAD_MIN):
    """Price the bot would publish if it ran at each UTC hour in `runs` (hours on the ET game date;
    24+ = after midnight UTC). Each game gets the latest run that is >= lead_min before tip and has a
    valid quote; the 09:00 UTC run is always available (all games tip after 09:00 UTC)."""
    p = d.k_utc09.copy()
    day = pd.to_datetime(d.game_date).dt.tz_localize("UTC")
    for x in sorted(runs):
        if x == 9:
            continue
        ok = (d.tip - (day + pd.Timedelta(hours=x)) >= pd.Timedelta(minutes=lead_min)) & d[f"k_utc{x:02d}"].notna()
        p = p.where(~ok, d[f"k_utc{x:02d}"])
    return p


POLICIES = {
    "09:00 UTC only (current main run)": [9],
    "09 + 12 UTC": [9, 12], "09 + 15 UTC": [9, 15], "09 + 17 UTC": [9, 17], "09 + 18 UTC": [9, 18],
    "09 + 19 UTC": [9, 19], "09 + 20 UTC": [9, 20], "09 + 21 UTC (current refresh)": [9, 21],
    "09 + 22 UTC": [9, 22], "09 + 23 UTC": [9, 23],
    "09 + 17 + 22 UTC": [9, 17, 22], "09 + 21 + 23 UTC": [9, 21, 23], "09 + 18 + 22 + 00 UTC": [9, 18, 22, 24],
    "hourly 09..02 UTC (each game: last whole hour >= 30 min before tip)": list(range(9, 27)),
}


def section_d(d):
    base = d[d.k_utc09.notna() & d.k_close.notna() & d["k_h0.5"].notna() & d.k_h1.notna()]
    y = base.home_win.to_numpy()
    p09 = base.k_utc09.to_numpy()
    ll09 = ll_vec(y, p09)
    pols = [("D-1 09:00 UTC (tomorrow's games, current)", base.k_prev_utc09)]
    pols += [(k, policy_prices(base, v)) for k, v in POLICIES.items()]
    pols += [("per game T-1h (staggered threads)", base.k_h1), ("per game T-30min (staggered threads)", base["k_h0.5"]),
             ("pre-tip T-1min (upper bound, not publishable)", base.k_close)]
    rows = []
    zb = float(norm.ppf(1 - 0.025 / len(pols)))  # Bonferroni over the policies compared
    for name, p in pols:
        p = np.asarray(p, float)
        mask = ~np.isnan(p)
        mt = metrics(y[mask], p[mask])
        m, lo, hi, se = boot_diff(ll_vec(y[mask], p[mask]), ll09[mask], NB)
        late = (~np.isclose(p, p09)).mean() if name in POLICIES else np.nan
        flip = mask & ((p > .5) != (p09 > .5))
        later_right = float(((p[flip] > .5) == (y[flip] == 1)).mean()) if flip.any() else np.nan
        net_per_season = 1230 * (((p[flip] > .5) == (y[flip] == 1)).sum() - ((p09[flip] > .5) == (y[flip] == 1)).sum()) / mask.sum()
        rows.append({"policy": name, **mt, "d_ll_vs_09": m, "ci_lo": lo, "ci_hi": hi,
                     "bonf_lo": m - zb * se, "bonf_hi": m + zb * se,
                     "d_brier_vs_09": mt["brier"] - metrics(y[mask], p09[mask])["brier"],
                     "d_acc_vs_09_pts": 100 * (mt["accuracy"] - metrics(y[mask], p09[mask])["accuracy"]),
                     "picks_changed_vs_09": float(((p[mask] > .5) != (p09[mask] > .5)).mean()),
                     "flips_later_pick_right": later_right, "net_correct_picks_per_1230": net_per_season,
                     "share_games_price_changed": late})
    t = pd.DataFrame(rows)
    full = t.loc[t.policy.str.startswith("pre-tip"), "d_ll_vs_09"].iloc[0]
    t["share_of_max_gain"] = t.d_ll_vs_09 / full
    show(f"D. Publish policies (n={len(base)} games; delta vs 09:00 UTC; negative = better; lead {LEAD_MIN} min)",
         t.round(4))
    t.to_csv(RES / "d_policies.csv", index=False)
    ps = []
    for season, s in base.groupby("season"):
        yy = s.home_win.to_numpy()
        ll0 = metrics(yy, s.k_utc09)["logloss"]
        for name, p in (("09:00 UTC", s.k_utc09), ("09+17 UTC", policy_prices(s, [9, 17])),
                        ("09+21 UTC", policy_prices(s, [9, 21])), ("09+22 UTC", policy_prices(s, [9, 22])),
                        ("hourly", policy_prices(s, list(range(9, 27)))), ("T-30min", s["k_h0.5"]),
                        ("T-1min", s.k_close)):
            mt = metrics(yy, p)
            ps.append({"season": season, "policy": name, **mt, "d_ll_vs_09": mt["logloss"] - ll0})
    show("D. Policies by season", pd.DataFrame(ps).round(4))
    pd.DataFrame(ps).to_csv(RES / "d_policies_by_season.csv", index=False)
    tips = base.tip.dt.hour + base.tip.dt.minute / 60
    hour_bins = np.where(tips < 12, tips + 24, tips)
    dist = pd.Series(np.floor(hour_bins)).value_counts(normalize=True).sort_index()
    dist.index = [f"{int(h) % 24:02d}:00" for h in dist.index]
    print()
    print("Scheduled tip-off hour (UTC), share of games:")
    print(dist.round(3).to_string())
    dist.to_csv(RES / "d_tip_hour_distribution.csv")
    SUMMARY["policies"] = t.set_index("policy")[["n", "logloss", "brier", "accuracy", "d_ll_vs_09", "ci_lo", "ci_hi",
                                                 "bonf_lo", "bonf_hi", "picks_changed_vs_09", "flips_later_pick_right", "net_correct_picks_per_1230",
                                                 "share_of_max_gain"]].to_dict(orient="index")
    return t


def section_e():
    o = espn_games()
    rows = []
    for name, s in (("2024-25 + 2025-26", o[o.season.isin(["2024-25", "2025-26"])]),
                    ("2023-24 .. 2025-26", o[o.season.isin(["2023-24", "2024-25", "2025-26"])])):
        s = s[s.mkt_open.notna() & s.mkt_close.notna()]
        mo, mc = metrics(s.home_win, s.mkt_open), metrics(s.home_win, s.mkt_close)
        m, lo, hi, _ = boot_diff(ll_vec(s.home_win, s.mkt_close), ll_vec(s.home_win, s.mkt_open), NB)
        rows.append({"sample": name, "n": len(s), "ll_open": mo["logloss"], "ll_close": mc["logloss"],
                     "d_close_minus_open": m, "ci_lo": lo, "ci_hi": hi, "acc_open": mo["accuracy"],
                     "acc_close": mc["accuracy"], "picks_flipped": ((s.mkt_open > .5) != (s.mkt_close > .5)).mean()})
    for season, s in o[o.mkt_open.notna() & o.mkt_close.notna()].groupby("season"):
        mo, mc = metrics(s.home_win, s.mkt_open), metrics(s.home_win, s.mkt_close)
        rows.append({"sample": season, "n": len(s), "ll_open": mo["logloss"], "ll_close": mc["logloss"],
                     "d_close_minus_open": mc["logloss"] - mo["logloss"], "acc_open": mo["accuracy"],
                     "acc_close": mc["accuracy"], "picks_flipped": ((s.mkt_open > .5) != (s.mkt_close > .5)).mean()})
    t = pd.DataFrame(rows)
    show("E. Sportsbook bounds: ESPN main-book open vs close", t.round(5))
    t.to_csv(RES / "e_open_vs_close.csv", index=False)
    SUMMARY["espn_open_vs_close"] = t.iloc[0].to_dict()
    return t


def section_e2(d):
    """Opening-line baseline: the bot's 09:00 UTC price and the 09+21 UTC policy vs the ESPN open (same games)."""
    s = d[d.mkt_open.notna() & d.k_utc09.notna() & d.k_close.notna()].copy()
    s["p0921"] = policy_prices(s, [9, 21])
    rows = []
    for season, g in list(s.groupby("season")) + [("all", s)]:
        y = g.home_win
        for name, col in (("Kalshi 09:00 UTC", "k_utc09"), ("09 + 21 UTC policy", "p0921"),
                          ("ESPN close", "mkt_close")):
            m, lo, hi, _ = boot_diff(ll_vec(y, g[col]), ll_vec(y, g.mkt_open), NB)
            rows.append({"season": season, "price": name, "n": len(g), "ll_espn_open": ll_vec(y, g.mkt_open).mean(),
                         "ll_price": ll_vec(y, g[col]).mean(), "delta_vs_open": m, "ci_lo": lo, "ci_hi": hi})
    t = pd.DataFrame(rows)
    show("E2. Versus the ESPN opening line (same games; negative = better than open)", t.round(5))
    t.to_csv(RES / "e2_vs_open.csv", index=False)
    SUMMARY["vs_open"] = t.to_dict(orient="records")
    return t


def section_f(d):
    ev = d[d.tip.dt.hour.isin([23, 0, 1, 2]) & d.k_close.notna()]
    rows = []
    for x in [9, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
        c = f"k_utc{x:02d}"
        s = ev[ev[c].notna()]
        rows.append({"utc": f"{x:02d}:00", "n": len(s), "mean_abs_move_to_close_pts": 100 * (s[c] - s.k_close).abs().mean(),
                     "mean_sq_logit_move_to_close": ((logit(s[c]) - logit(s.k_close)) ** 2).mean()})
    for h in (1, 0.5, 0.25):
        s = ev[ev[f"k_h{h:g}"].notna()]
        rows.append({"utc": f"T-{h:g}h", "n": len(s),
                     "mean_abs_move_to_close_pts": 100 * (s[f"k_h{h:g}"] - s.k_close).abs().mean(),
                     "mean_sq_logit_move_to_close": ((logit(s[f"k_h{h:g}"]) - logit(s.k_close)) ** 2).mean()})
    t = pd.DataFrame(rows)
    t["share_of_09utc_remaining"] = t.mean_sq_logit_move_to_close / t.mean_sq_logit_move_to_close.iloc[0]
    show("F. Evening games (tip 23:00-02:59 UTC): price movement still to come, by UTC clock time", t.round(4))
    t.to_csv(RES / "f_move_remaining.csv", index=False)
    SUMMARY["remaining_move"] = t.set_index("utc").share_of_09utc_remaining.round(3).to_dict()
    return t


def _fit_predict(Xtr, ytr, Xte):
    m = LogisticRegression(C=1e4, max_iter=5000).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


VARIANTS = {
    "espn_close + kalshi_close": lambda s: np.c_[logit(s.mkt_close), logit(s.k_close)],
    "espn_close + kalshi_late_move(09utc->tip)": lambda s: np.c_[logit(s.mkt_close), logit(s.k_close) - logit(s.k_utc09)],
    "kalshi_close alone (recalibrated)": lambda s: np.c_[logit(s.k_close)],
}
# unfitted: average of the two pre-tip prices in logit space (no parameters, no leakage)
AVG = "avg(logit espn_close, logit kalshi_close), unfitted"


def section_g(d):
    """Protocol test. Seasons with Kalshi data: 2024-25 (play-in + playoffs only) and 2025-26.

    (1) season walk-forward: fit on 2024-25, predict 2025-26 (the only possible split);
    (2) monthly expanding window inside the data (fit on all earlier months, predict month m)."""
    s = d[d.k_close.notna() & d.mkt_close.notna() & d.k_utc09.notna()].copy()
    s["month"] = pd.to_datetime(s.game_date).dt.to_period("M")
    out, per, monthly_signs = [], [], {}
    K = len(VARIANTS) * 2 + 2
    z = float(norm.ppf(1 - 0.025 / K))  # two-sided Bonferroni over all log-loss variants

    def add(name, scheme, y, p, b, months=None):
        m, lo, hi, se = boot_diff(ll_vec(y, p), ll_vec(y, b), NB)
        row = {"variant": name, "scheme": scheme, "n": len(y), "ll_model": ll_vec(y, p).mean(),
               "ll_espn_close": ll_vec(y, b).mean(), "delta": m, "ci_lo": lo, "ci_hi": hi,
               "bonf_lo": m - z * se, "bonf_hi": m + z * se, "K": K}
        if months is not None:
            dm = pd.Series(ll_vec(y, p) - ll_vec(y, b)).groupby(np.asarray(months)).mean()
            row["months_negative"] = f"{int((dm < 0).sum())}/{len(dm)}"
            monthly_signs[f"{name} | {scheme}"] = dm.round(5).to_dict()
        out.append(row)

    for name, fx in VARIANTS.items():
        for scheme in ("season", "monthly"):
            preds, ys, base, mon = [], [], [], []
            if scheme == "season":
                splits = [(s[s.season == "2024-25"], s[s.season == "2025-26"], "2025-26")]
            else:
                months = sorted(s.month.unique())
                splits = [(s[s.month < m], s[s.month == m], str(m)) for m in months if (s.month < m).sum() >= 150]
            for tr, te, lab in splits:
                if len(tr) < 50 or len(te) == 0:
                    continue
                p = _fit_predict(fx(tr), tr.home_win, fx(te))
                preds.append(p)
                ys.append(te.home_win.to_numpy())
                base.append(te.mkt_close.to_numpy())
                mon.append(te.month.astype(str).to_numpy())
                if scheme == "season":
                    per.append({"variant": name, "test": lab, "n": len(te),
                                "delta_vs_espn_close": ll_vec(te.home_win, p).mean() - ll_vec(te.home_win, te.mkt_close).mean()})
            if preds:
                add(name, scheme, np.concatenate(ys), np.concatenate(preds), np.concatenate(base), np.concatenate(mon))
    # unfitted variants on every game with both prices (no training, so no walk-forward needed)
    avg = 1 / (1 + np.exp(-(logit(s.mkt_close) + logit(s.k_close)) / 2))
    for name, p in (("kalshi_close raw (unfitted)", s.k_close), (AVG, avg)):
        add(name, "unfitted, all games", s.home_win.to_numpy(), np.asarray(p), s.mkt_close.to_numpy(),
            s.month.astype(str).to_numpy())
        for season, g in s.groupby("season"):
            pg = np.asarray(p)[s.season.to_numpy() == season]
            per.append({"variant": name, "test": season, "n": len(g),
                        "delta_vs_espn_close": ll_vec(g.home_win, pg).mean() - ll_vec(g.home_win, g.mkt_close).mean()})
    t = pd.DataFrame(out)
    show("G. Does Kalshi add information beyond the ESPN closing line? (delta < 0 = better than close)", t.round(5))
    t.to_csv(RES / "g_protocol_logloss.csv", index=False)

    # betting at the actual ESPN closing moneyline (incl. vig) when Kalshi's pre-tip price is higher
    s["imp_h"], s["imp_a"] = am_to_p(s.home_ml_close), am_to_p(s.away_ml_close)
    s["dec_h"], s["dec_a"] = am_to_dec(s.home_ml_close), am_to_dec(s.away_ml_close)
    s = s[s.imp_h.notna() & s.imp_a.notna()]
    bets = []
    for src, col in (("kalshi T-1min", "k_close"), ("kalshi 09:00 UTC", "k_utc09")):
        for thr in (0.0, 0.02, 0.04):
            eh = s[col] - s.imp_h
            ea = (1 - s[col]) - s.imp_a
            bh, ba = eh > thr, (ea > thr) & ~(eh > thr)
            pnl = np.concatenate([np.where(s.home_win[bh] == 1, s.dec_h[bh] - 1, -1),
                                  np.where(s.home_win[ba] == 0, s.dec_a[ba] - 1, -1)])
            if len(pnl) < 20:
                bets.append({"signal": src, "threshold": thr, "bets": len(pnl)})
                continue
            rng = np.random.default_rng(1)
            bs = pnl[rng.integers(0, len(pnl), (NB, len(pnl)))].mean(axis=1)
            a = 100 * 0.05 / 6 / 2  # Bonferroni over the 6 betting rules
            bets.append({"signal": src, "threshold": thr, "bets": len(pnl), "roi": pnl.mean(),
                         "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                         "bonf_lo": np.percentile(bs, a), "bonf_hi": np.percentile(bs, 100 - a)})
    b = pd.DataFrame(bets)
    show("G. Betting at the ESPN closing moneyline (incl. vig) when Kalshi says the side is underpriced", b.round(4))
    b.to_csv(RES / "g_betting.csv", index=False)
    SUMMARY["protocol"] = t.to_dict(orient="records")
    SUMMARY["betting"] = b.to_dict(orient="records")
    SUMMARY["protocol_per_season"] = per
    SUMMARY["protocol_monthly"] = monthly_signs
    show("G. Per-season deltas vs ESPN close", pd.DataFrame(per).round(5))
    return t, b


def main():
    d = dataset(force="--rebuild" in sys.argv)
    section_a(d)
    section_b(d)
    section_c(d)
    section_d(d)
    section_e()
    section_e2(d)
    section_f(d)
    section_g(d)
    (RES / "summary.json").write_text(json.dumps(SUMMARY, indent=1, default=str))


if __name__ == "__main__":
    main()
