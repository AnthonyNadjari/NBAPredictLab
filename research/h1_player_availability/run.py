"""H1 - Player availability vs the betting market.

    python research/h1_player_availability/run.py            # uses caches
    python research/h1_player_availability/run.py --rebuild  # recompute features

Steps: fetch player logs (cached) -> build availability features (build.py) ->
walk-forward tests vs closing and opening line, line-move decomposition, upset
profile, flat-bet ROI at the real book odds. Results printed and saved to
research/h1_player_availability/results/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402
import fetch_players  # noqa: E402

RES = HERE / "results"
RES.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261004)
NBOOT = 2000
TEST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

# pre-registered variants: 3 value metrics x {oracle, previous-game proxy}
PRIMARY = {
    "oracle_min": ["d_miss_min_oracle"], "oracle_gs": ["d_miss_gs_oracle"], "oracle_oo": ["d_miss_oo_oracle"],
    "prev_min": ["d_miss_min_prev"], "prev_gs": ["d_miss_gs_prev"], "prev_oo": ["d_miss_oo_prev"],
}
# exploratory (reported, not used for the verdict; counted in the variant total)
EXPLORATORY = {
    "oracle_all3": ["d_miss_min_oracle", "d_miss_gs_oracle", "d_miss_oo_oracle"],
    "oracle_top2": ["d_top2_oracle"],
    "gs_decomposed": ["d_miss_gs_prev", "d_miss_gs_new", "d_miss_gs_return"],
}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def dec(am):
    am = np.asarray(am, float)
    return np.where(am < 0, 1 + 100 / -am, 1 + am / 100)


def boot_mean(x, alpha=0.05, n=NBOOT):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def load():
    fetch_players.main()
    g = build.build(force="--rebuild" in sys.argv)
    for c in [c for c in g.columns if c.startswith("home_") and c[5:].startswith(("miss_", "top2_", "top1_"))]:
        g["d_" + c[5:]] = g[c] - g["away_" + c[5:]]
    g["y"] = g.home_win.astype(float)
    return g


# ----------------------------------------------------------------------------- models
def fit_offset(tr, te, base, feats):
    """logit(p) = logit(market) + b . z  (slope on the market fixed at 1, no intercept):
    the cleanest 'does X shift the price' test - with no feature it IS the market."""
    if not feats:
        return te[base].to_numpy(), np.array([1.0])
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    Ztr, Zte = ((tr[feats] - mu) / sd).to_numpy(), ((te[feats] - mu) / sd).to_numpy()
    otr, y = logit(tr[base]), tr.y.to_numpy()

    def nll(b):
        eta = otr + Ztr @ b
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    b = minimize(nll, np.zeros(len(feats)), method="BFGS").x
    return 1 / (1 + np.exp(-(logit(te[base]) + Zte @ b))), np.r_[1.0, b]


def fit_predict(tr, te, base, feats, spec="lr"):
    if spec == "offset":
        return fit_offset(tr, te, base, feats)
    if not feats:
        Xtr, Xte = logit(tr[base])[:, None], logit(te[base])[:, None]
    else:
        mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
        Xtr = np.column_stack([logit(tr[base])] + [((tr[f] - mu[f]) / sd[f]).to_numpy() for f in feats])
        Xte = np.column_stack([logit(te[base])] + [((te[f] - mu[f]) / sd[f]).to_numpy() for f in feats])
    m = LogisticRegression(C=1e3, max_iter=5000).fit(Xtr, tr.y)
    return m.predict_proba(Xte)[:, 1], m.coef_[0]


def evaluate(g, base, feats, tests, scheme="wf", spec="lr"):
    """Return per-game frame for the test seasons: market prob, model prob, log-losses."""
    out, coefs = [], []
    d = g[g[base].notna()]
    for s in tests:
        te = d[d.season == s]
        tr = d[d.season < s] if scheme == "wf" else d[d.season != s]
        if len(tr) == 0 or len(te) == 0:
            continue
        p, c = fit_predict(tr, te, base, feats, spec)
        o = te[["game_id", "season", "y", base, "ml_home_close", "ml_away_close", "ml_home_open",
                "ml_away_open"]].copy()
        o["p_mkt"], o["p_model"] = te[base].to_numpy(), p
        o["ll_mkt"], o["ll_model"] = ll(o.y, o.p_mkt), ll(o.y, o.p_model)
        out.append(o)
        coefs.append(c)
    return pd.concat(out, ignore_index=True), np.array(coefs)


def summarize(o, alpha_bonf):
    diff = (o.ll_model - o.ll_mkt).to_numpy()
    m, lo, hi = boot_mean(diff)
    _, blo, bhi = boot_mean(diff, alpha=alpha_bonf)
    per = o.assign(dd=diff).groupby("season").dd.mean()
    return {"n": len(o), "ll_mkt": o.ll_mkt.mean(), "ll_model": o.ll_model.mean(),
            "delta": m, "ci_lo": lo, "ci_hi": hi, "bonf_lo": blo, "bonf_hi": bhi,
            "neg_seasons": int((per < 0).sum()), "n_seasons": len(per),
            **{f"d_{s}": v for s, v in per.items()}}


def roi(o, side_odds, thr):
    """Flat 1u bets at the real (vigged) odds where model prob - implied prob > thr."""
    oh, oa = dec(o[f"ml_home_{side_odds}"]), dec(o[f"ml_away_{side_odds}"])
    p, y = o.p_model.to_numpy(), o.y.to_numpy()
    ok = np.isfinite(oh) & np.isfinite(oa)
    eh, ea = p - 1 / oh, (1 - p) - 1 / oa
    bet_h, bet_a = ok & (eh > thr) & (eh >= ea), ok & (ea > thr) & (ea > eh)
    prof = np.concatenate([np.where(y[bet_h] == 1, oh[bet_h] - 1, -1.0),
                           np.where(y[bet_a] == 0, oa[bet_a] - 1, -1.0)])
    if len(prof) < 20:
        return {"bets": len(prof), "roi": np.nan, "lo": np.nan, "hi": np.nan}
    m, lo, hi = boot_mean(prof)
    return {"bets": len(prof), "roi": m, "lo": lo, "hi": hi}


# ----------------------------------------------------------------------------- analyses
def residual_tests(g):
    rows, rois, preds = [], [], {}
    k_bonf = 2 * len(PRIMARY)  # 6 features x 2 model specs per market baseline
    setups = [  # (label, market col, test seasons, scheme, odds used for ROI)
        ("close_wf", "mkt_close", TEST_SEASONS, "wf", "close"),
        ("open_wf", "mkt_open", TEST_SEASONS, "wf", "open"),          # only 2024-25, 2025-26 trainable
        ("open_loso", "mkt_open", ["2023-24", "2024-25", "2025-26"], "loso", "open"),
    ]
    for label, base, tests, scheme, odds in setups:
        for spec in ("lr", "offset"):
            if spec == "lr":
                recal, _ = evaluate(g, base, [], tests, scheme, spec)
                rows.append({"setup": label, "spec": spec, "variant": "market_refit_only", "kind": "control",
                             **summarize(recal, 0.05 / k_bonf)})
                for thr in (0.0, 0.02, 0.05):
                    rois.append({"setup": label, "spec": spec, "variant": "market_refit_only", "thr": thr,
                                 **roi(recal, odds, thr)})
            for kind, variants in (("primary", PRIMARY), ("exploratory", EXPLORATORY)):
                for name, feats in variants.items():
                    o, coefs = evaluate(g, base, feats, tests, scheme, spec)
                    r = summarize(o, 0.05 / k_bonf)
                    r.update({"setup": label, "spec": spec, "variant": name, "kind": kind,
                              "coef_feat_mean": float(coefs[:, 1:].mean()) if coefs.shape[1] > 1 else np.nan})
                    rows.append(r)
                    preds[(label, spec, name)] = o
                    if kind == "primary":
                        for thr in (0.0, 0.02, 0.05):
                            rois.append({"setup": label, "spec": spec, "variant": name, "thr": thr,
                                         **roi(o, odds, thr)})
    res = pd.DataFrame(rows)
    res["passes"] = (res.ci_hi < 0) & (res.neg_seasons >= np.minimum(3, res.n_seasons))
    res["passes_bonf"] = (res.bonf_hi < 0) & (res.neg_seasons >= np.minimum(3, res.n_seasons))
    return res, pd.DataFrame(rois), preds


def no_market_sanity(g):
    """Does the absence signal predict outcomes at all? Elo-only model vs Elo + absence."""
    d = g[g.elo_win_prob.notna()]
    rows = []
    for name, feats in PRIMARY.items():
        o, _ = evaluate(d, "elo_win_prob", feats, ["2023-24", "2024-25", "2025-26"], "wf")
        oe, _ = evaluate(d, "elo_win_prob", [], ["2023-24", "2024-25", "2025-26"], "wf")
        diff = o.ll_model.to_numpy() - oe.ll_model.to_numpy()
        m, lo, hi = boot_mean(diff)
        rows.append({"variant": name, "ll_elo_refit": oe.ll_model.mean(), "ll_elo_plus": o.ll_model.mean(),
                     "delta": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def line_moves(g):
    """How much of logit(close) - logit(open) do absences explain?"""
    d = g[g.mkt_open.notna() & g.has_rot].copy()
    d["move"] = logit(d.mkt_close) - logit(d.mkt_open)
    sets = {
        "oracle_gs": ["d_miss_gs_oracle"], "oracle_min": ["d_miss_min_oracle"], "oracle_oo": ["d_miss_oo_oracle"],
        "oracle_all3": ["d_miss_min_oracle", "d_miss_gs_oracle", "d_miss_oo_oracle"],
        "prev_gs": ["d_miss_gs_prev"], "prev_all3": ["d_miss_min_prev", "d_miss_gs_prev", "d_miss_oo_prev"],
        "gs_decomposed(prev,new,return)": ["d_miss_gs_prev", "d_miss_gs_new", "d_miss_gs_return"],
    }
    rows = []
    seasons = sorted(d.season.unique())
    for name, f in sets.items():
        lr = LinearRegression().fit(d[f], d.move)
        r2_in = lr.score(d[f], d.move)
        # leave-one-season-out out-of-sample R2
        sse = sst = 0.0
        for s in seasons:
            tr, te = d[d.season != s], d[d.season == s]
            pr = LinearRegression().fit(tr[f], tr.move).predict(te[f])
            sse += ((te.move - pr) ** 2).sum()
            sst += ((te.move - tr.move.mean()) ** 2).sum()
        rows.append({"features": name, "n": len(d), "r2_in": r2_in, "r2_loso": 1 - sse / sst,
                     "coefs": np.round(lr.coef_, 4).tolist()})
    # direction: when a team's oracle missing value rises, does the line move against it?
    big = d[np.abs(d.d_miss_gs_new) >= 5]
    agree = (np.sign(-big.d_miss_gs_new) == np.sign(big.move)).mean()
    return pd.DataFrame(rows), {"n_big_new_absence_games": int(len(big)), "move_direction_agrees": float(agree),
                                "sd_move_logit": float(d.move.std())}


def upsets(g):
    """Step 4: favourites missing a top-2 rotation player (by prior GameScore value)."""
    d = g[g.season.isin(TEST_SEASONS) & g.has_rot].copy()
    d["fav_home"] = d.mkt_close >= 0.5
    d["fav_p"] = np.where(d.fav_home, d.mkt_close, 1 - d.mkt_close)
    d["fav_p_open"] = np.where(d.fav_home, d.mkt_open, 1 - d.mkt_open)
    d["fav_won"] = np.where(d.fav_home, d.y, 1 - d.y)
    out = {}
    for kind in ("oracle", "prev"):
        fav_t2 = np.where(d.fav_home, d[f"home_top2_{kind}"], d[f"away_top2_{kind}"]) > 0
        dog_t2 = np.where(d.fav_home, d[f"away_top2_{kind}"], d[f"home_top2_{kind}"]) > 0
        up = d.fav_won == 0
        res = {"games": int(len(d)), "upsets": int(up.sum()),
               "share_all_games_fav_missing_top2": float(fav_t2.mean()),
               "share_upsets_fav_missing_top2": float(fav_t2[up].mean()),
               "share_fav_wins_fav_missing_top2": float(fav_t2[~up].mean()),
               "share_upsets_dog_missing_top2": float(dog_t2[up].mean()),
               "share_all_games_dog_missing_top2": float(dog_t2.mean())}
        for lab, m in (("fav_missing_top2", fav_t2), ("dog_missing_top2", dog_t2), ("neither", ~fav_t2 & ~dog_t2)):
            s = d[m]
            sd = np.sqrt((s.fav_p * (1 - s.fav_p)).sum()) / len(s)
            res[lab] = {"n": int(len(s)), "fav_won": float(s.fav_won.mean()), "priced_close": float(s.fav_p.mean()),
                        "gap_pts_close": float(100 * (s.fav_won.mean() - s.fav_p.mean())),
                        "z_close": float((s.fav_won.mean() - s.fav_p.mean()) / sd)}
            so = s[s.fav_p_open.notna()]
            if len(so):
                sdo = np.sqrt((so.fav_p_open * (1 - so.fav_p_open)).sum()) / len(so)
                res[lab].update({"n_open": int(len(so)), "fav_won_open_subset": float(so.fav_won.mean()),
                                 "priced_open": float(so.fav_p_open.mean()),
                                 "gap_pts_open": float(100 * (so.fav_won.mean() - so.fav_p_open.mean())),
                                 "z_open": float((so.fav_won.mean() - so.fav_p_open.mean()) / sdo)})
        out[kind] = res
    return out


def segments(g):
    """Team-perspective calibration in absence segments (no conditioning on who is favourite,
    so the opening-line numbers are not biased by using the closing line to pick sides)."""
    d = g[g.season.isin(TEST_SEASONS) & g.has_rot]
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        sgn = 1 if side == "home" else 0
        t = pd.DataFrame({
            "win": d.y if sgn else 1 - d.y,
            "p_close": d.mkt_close if sgn else 1 - d.mkt_close,
            "p_open": d.mkt_open if sgn else 1 - d.mkt_open,
        })
        c = lambda k, who=side: d[f"{who}_{k}"] > 0
        t["top2_oracle_only"] = c("top2_oracle") & ~c("top2_oracle", opp)
        t["top1_oracle_only"] = c("top1_oracle") & ~c("top1_oracle", opp)
        t["top2_new_only"] = c("top2_new") & ~c("top2_new", opp)
        t["top2_out_prev_and_today"] = c("top2_oracle") & c("top2_prev")
        t["top2_returns_today"] = c("top2_return")
        t["top2_prev_proxy_only"] = c("top2_prev") & ~c("top2_prev", opp)
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    labels = {
        "top2_oracle_only": "team missing a top-2 player (oracle), opponent not",
        "top1_oracle_only": "team missing its top-1 player (oracle), opponent not",
        "top2_new_only": "team top-2 NEW absence (played last game), opponent no new one",
        "top2_out_prev_and_today": "team top-2 out last game AND today (carried over)",
        "top2_returns_today": "team top-2 out last game, RETURNS today",
        "top2_prev_proxy_only": "team missing top-2 per prev-game proxy, opponent not",
    }
    out = []
    for k, name in labels.items():
        s = t[t[k]]
        so = s[s.p_open.notna()]
        gap = s.win.mean() - s.p_close.mean()
        sd = np.sqrt((s.p_close * (1 - s.p_close)).sum()) / len(s)
        gapo = so.win.mean() - so.p_open.mean()
        sdo = np.sqrt((so.p_open * (1 - so.p_open)).sum()) / len(so)
        out.append({"segment": name, "n": len(s), "won": s.win.mean(), "priced_close": s.p_close.mean(),
                    "gap_close_pts": 100 * gap, "z_close": gap / sd, "n_open": len(so),
                    "won_open_subset": so.win.mean(), "priced_open": so.p_open.mean(),
                    "gap_open_pts": 100 * gapo, "z_open": gapo / sdo,
                    "move_open_to_close_pts": 100 * (so.p_close.mean() - so.p_open.mean())})
    return pd.DataFrame(out)


def main():
    g = load()
    print(f"games: {len(g)}  by season:\n{g.groupby('season').size().to_string()}")
    print(f"rotation players per team-game ~{(g.home_n_rot.mean() + g.away_n_rot.mean()) / 2:.1f}; "
          f"oracle-absent per team-game {(g.home_miss_min_oracle > 0).mean():.1%} have >=1 absence")

    res, rois, preds = residual_tests(g)
    pd.set_option("display.width", 250)
    cols = ["setup", "spec", "variant", "kind", "n", "ll_mkt", "ll_model", "delta", "ci_lo", "ci_hi", "bonf_lo", "bonf_hi",
            "neg_seasons", "n_seasons", "d_2022-23", "d_2023-24", "d_2024-25", "d_2025-26", "coef_feat_mean",
            "passes", "passes_bonf"]
    res = res[[c for c in cols if c in res.columns]]
    print("\n=== Log-loss vs market (delta = model - market, negative = better) ===")
    print(res.round(5).to_string(index=False))
    res.to_csv(RES / "residual_tests.csv", index=False)
    print("\n=== Flat-bet ROI at real book odds (incl. vig) ===")
    print(rois.round(4).to_string(index=False))
    rois.to_csv(RES / "roi.csv", index=False)

    san = no_market_sanity(g)
    print("\n=== Sanity: absence signal WITHOUT the market (Elo + feature vs Elo, WF 2023-24..2025-26) ===")
    print(san.round(5).to_string(index=False))
    san.to_csv(RES / "no_market_sanity.csv", index=False)

    lm, lm_extra = line_moves(g)
    print("\n=== Line moves: logit(close) - logit(open) explained by absences (2023-24..2025-26) ===")
    print(lm.round(4).to_string(index=False))
    print(lm_extra)
    lm.to_csv(RES / "line_moves.csv", index=False)

    seg = segments(g)
    print("\n=== Segments (team perspective): win rate vs price, z = gap / binomial sd; open = 2023-24..2025-26 ===")
    print(seg.round(3).to_string(index=False))
    seg.to_csv(RES / "segments.csv", index=False)

    up = upsets(g)
    print("\n=== Upsets and favourites missing a top-2 player ===")
    print(json.dumps(up, indent=1))
    (RES / "upsets.json").write_text(json.dumps({"upsets": up, "line_move_extra": lm_extra}, indent=1))


if __name__ == "__main__":
    main()
