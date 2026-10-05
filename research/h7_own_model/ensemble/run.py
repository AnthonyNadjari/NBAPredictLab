"""H7 / ensemble: stack the three odds-free model families and score them against the market.

    python research/h7_own_model/ensemble/run.py            # ~20 s, reads the families' preds.csv
    python research/h7_own_model/ensemble/run.py --rerun    # first re-runs the 3 family scripts
                                                            # in parallel (~6 min), then stacks

Inputs (local, no network):
  ../player_impact/preds.csv        pregame_plus_team   (deployable best of the player family)
  ../margin_ratings/preds.csv       margin_full_avail   (deployable best of the margin family)
  ../margin_ratings/preds_team_only.csv  margin_full    (same, no player data)
  ../elo_plus/preds.csv             elo_plus            (deployable best of the Elo family)
  research/data/games_features_odds.csv  pre-game team features -> current production fallback
                                          (research/backtest.py logit on FEATS) and plain Elo
  research/data/upsets_dataset.csv  evaluation set (5,197 games 2022-23..2025-26) + de-vigged
                                     ESPN close / open: ONLY to score, never an input.
Every family prediction is already pre-game and walk-forward (audited: see README and
leak_test.py). The stacker only combines those predictions:
  avg3          mean of the three logits (nothing fitted)                     <- simplest
  stack3        logistic regression on the three logits, refitted at the start of every test
                season on the earlier TEST seasons' out-of-sample predictions only
                (2022-23 has no earlier out-of-sample season -> avg3 there)    <- PRIMARY (declared
                before looking at the test results of the ensemble)
  stack3_monthly  same, refitted on the 1st of every month on all earlier out-of-sample games
                (in-season updating; avg3 until 300 games are available)
  stack4        stack3 + the current fallback's logit as a 4th input
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
H7 = HERE.parent
ROOT = H7.parents[1]
DATA = ROOT / "research" / "data"
TEST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
FEATS = ["elo_diff", "net_diff", "form_diff", "rest_diff", "b2b_diff", "g5_diff"]   # backtest.py
FAMILY_FILES = {"player_impact": H7 / "player_impact" / "preds.csv",
                "margin_ratings": H7 / "margin_ratings" / "preds.csv",
                "margin_team_only": H7 / "margin_ratings" / "preds_team_only.csv",
                "elo_plus": H7 / "elo_plus" / "preds.csv"}
TRIO = ["player_impact", "margin_ratings", "elo_plus"]
PRIMARY = "stack3"
N_BOOT = 4000
SEED = 2026
EPS = 1e-6


# ------------------------------------------------------------------------------- helpers
def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigm(z):
    return 1.0 / (1.0 + np.exp(-z))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"n": int(len(y)), "logloss": round(float(ll_vec(y, p).mean()), 5),
            "brier": round(float(((p - y) ** 2).mean()), 5), "acc": round(float(((p > .5) == (y == 1)).mean()), 4)}


def boot(y, pa, pb, days, rng):
    """Paired bootstrap of mean log-loss(a) - log-loss(b), over games and over game-days."""
    d = ll_vec(y, pa) - ll_vec(y, pb)
    n = len(d)
    bs = np.concatenate([d[rng.integers(0, n, (500, n))].mean(1) for _ in range(N_BOOT // 500)])
    s = pd.Series(d).groupby(np.asarray(days)).agg(["sum", "size"])
    idx = rng.integers(0, len(s), (N_BOOT, len(s)))
    bd = s["sum"].to_numpy()[idx].sum(1) / s["size"].to_numpy()[idx].sum(1)
    q = lambda b: [round(float(np.quantile(b, .025)), 5), round(float(np.quantile(b, .975)), 5)]
    return {"n": int(n), "diff": round(float(d.mean()), 5), "ci95": q(bs), "ci95_dayblock": q(bd)}


# ------------------------------------------------------------------------------- data
def rerun_families():
    t0 = time.time()
    procs = {f: subprocess.Popen([sys.executable, str(H7 / f / "run.py")], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.STDOUT) for f in TRIO}
    for f, p in procs.items():
        p.wait()
        print(f"  re-ran {f}: exit {p.returncode} ({time.time() - t0:.0f}s)", flush=True)
        assert p.returncode == 0, f


def fallback_and_elo():
    """Current production fallback, exactly research/backtest.py: logit(C=1) on FEATS, trained on
    seasons 2019-20..S-1, predicting season S. Plain Elo = features.py elo_prob."""
    g = pd.read_csv(DATA / "games_features_odds.csv")
    out = pd.DataFrame({"GAME_ID": g.GAME_ID, "elo": g.elo_prob, "fallback": np.nan})
    for s in TEST_SEASONS:
        tr = g[(g.season < s) & (g.season >= "2019-20")]
        te = (g.season == s).to_numpy()
        m = LogisticRegression(C=1.0, max_iter=2000).fit(tr[FEATS], tr.home_win)
        out.loc[te, "fallback"] = m.predict_proba(g.loc[te, FEATS])[:, 1]
    return out


def load():
    u = pd.read_csv(DATA / "upsets_dataset.csv",
                    usecols=["game_id", "game_date", "season", "season_type", "home", "away", "home_win",
                             "mkt_close", "mkt_open"]).rename(columns={"game_id": "GAME_ID", "game_date": "date"})
    ev = u.merge(fallback_and_elo(), on="GAME_ID", how="left")
    for name, f in FAMILY_FILES.items():
        p = pd.read_csv(f)[["GAME_ID", "home_win", "p"]].rename(columns={"p": name, "home_win": f"y_{name}"})
        ev = ev.merge(p, on="GAME_ID", how="left")
        assert ev[name].notna().all(), f"{name}: missing predictions on the evaluation set"
        assert (ev[f"y_{name}"] == ev.home_win).all(), f"{name}: outcome mismatch"
        ev = ev.drop(columns=f"y_{name}")
    assert len(ev) == 5197 and ev.fallback.notna().all()
    ev["date"] = pd.to_datetime(ev.date)
    ev["market_close"], ev["market_open"] = ev.mkt_close, ev.mkt_open
    return ev.sort_values(["date", "GAME_ID"]).reset_index(drop=True)


# ------------------------------------------------------------------------------- stacking
def stack(ev, inputs, mode):
    """Walk-forward logistic stacker on the logits of `inputs`.
    mode 'season': refit at each season start on earlier test seasons; 'monthly': refit on the 1st
    of each month on every earlier game. Falls back to the plain logit average without data."""
    X = np.column_stack([logit(ev[c]) for c in inputs])
    y = ev.home_win.to_numpy()
    avg = sigm(X[:, :3].mean(1)) if len(inputs) >= 3 else sigm(X.mean(1))
    p = avg.copy()
    if mode == "season":
        blocks = [(ev.season == s).to_numpy() for s in TEST_SEASONS]
        train = [(ev.season < s).to_numpy() for s in TEST_SEASONS]
    else:
        per = ev.date.dt.to_period("M")
        blocks = [(per == m).to_numpy() for m in per.unique()]
        train = [(ev.date < m.to_timestamp()).to_numpy() for m in per.unique()]
    coefs = []
    for te, tr in zip(blocks, train):
        if tr.sum() < 300:
            continue
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        coefs.append({"first_date": str(ev.date[te].min().date()), "n_train": int(tr.sum()),
                      "intercept": round(float(m.intercept_[0]), 4),
                      **{c: round(float(w), 4) for c, w in zip(inputs, m.coef_[0])}})
    return p, coefs


# ------------------------------------------------------------------------------- content view
def content_view(ev, model, thr=0.05):
    """When our model and the closing line disagree by more than `thr`, who is closer to the truth?"""
    p, c, y = ev[model].to_numpy(), ev.mkt_close.to_numpy(), ev.home_win.to_numpy()
    dis = np.abs(p - c) > thr
    week = ev.date.dt.to_period("W")
    out = {"threshold": thr, "n_games": int(len(ev)), "n_disagree": int(dis.sum()),
           "share": round(float(dis.mean()), 4)}
    # frequency: per week of the regular season (weeks with at least 20 games)
    rs = (ev.season_type == "Regular Season").to_numpy()
    wk = pd.DataFrame({"w": week[rs], "d": dis[rs]}).groupby("w").d.agg(["sum", "size"])
    wk = wk[wk["size"] >= 20]
    out["per_regular_season_week"] = {"mean": round(float(wk["sum"].mean()), 1),
                                      "median": float(wk["sum"].median()),
                                      "games_per_week": round(float(wk["size"].mean()), 1)}
    out["per_season"] = {s: int(dis[(ev.season == s).to_numpy()].sum()) for s in TEST_SEASONS}
    # who is right
    d = dis
    up = d & (p > c)            # we like the home team more than the market
    dn = d & (p < c)
    out["model_higher_on_home"] = {"n": int(up.sum()), "mean_model": round(float(p[up].mean()), 4),
                                   "mean_close": round(float(c[up].mean()), 4),
                                   "home_win_rate": round(float(y[up].mean()), 4)}
    out["model_lower_on_home"] = {"n": int(dn.sum()), "mean_model": round(float(p[dn].mean()), 4),
                                  "mean_close": round(float(c[dn].mean()), 4),
                                  "home_win_rate": round(float(y[dn].mean()), 4)}
    # direction of reality vs the two: share of games where the outcome went the model's way
    # relative to the market = P(outcome on the side the model moved toward)
    moved_right = np.where(p > c, y == 1, y == 0)[d]
    out["outcome_on_model_side_share"] = round(float(moved_right.mean()), 4)
    out["logloss_on_disagreements"] = {"model": round(float(ll_vec(y[d], p[d]).mean()), 4),
                                       "close": round(float(ll_vec(y[d], c[d]).mean()), 4)}
    out["brier_on_disagreements"] = {"model": round(float(((p[d] - y[d]) ** 2).mean()), 4),
                                     "close": round(float(((c[d] - y[d]) ** 2).mean()), 4)}
    # picks that differ (one says home, the other away)
    flip = (p > .5) != (c > .5)
    out["different_pick"] = {"n": int(flip.sum()),
                             "per_regular_season_week": round(float(
                                 pd.DataFrame({"w": week[rs], "f": flip[rs]}).groupby("w").f.sum()
                                 .reindex(wk.index).mean()), 2),
                             "model_pick_won": int(((p > .5) == (y == 1))[flip].sum()),
                             "market_pick_won": int(((c > .5) == (y == 1))[flip].sum())}
    return out


# ------------------------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    if "--rerun" in sys.argv:
        rerun_families()
    ev = load()
    ev["avg3"] = sigm(np.column_stack([logit(ev[c]) for c in TRIO]).mean(1))
    coefs = {}
    ev["stack3"], coefs["stack3"] = stack(ev, TRIO, "season")
    ev["stack3_monthly"], coefs["stack3_monthly"] = stack(ev, TRIO, "monthly")
    ev["stack4"], coefs["stack4"] = stack(ev, TRIO + ["fallback"], "season")
    ev["avg2_team"] = sigm((logit(ev.margin_team_only) + logit(ev.elo_plus)) / 2)   # informational

    own = ["stack3", "stack3_monthly", "stack4", "avg3", "player_impact", "margin_ratings", "elo_plus",
           "margin_team_only"]
    base = ["fallback", "elo", "market_close", "market_open"]
    y = ev.home_win.to_numpy().astype(float)
    op = ev.mkt_open.notna().to_numpy()
    days = ev.date.dt.strftime("%Y-%m-%d").to_numpy()
    res = {"n_games": int(len(ev)), "n_open_subset": int(op.sum()), "primary": PRIMARY,
           "pooled": {}, "pooled_open_subset": {}, "seasons": {}, "bootstrap": {}, "stack_coefs": coefs}
    for n in own + base:
        m = op if n == "market_open" else np.ones(len(ev), bool)
        res["pooled"][n] = metrics(y[m], ev[n].to_numpy()[m])
        res["pooled_open_subset"][n] = metrics(y[op], ev[n].to_numpy()[op])
        res["seasons"][n] = {s: metrics(y[m & (ev.season == s).to_numpy()], ev[n].to_numpy()[m & (ev.season == s).to_numpy()])
                             for s in TEST_SEASONS if (m & (ev.season == s).to_numpy()).any()}
    rng = np.random.default_rng(SEED)
    for n in own + ["elo"]:
        b = {"vs_fallback": boot(y, ev[n], ev.fallback, days, rng),
             "vs_market_close": boot(y, ev[n], ev.mkt_close, days, rng),
             "vs_market_open_open_subset": boot(y[op], ev[n][op], ev.mkt_open[op], days[op], rng)}
        b["seasons_better_than_fallback"] = int(sum(res["seasons"][n][s]["logloss"] < res["seasons"]["fallback"][s]["logloss"]
                                                    for s in TEST_SEASONS))
        b["seasons_better_than_close"] = int(sum(res["seasons"][n][s]["logloss"] < res["seasons"]["market_close"][s]["logloss"]
                                                 for s in TEST_SEASONS))
        res["bootstrap"][n] = b
    for n in TRIO:
        res["bootstrap"][f"{PRIMARY}_vs_{n}"] = boot(y, ev[PRIMARY], ev[n], days, rng)
    res["bootstrap"]["avg3_vs_stack3"] = boot(y, ev.avg3, ev.stack3, days, rng)
    res["bootstrap"]["market_close_vs_fallback"] = boot(y, ev.mkt_close, ev.fallback, days, rng)

    # correlation between families (why stacking helps little or a lot)
    res["logit_corr"] = pd.DataFrame({c: logit(ev[c]) for c in TRIO + ["fallback", "mkt_close"]}).corr().round(3).to_dict()
    # season phase
    mo = ev.date.dt.month
    ph = np.select([ev.season_type != "Regular Season", mo.isin([10, 11, 12]), mo.isin([1, 2])],
                   ["4_playin_playoffs", "1_oct_dec", "2_jan_feb"], "3_mar_apr")
    res["by_phase_logloss"] = {k: {"n": int((ph == k).sum()),
                                   **{n: round(float(ll_vec(y[ph == k], ev[n].to_numpy()[ph == k]).mean()), 5)
                                      for n in [PRIMARY, "fallback", "market_close"]}}
                               for k in sorted(set(ph))}
    # calibration of the primary (diagnostic only)
    bins = np.clip((ev[PRIMARY] * 10).astype(int), 0, 9)
    res["calibration_primary"] = [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "n": int((bins == b).sum()),
                                   "mean_p": round(float(ev[PRIMARY][bins == b].mean()), 4),
                                   "home_win_rate": round(float(y[bins == b].mean()), 4)}
                                  for b in range(10) if (bins == b).any()]
    res["content_view"] = {PRIMARY: content_view(ev, PRIMARY, 0.05),
                           f"{PRIMARY}_10pts": content_view(ev, PRIMARY, 0.10)}
    res["runtime_s"] = round(time.time() - t0, 1)

    out = ev[["GAME_ID", "season", "date", "home", "away", "home_win"]].copy()
    out["date"] = out.date.dt.strftime("%Y-%m-%d")
    out["home_win"] = out.home_win.astype(int)
    out["p"] = ev[PRIMARY].round(5)
    out.to_csv(HERE / "preds.csv", index=False)
    (HERE / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # console
    rows = []
    for n in own + base:
        r = res["pooled"][n]
        rows.append({"model": n, "n": r["n"], "logloss": r["logloss"], "brier": r["brier"], "acc": r["acc"],
                     "ll_open_subset": res["pooled_open_subset"][n]["logloss"],
                     **{f"ll_{s}": res["seasons"][n].get(s, {}).get("logloss") for s in TEST_SEASONS}})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nbootstrap (diff, CI games, CI game-days):")
    for n, b in res["bootstrap"].items():
        if "vs_fallback" in b:
            f = lambda x: f"{x['diff']:+.4f} [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] [{x['ci95_dayblock'][0]:+.4f},{x['ci95_dayblock'][1]:+.4f}]"
            print(f"  {n:16s} fb {f(b['vs_fallback'])} | close {f(b['vs_market_close'])} | open {f(b['vs_market_open_open_subset'])}"
                  f" | seasons<fb {b['seasons_better_than_fallback']}/4")
        else:
            print(f"  {n:28s} {b['diff']:+.4f} {b['ci95']} {b['ci95_dayblock']}")
    print("\nstack coefs:", json.dumps(coefs["stack3"]))
    print("logit corr:", json.dumps(res["logit_corr"]))
    print("phase:", json.dumps(res["by_phase_logloss"]))
    print("content:", json.dumps(res["content_view"], indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
