#!/usr/bin/env python3
"""Export the own model's frozen parameters (data/own_model_params.json) from the research fits.

    python scripts/export_own_model_params.py [--research-dir research] [--parity]

Runs the research code of research/h7_own_model (elo_plus, margin_ratings, player_impact) on the
research data (research/data/, not in git: re-create it with research/fetch_*.py and
research/h1_player_availability/fetch_players.py). About 10 minutes.

Production parameters = what the research walk-forward would use for the next season: every
family re-fitted on all seasons up to and including FIT_THROUGH.
  elo_plus, mov_rest  L-BFGS on seasons 2019-20..FIT_THROUGH (warm start: the research's last fit),
                      P(absent | games missed) from every player season, replacement level 2020-21
  margin_ratings      hyper-parameter grid, sigma, schedule / late-season / absence coefficients on
                      seasons 2019-20..FIT_THROUGH
  player_impact       ridge gap + logit on every game since 2021-22; P(plays) table, replacement
                      level and Kalman settings exactly as in the research run
--parity also writes tests/fixtures/own_model_params_2024_25.json: the parameters the research used
for its 2024-25 test season (with the monthly player_impact refits), for the parity test.
Re-run once before each season (then commit the json).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # tiny 31x31 solves: BLAS threading makes them much slower
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIT_THROUGH = "2025-26"
NEXT = "2026-27"
PARITY = "2024-25"


def load_module(path: Path, name: str):
    sys.dont_write_bytecode = True       # read-only use of the research folder
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def f(x):
    return float(x)


def elo_params(rd: Path, parity: bool):
    ep = load_module(rd / "h7_own_model" / "elo_plus" / "run.py", "h7_elo_plus")
    res = json.loads((rd / "h7_own_model" / "elo_plus" / "results.json").read_text())
    g = ep.load_games()
    a = ep.load_availability(False)
    repl = ep._replacement_rate(ep._player_logs())

    def pout(season):
        return [f(x) for x in ep.p_out_table(a, sorted(s for s in a.season.unique() if s < season))]

    pn = pout(NEXT)
    A = ep.model_arrays(g, ep.team_game_availability(a, np.array(pn)))
    mask = ((g.season >= ep.FIRST_TUNE_SEASON) & (g.season < NEXT)).to_numpy().astype(float)
    out = {}
    for v, key in (("elo_plus", "elo_plus"), ("mov_rest", "elo_team")):
        t0 = time.time()
        par, ll, calls = ep.tune(v, A, mask, x0=res["params"][v][FIT_THROUGH])
        log(f"elo {v}: tune log-loss {ll:.5f} ({calls} passes, {time.time() - t0:.0f}s)")
        out[key] = {"variant": v, "par": {k: f(x) for k, x in par.items()}}
    out["elo_plus"].update({"p_out": pn, "repl": f(repl), "gap_buckets": ep.GAP_BUCKETS[:-1]})
    par_out = None
    if parity:
        par_out = {"elo_plus": {"variant": "elo_plus", "par": res["params"]["elo_plus"][PARITY],
                                "p_out": pout(PARITY), "repl": f(repl)},
                   "elo_team": {"variant": "mov_rest", "par": res["params"]["mov_rest"][PARITY]}}
    return out, par_out


def margin_params(rd: Path, parity: bool):
    mr = load_module(rd / "h7_own_model" / "margin_ratings" / "run.py", "h7_margin")
    h1 = load_module(rd / "h1_player_availability" / "build.py", "h1_build")
    repl = f(h1.replacement_rate(h1.player_logs()))
    g = mr.load_games()
    blocks = mr.date_blocks(g)
    y = g.home_win.to_numpy(float)
    margin = g.margin.to_numpy(float)
    luck = mr.shooting_luck(g)
    S = mr.schedule_terms(g)
    t0 = time.time()
    preds = mr.run_grid(g, blocks, luck)
    log(f"margin grid: {len(preds)} configs ({time.time() - t0:.0f}s)")

    def fit(season):
        tr = mr.season_mask(g, mr.TUNE_FROM, season)
        ll, cfg, _ = mr.select(preds, y, tr)
        hl, carry, k, cap, alpha, per100 = cfg
        ladj = alpha * (luck[0] - luck[1])
        resid = margin - preds[cfg]
        cut = mr.choose_late_cut(g, S, resid, tr)
        SL = pd.concat([S, mr.late_terms(g, cut)], axis=1)
        b_f = mr.ols(SL.to_numpy(float)[tr], resid[tr])
        adj_f = SL.to_numpy(float) @ b_f
        mu_f, _ = mr.rate(g, blocks, hl=hl, carry=carry, k=k, cap=cap, per100=per100, adj=ladj + adj_f)
        mu_f = mu_f + adj_f
        sig_f = mr.fit_sigma(mu_f[tr], y[tr])
        trp = tr & g.has_players.to_numpy()
        xa = g.avail_prev_diff.to_numpy()
        r2 = margin - mu_f
        b_a = float(xa[trp] @ r2[trp] / (xa[trp] @ xa[trp]))
        sig_a = mr.fit_sigma((mu_f + b_a * xa)[tr], y[tr])
        log(f"margin fit < {season}: {mr.cfg_dict(cfg)} late_cut={cut} sigma={sig_f:.3f}/{sig_a:.3f} "
            f"b_avail={b_a:.4f} train-ll={ll:.5f}")
        return {"hl": f(hl), "carry": f(carry), "k": f(k), "cap": None if not np.isfinite(cap) else f(cap),
                "alpha": f(alpha), "per100": bool(per100), "late_cut": f(cut),
                "beta": {c: f(b) for c, b in zip(SL.columns, b_f)}, "sigma_team": f(sig_f),
                "beta_avail": b_a, "sigma_avail": f(sig_a), "repl": repl}

    return fit(NEXT), (fit(PARITY) if parity else None)


def player_impact_params(rd: Path, parity: bool):
    from sklearn.linear_model import LogisticRegression
    pi = load_module(rd / "h7_own_model" / "player_impact" / "run.py", "h7_player_impact")
    t0 = time.time()
    p, t, g = pi.load_players(), pi.load_team_games(), pi.load_games()
    st, prior = pi.player_states(p, t)
    kal = pi.KalmanAPM(p, t)
    cand = pi.attach_state(pi.candidate_rows(p, t), st, prior, kal)
    cand = cand[cand.last_team == cand.TEAM_ID].copy()
    tab = pi.fit_p_play(cand)
    cand["e_min"] = pd.Series(pi.play_key(cand)).map(tab).fillna(0.5).to_numpy() * cand.m_hat.to_numpy()
    gg = g[g.season >= "2020-21"]
    D = pi.game_diffs(pi.composites(cand, "e_min"), gg)
    d = gg.merge(D, on="GAME_ID", how="inner")
    log(f"player_impact states + lineups ({time.time() - t0:.0f}s)")
    dcols = [f"d_{c}" for c in pi.XCOLS]
    cols = ["gap"] + pi.SCHED + pi.TEAM_EXTRA

    def fit(tr):
        gm = pi.GapModel().fit(tr[dcols].to_numpy(), tr.y100.to_numpy())
        X = tr[pi.SCHED + pi.TEAM_EXTRA].assign(gap=gm.gap(tr[dcols].to_numpy()))[cols]
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, tr.home_win)
        return {"n_train": int(len(tr)), "gap_beta": {c: f(b) for c, b in zip(pi.XCOLS, gm.beta)},
                "logit": {**{c: f(w) for c, w in zip(cols, lr.coef_[0])}, "intercept": f(lr.intercept_[0])}}

    kal_cfg = {k: v for k, v in pi.KAL.items() if k != "burnin_end"}
    kal_cfg["burnin_days"] = int((pd.Timestamp(pi.KAL["burnin_end"]) - p.date.min()).days)
    base = {"prior": {c: f(prior[c]) for c in pi.RATE_COLS}, "kalman": kal_cfg,
            "p_play": {str(int(k)): f(v) for k, v in tab.items()}}
    prod = {**base, **fit(d[d.season >= pi.FIRST_TRAIN])}
    par_out = None
    if parity:
        ds = pi.refit_dates(d, PARITY)
        refits = []
        for lo in ds[:-1]:
            refits.append({"from": lo.strftime("%Y-%m-%d"), **fit(d[(d.season >= pi.FIRST_TRAIN) & (d.date < lo)])})
        par_out = {**base, "refits": refits}
    return prod, par_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-dir", default=str(ROOT / "research"))
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "own_model_params.json"))
    args = ap.parse_args()
    rd = Path(args.research_dir).resolve()
    t0 = time.time()
    pi_prod, pi_par = player_impact_params(rd, args.parity)
    m_prod, m_par = margin_params(rd, args.parity)
    e_prod, e_par = elo_params(rd, args.parity)
    meta = {"version": 1, "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "research/h7_own_model via scripts/export_own_model_params.py",
            "mode": {"stale_games": 3}}
    prod = {**meta, "fitted_through": FIT_THROUGH, "for_season": NEXT, **e_prod, "margin": m_prod,
            "player_impact": pi_prod}
    Path(args.out).write_text(json.dumps(prod, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {args.out}")
    if args.parity:
        par = {**meta, "for_season": PARITY, "note": "research walk-forward parameters of the 2024-25 test season",
               **e_par, "margin": m_par, "player_impact": pi_par}
        fx = ROOT / "tests" / "fixtures" / "own_model_params_2024_25.json"
        fx.write_text(json.dumps(par, indent=1) + "\n", encoding="utf-8")
        log(f"wrote {fx}")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
