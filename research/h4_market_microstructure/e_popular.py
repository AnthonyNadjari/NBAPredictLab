"""(e) Behavioural biases: popular-team pricing, overreaction to the previous result
(20+ point wins/losses) and to streaks. All inputs are known before tip-off (previous game
margin and streak entering the game come from strictly earlier team logs)."""
import numpy as np
import pandas as pd

from common import TEST_SEASONS, am_to_dec, bet_roi, compare, logit, pnl_roi, walk_forward

POP4 = {"LAL", "GSW", "BOS", "NYK"}
# approx. top-10 NBA franchises by combined social-media following (FB+X+IG, ~2022-24)
POP10 = {"GSW", "LAL", "CHI", "MIA", "BOS", "CLE", "HOU", "SAS", "OKC", "NYK"}


def add_cols(g):
    g = g.copy()
    for k, s in (("pop4", POP4), ("pop10", POP10)):
        g[f"h_{k}"], g[f"a_{k}"] = g.home.isin(s).astype(int), g.away.isin(s).astype(int)
        g[f"{k}_diff"] = g[f"h_{k}"] - g[f"a_{k}"]
    g["fav_sign"] = np.where(g.p_close >= .5, 1, -1)
    for side in ("home", "away"):
        pm = g[f"{side}_prev_margin"]
        g[f"{side}_bw"] = (pm >= 20).astype(int)
        g[f"{side}_bl"] = (pm <= -20).astype(int)
        st = g[f"{side}_streak"].fillna(0)
        g[f"{side}_hot"] = (st >= 5).astype(int)
        g[f"{side}_cold"] = (st <= -5).astype(int)
    g["bw_diff"] = g.home_bw - g.away_bw
    g["bl_diff"] = g.home_bl - g.away_bl
    g["prev_margin_diff"] = (g.home_prev_margin.fillna(0) - g.away_prev_margin.fillna(0)) / 10
    g["hot_diff"] = g.home_hot - g.away_hot
    g["cold_diff"] = g.home_cold - g.away_cold
    g["streak_diff"] = (g.home_streak.fillna(0) - g.away_streak.fillna(0)) / 5
    return g


def _side_rule(t, home_sel, away_sel, label):
    """Bet the home team where home_sel, the away team where away_sel, at the closing ML."""
    pnl = np.r_[np.where(t.home_win[home_sel] == 1, am_to_dec(t.h_close[home_sel]) - 1, -1.0),
                np.where(t.home_win[away_sel] == 0, am_to_dec(t.a_close[away_sel]) - 1, -1.0)]
    r = pnl_roi(pnl, np.r_[t.season[home_sel], t.season[away_sel]], label=label)
    # calibration of the bet side: won vs de-vigged price
    p = np.r_[t.p_close[home_sel], 1 - t.p_close[away_sel]]
    w = np.r_[t.home_win[home_sel], 1 - t.home_win[away_sel]]
    if len(p):
        r["priced"], r["won"] = float(p.mean()), float(w.mean())
        r["z"] = float((w.mean() - p.mean()) / np.sqrt((p * (1 - p)).sum()) * len(p))
    return r


def run(g):
    out = {"logloss": [], "bets": [], "tables": {}}
    g = add_cols(g)
    t = g[g.season.isin(TEST_SEASONS)]
    variants = {
        "pop4_diff": lambda d: d.pop4_diff,
        "pop10_diff": lambda d: d.pop10_diff,
        "pop10_diff+pop10_x_fav": lambda d: np.c_[d.pop10_diff, d.pop10_diff * d.fav_sign],
        "prev_game_blowout(20+) win/loss": lambda d: np.c_[d.bw_diff, d.bl_diff],
        "prev_game_margin": lambda d: d.prev_margin_diff,
        "streak_5+_hot/cold": lambda d: np.c_[d.hot_diff, d.cold_diff],
        "streak_continuous": lambda d: d.streak_diff,
    }
    platt = walk_forward(g, None)
    preds = {}
    for name, fx in variants.items():
        p = walk_forward(g, fx)
        preds[name] = p
        r = compare(t, p[t.index].to_numpy(), t.p_close.to_numpy(), name)
        r["delta_vs_platt"] = compare(t, p[t.index].to_numpy(), platt[t.index].to_numpy(), name)["delta"]
        out["logloss"].append({"part": "e", **r})

    for name in ("pop10_diff+pop10_x_fav", "prev_game_blowout(20+) win/loss"):
        for th in (0.0, 0.02):
            out["bets"].append({"part": "e", **bet_roi(t, preds[name][t.index].to_numpy(), th,
                                                         label=f"{name} model edge>{th}")})
    # direct contrarian rules at the closing ML (test seasons; no fitting)
    fh = t.p_close >= .5
    rules = [
        ("fade popular (top10) favourite vs non-popular dog",
         ~fh & (t.a_pop10 == 1) & (t.h_pop10 == 0), fh & (t.h_pop10 == 1) & (t.a_pop10 == 0)),
        ("back popular (top10) underdog vs non-popular fav",
         ~fh & (t.h_pop10 == 1) & (t.a_pop10 == 0), fh & (t.a_pop10 == 1) & (t.h_pop10 == 0)),
        ("fade LAL/GSW/BOS/NYK favourite",
         ~fh & (t.a_pop4 == 1) & (t.h_pop4 == 0), fh & (t.h_pop4 == 1) & (t.a_pop4 == 0)),
        ("back LAL/GSW/BOS/NYK underdog",
         ~fh & (t.h_pop4 == 1) & (t.a_pop4 == 0), fh & (t.a_pop4 == 1) & (t.h_pop4 == 0)),
        ("back team coming off a 20+ pt loss", (t.home_bl == 1) & (t.away_bl == 0), (t.away_bl == 1) & (t.home_bl == 0)),
        ("fade team coming off a 20+ pt win", (t.away_bw == 1) & (t.home_bw == 0), (t.home_bw == 1) & (t.away_bw == 0)),
        ("fade team on 5+ win streak", (t.away_hot == 1) & (t.home_hot == 0), (t.home_hot == 1) & (t.away_hot == 0)),
        ("back team on 5+ losing streak", (t.home_cold == 1) & (t.away_cold == 0), (t.away_cold == 1) & (t.home_cold == 0)),
        # added after seeing that popular favourites beat their price (opposite of the
        # hypothesis): reported for transparency, NOT a pre-specified test
        ("POST-HOC mirror: back popular (top10) favourite vs non-popular dog",
         fh & (t.h_pop10 == 1) & (t.a_pop10 == 0), ~fh & (t.a_pop10 == 1) & (t.h_pop10 == 0)),
    ]
    for label, hs, as_ in rules:
        out["bets"].append({"part": "e", **_side_rule(t, hs.to_numpy(), as_.to_numpy(), label)})
    return out
