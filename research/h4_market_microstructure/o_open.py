"""Protocol baseline (b): the OPENING line. Opening moneylines exist from 2023-24 only, so
the walk-forward test seasons are 2024-25 (train 2023-24) and 2025-26 (train 2023-25).
Do the same price-structure / behavioural features add information to the OPENING price,
and can one profit by betting at the opening price (incl. vig)?"""
import numpy as np
import pandas as pd

from common import am_num, am_to_p, bet_roi, compare, logit, walk_forward
from e_popular import add_cols

TEST = ["2024-25", "2025-26"]
K_PTS = 12.5 * np.sqrt(2 * np.pi)


def run(g, books):
    out = {"logloss": [], "bets": [], "tables": {}}
    g = add_cols(g)
    g = g[g.p_open.notna() & g.season.isin(["2023-24"] + TEST)].reset_index(drop=True)
    g["sp_open_eff"] = np.nan
    if books is not None and "h_sp_open" in books.columns:
        j = books[["event_id", "book", "h_sp_open", "h_so_open", "a_so_open"]].copy()
        ph, pa = am_to_p(j.h_so_open), am_to_p(j.a_so_open)
        pc = np.where(np.isfinite(ph / (ph + pa)) & (ph + pa > 1) & (ph + pa < 1.12), ph / (ph + pa), 0.5)
        j["sp_open_eff"] = am_num(j.h_sp_open) - (pc - 0.5) * K_PTS
        g = g.drop(columns="sp_open_eff").merge(j[["event_id", "book", "sp_open_eff"]],
                                               on=["event_id", "book"], how="left")
    g["fav_sign_open"] = np.where(g.p_open >= .5, 1, -1)
    t = g[g.season.isin(TEST)]
    variants = {
        "open: platt": None,
        "open: +|logit| (FLB)": lambda d: np.abs(logit(d.p_open)),
        "open: +pop10_diff+pop10_x_fav": lambda d: np.c_[d.pop10_diff, d.pop10_diff * d.fav_sign_open],
        "open: +prev_game_blowout": lambda d: np.c_[d.bw_diff, d.bl_diff],
        "open: +streak_5+": lambda d: np.c_[d.hot_diff, d.cold_diff],
    }
    if g.sp_open_eff.notna().mean() > .9:
        # games without an opening spread get NaN -> excluded from this variant's comparison
        variants["open: +open_spread_eff"] = lambda d: -d.sp_open_eff
    for name, fx in variants.items():
        p = walk_forward(g, fx, base="p_open", test_seasons=TEST)
        out["logloss"].append({"part": "open", "baseline": "open",
                               **compare(t, p[t.index].to_numpy(), t.p_open.to_numpy(), name)})
        for th in (0.0, 0.02):
            out["bets"].append({"part": "open", **bet_roi(t, p[t.index].to_numpy(), th, h_odds="h_open",
                                                            a_odds="a_open", label=f"{name} edge>{th} @open")})
    return out
