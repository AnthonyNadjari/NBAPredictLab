"""(d) Line movement open -> close (main book). Opening lines exist from 2023-24 only, so
the walk-forward test seasons are 2024-25 and 2025-26 (the 3-of-4 rule cannot be met)."""
import numpy as np
import pandas as pd

from common import am_num, bet_roi, compare, ll_vec, logit, pnl_roi, walk_forward, am_to_dec

TEST = ["2024-25", "2025-26"]
OPEN_SEASONS = ["2023-24", "2024-25", "2025-26"]


def add_move_cols(g, books):
    g = g.copy()
    g["move"] = logit(g.p_close) - logit(g.p_open)  # + = moved toward home
    g["sp_move"] = np.nan
    if books is not None and "h_sp_open" in books.columns:
        # ESPN's 'close.pointSpread' field is unreliable (holds odds in 2022-23), so the move
        # is measured from the open spread to the main-book closing spread already in g
        j = books[["event_id", "book", "h_sp_open"]].copy()
        j["sp_open"] = am_num(j.h_sp_open)
        g = g.drop(columns="sp_move").merge(j[["event_id", "book", "sp_open"]], on=["event_id", "book"], how="left")
        g["sp_move"] = g.spread - g.sp_open  # negative = moved toward home
        g.loc[g.sp_move.abs() > 12, "sp_move"] = np.nan
    return g


def run(g, books):
    out = {"logloss": [], "bets": [], "tables": {}}
    g = add_move_cols(g, books)
    g = g[g.season.isin(OPEN_SEASONS) & g.p_open.notna()].reset_index(drop=True)
    t = g[g.season.isin(TEST)]
    # descriptive: open vs close accuracy / log-loss; size of moves
    rows = []
    for s, x in g.groupby("season"):
        rows.append({"season": s, "n": len(x), "ll_open": ll_vec(x.home_win, x.p_open).mean(),
                     "ll_close": ll_vec(x.home_win, x.p_close).mean(),
                     "acc_open": ((x.p_open > .5) == (x.home_win == 1)).mean(),
                     "acc_close": ((x.p_close > .5) == (x.home_win == 1)).mean(),
                     "mean_abs_move_prob": (x.p_close - x.p_open).abs().mean(),
                     "share_moved": (x.move.abs() > 1e-6).mean(),
                     "sp_move_known": x.sp_move.notna().mean()})
    out["tables"]["d_open_vs_close"] = pd.DataFrame(rows)
    # protocol baseline (b): the opening line. Close vs open (no fitting).
    out["logloss"].append({"part": "d", **compare(t, t.p_close.to_numpy(), t.p_open.to_numpy(),
                                                   "close_vs_open (reference)")})
    g["fav_sign"] = np.sign(logit(g.p_close))
    g["sp_move_f"] = g.sp_move.fillna(0)
    variants = {
        "close+move": lambda d: d.move,
        "close+move+steam(|move|>0.2)": lambda d: np.c_[d.move, d.move * (d.move.abs() > 0.2)],
        "close+move_toward_fav": lambda d: d.move * d.fav_sign,
        "close+spread_move": lambda d: d.sp_move_f,
        "close+move+spread_move": lambda d: np.c_[d.move, d.sp_move_f],
    }
    platt = walk_forward(g, None, test_seasons=TEST)
    for name, fx in variants.items():
        p = walk_forward(g, fx, test_seasons=TEST)
        r = compare(t, p[t.index].to_numpy(), t.p_close.to_numpy(), name)
        r["delta_vs_platt"] = compare(t, p[t.index].to_numpy(), platt[t.index].to_numpy(), name)["delta"]
        out["logloss"].append({"part": "d", **r})
        if name == "close+move":
            for th in (0.0, 0.02):
                out["bets"].append({"part": "d", **bet_roi(t, p[t.index].to_numpy(), th,
                                                             label=f"close+move model edge>{th}")})
    # simple rules at the CLOSING price, all seasons with opens (no fitting involved)
    for th in (0.1, 0.2, 0.3):
        for mode in ("follow", "fade"):
            home = g.move > th if mode == "follow" else g.move < -th
            away = g.move < -th if mode == "follow" else g.move > th
            pnl = np.r_[np.where(g.home_win[home] == 1, am_to_dec(g.h_close[home]) - 1, -1.0),
                        np.where(g.home_win[away] == 0, am_to_dec(g.a_close[away]) - 1, -1.0)]
            ss = np.r_[g.season[home], g.season[away]]
            out["bets"].append({"part": "d", **pnl_roi(pnl, ss, label=f"{mode} move |logit|>{th} at close [23-26]")})
    # oracle (NOT a strategy): bet at the OPEN on the side the line later moves toward
    home, away = g.move > 0.1, g.move < -0.1
    pnl = np.r_[np.where(g.home_win[home] == 1, am_to_dec(g.h_open[home]) - 1, -1.0),
                np.where(g.home_win[away] == 0, am_to_dec(g.a_open[away]) - 1, -1.0)]
    out["bets"].append({"part": "d", "oracle": True,
                        **pnl_roi(pnl, np.r_[g.season[home], g.season[away]],
                                  label="ORACLE: bet at open the side the line moves to (|move|>0.1)")})
    b = g.assign(move_bin=pd.cut(g.move, [-9, -.3, -.1, -.02, .02, .1, .3, 9]))
    out["tables"]["d_move_bins"] = b.groupby("move_bin", observed=True).agg(
        n=("home_win", "size"), p_open=("p_open", "mean"), p_close=("p_close", "mean"),
        home_won=("home_win", "mean")).reset_index().assign(move_bin=lambda x: x.move_bin.astype(str))
    return out
