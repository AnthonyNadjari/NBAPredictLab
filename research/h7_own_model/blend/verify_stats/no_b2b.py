"""Bets where neither team played the day before (model inputs knowable when the open was posted, ~T-24h)."""
import numpy as np

from verify import boot_ratio, load, stakes, wf_blend


def main():
    ev = load()
    p, _ = wf_blend(ev)
    m = ~np.isnan(p)
    e, p = ev[m].reset_index(drop=True), p[m]
    y, days = e.home_win.to_numpy(), e.game_date.to_numpy()
    sh, sa = stakes(p, e.dh, e.da)
    bet, home = (sh + sa) > 0, sh > 0
    d_open = np.where(home, e.dh, e.da)
    won = np.where(home, y == 1, y == 0)
    pnl = np.where(won, d_open - 1, -1.0)
    pc = np.where(home, e.pc_mult, 1 - e.pc_mult)
    clv = pc * d_open - 1
    b2b = ((e.home_back_to_back == 1) | (e.away_back_to_back == 1)).to_numpy()
    for lab, mm in (("neither on B2B", bet & ~b2b), ("some team on B2B", bet & b2b)):
        for s in ("all", "2024-25", "2025-26"):
            k = mm & ((e.season == s).to_numpy() if s != "all" else True) & ~np.isnan(clv)
            (lo, hi), _ = boot_ratio(pnl[k], np.ones(k.sum()), days[k])
            (clo, chi), _ = boot_ratio(clv[k], np.ones(k.sum()), days[k])
            print(f"{lab:17s} {s:8s} n={k.sum():4d} ROI {100*pnl[k].mean():+6.1f}% [{100*lo:+.1f}, {100*hi:+.1f}]"
                  f"  CLV {100*clv[k].mean():+.2f}% [{100*clo:+.2f}, {100*chi:+.2f}]")


if __name__ == "__main__":
    main()
