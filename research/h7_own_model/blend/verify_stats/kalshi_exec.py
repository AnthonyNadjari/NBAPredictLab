"""Executable-price sensitivity: blend own model with the Kalshi price at time t and bet at that price.

    python research/h7_own_model/blend/verify_stats/kalshi_exec.py
"""
import numpy as np
import pandas as pd

from verify import DATA, blend, boot_ratio, load, ll, wf_blend


def main():
    ev = load()
    p_open, _ = wf_blend(ev)
    k = pd.read_csv(DATA / "h6_timing" / "kalshi_snapshots.csv")
    e = ev.assign(p_open=p_open).merge(k, on="event_id", how="inner")
    e = e[e.season == "2025-26"].reset_index(drop=True)
    y = e.home_win.to_numpy()
    days = e.game_date.to_numpy()
    print(f"2025-26 games with Kalshi: {len(e)}")
    for col in ("k_prev_utc09", "k_h24", "k_utc09", "k_utc17", "k_utc21"):
        kk = e[col].to_numpy()
        ok = ~np.isnan(kk)
        print(f"\n{col}: {ok.sum()} games; log-loss kalshi {ll(y[ok], kk[ok]).mean():.4f} | ESPN open {ll(y[ok], e.mkt_open.to_numpy()[ok]).mean():.4f}"
              f" | own {ll(y[ok], e.own.to_numpy()[ok]).mean():.4f}")
        for w in (0.25, 0.45):
            pb = blend(e.own.to_numpy(), np.clip(kk, .01, .99), w)
            print(f"   blend(own, kalshi, w={w}) log-loss {ll(y[ok], pb[ok]).mean():.4f} (vs kalshi {ll(y[ok], pb[ok]).mean() - ll(y[ok], kk[ok]).mean():+.4f})")
            for spread, fee in ((0.0, 0.0), (0.01, 0.0), (0.01, 0.07)):
                ch = np.clip(kk + spread, .01, .99); ca = np.clip(1 - kk + spread, .01, .99)
                ch, ca = ch + fee * ch * (1 - ch), ca + fee * ca * (1 - ca)
                dh, da = 1 / ch, 1 / ca
                eh, ea = pb * dh - 1, (1 - pb) * da - 1
                hb, ab = ok & (eh > 0) & (eh >= ea), ok & (ea > 0) & (ea > eh)
                pn = np.where(hb, np.where(y == 1, dh - 1, -1), 0) + np.where(ab, np.where(y == 0, da - 1, -1), 0)
                nb = hb | ab
                (lo, hi), _ = boot_ratio(pn[nb], np.ones(nb.sum()), days[nb])
                print(f"     half-spread {spread:.2f} fee {fee:.2f}: bets {nb.sum():4d} ROI {100*pn[nb].mean():+6.1f}% [{100*lo:+.1f}, {100*hi:+.1f}]")


if __name__ == "__main__":
    main()
