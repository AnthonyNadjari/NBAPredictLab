"""Follow-ups to verify.py: outlier vig row, stale open==close rows, Kalshi-priced headline bets with costs,
open-vs-Kalshi timing on B2B games.

    python research/h7_own_model/blend/verify_data/extra.py
"""
import numpy as np
import pandas as pd

import verify as v


def main():
    u = pd.read_csv(v.DATA / "upsets_dataset.csv", usecols=["game_id", "game_date", "season", "season_type", "home", "away",
                                                             "home_win", "mkt_open", "mkt_close"])
    pr = pd.read_csv(v.PREDS)[["GAME_ID", "p"]].rename(columns={"GAME_ID": "game_id", "p": "own"})
    o = pd.concat([pd.read_csv(v.DATA / f"espn_{s}.csv") for s in ("2023-24", "2024-25", "2025-26")], ignore_index=True)
    o["home"], o["away"] = o.home.replace(v.MAP), o.away.replace(v.MAP)
    o["tip"] = pd.to_datetime(o.date_utc, utc=True)
    o["game_date"] = o.tip.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    e = u.merge(pr, on="game_id").merge(o.drop(columns=["season", "season_type"]), on=["game_date", "home", "away"])
    e = e[e.mkt_open.notna() & e.home_ml_open.notna()].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    e["vig_open"] = v.am_p(e.home_ml_open) + v.am_p(e.away_ml_open) - 1
    print("vig_open > 0.08 rows:")
    print(e[e.vig_open > 0.08][["game_date", "home", "away", "home_ml_open", "away_ml_open", "home_ml_close",
                                "away_ml_close", "mkt_open", "mkt_close", "season"]].to_string())
    e["dh"], e["da"] = v.am_dec(e.home_ml_open), v.am_dec(e.away_ml_open)
    p = np.full(len(e), np.nan)
    for s in ("2024-25", "2025-26"):
        past, cur = (e.season < s).to_numpy(), (e.season == s).to_numpy()
        w = min(v.W, key=lambda w: v.ll(e.home_win.to_numpy()[past], v.bl(e.own.to_numpy()[past], e.mkt_open.to_numpy()[past], w)).mean())
        p[cur] = v.bl(e.own.to_numpy()[cur], e.mkt_open.to_numpy()[cur], w)
    e["blend"] = p
    t = e[e.blend.notna()].reset_index(drop=True)
    y, days, dh, da = t.home_win.to_numpy(), t.game_date.to_numpy(), t.dh.to_numpy(), t.da.to_numpy()
    sh, sa = v.value(t.blend.to_numpy(), dh, da)
    same = ((pd.to_numeric(t.home_ml_open) == pd.to_numeric(t.home_ml_close)) &
            (pd.to_numeric(t.away_ml_open) == pd.to_numeric(t.away_ml_close))).to_numpy()
    print(f"\ntest games with open == close: {same.sum()}; headline bets among them: {int((sh + sa)[same].sum())}")
    v.roi_line("headline bets, open==close games", sh * same, sa * same, y, dh, da, days)
    v.roi_line("headline bets, open!=close games", sh * ~same, sa * ~same, y, dh, da, days)
    hv = (t.vig_open > 0.08).to_numpy()
    v.roi_line("headline excluding vig_open>8% rows", sh * ~hv, sa * ~hv, y, dh, da, days)
    # largest single-bet contributions
    pnl = sh * np.where(y == 1, dh - 1, -1) + sa * np.where(y == 0, da - 1, -1)
    print(f"total P&L {pnl.sum():+.1f}u on {int((sh + sa).sum())} bets; top-10 winners sum {np.sort(pnl)[-10:].sum():+.1f}u; "
          f"max single {pnl.max():+.2f}u")
    # leave-out-the-biggest-odds robustness
    odds = np.where(sh == 1, dh, np.where(sa == 1, da, np.nan))
    for cap in (6, 4):
        k = ~(odds > cap)
        v.roi_line(f"headline excluding bets at decimal > {cap}", sh * k, sa * k, y, dh, da, days)

    # Kalshi 09 UTC priced, with costs
    k = pd.read_csv(v.DATA / "h6_timing" / "kalshi_snapshots.csv")[["event_id", "k_utc09", "k_prev_utc09", "k_h24"]]
    tk = t.merge(k, on="event_id")
    tk = tk[tk.k_utc09.notna()].reset_index(drop=True)
    yk, dk = tk.home_win.to_numpy(), tk.game_date.to_numpy()
    km = tk.k_utc09.to_numpy()
    sh3, sa3 = v.value(tk.blend.to_numpy(), tk.dh.to_numpy(), tk.da.to_numpy())
    fee = 0.01 + 0.07 * km * (1 - km)
    print(f"\nKalshi games {len(tk)}; headline bets there {int((sh3 + sa3).sum())}")
    v.roi_line("headline bets @ open ML", sh3, sa3, yk, tk.dh.to_numpy(), tk.da.to_numpy(), dk)
    v.roi_line("headline bets @ Kalshi 09 mid", sh3, sa3, yk, 1 / km, 1 / (1 - km), dk)
    v.roi_line("headline bets @ Kalshi 09 ask(+1c)+fee", sh3, sa3, yk, 1 / (km + fee), 1 / (1 - km + fee), dk)
    # only re-check value at the 09 price (bet only if still value at the 09 price incl cost)
    ph, pa = km + fee, 1 - km + fee
    still_h = sh3 * (tk.blend.to_numpy() / ph - 1 > 0)
    still_a = sa3 * ((1 - tk.blend.to_numpy()) / pa - 1 > 0)
    v.roi_line("headline bets still value at Kalshi 09 ask+fee", still_h, still_a, yk, 1 / ph, 1 / pa, dk)
    # how stale is the open vs Kalshi at 09 UTC game day / T-24h, B2B vs not
    allg = pd.concat([u[["game_date", "home"]].rename(columns={"home": "x"}), u[["game_date", "away"]].rename(columns={"away": "x"})])
    played = set(zip(allg.game_date, allg.x))
    prev = (pd.to_datetime(tk.game_date) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    b2b = np.array([((pv, h) in played) or ((pv, a) in played) for pv, h, a in zip(prev, tk.home, tk.away)])
    for nm, g in (("B2B", b2b), ("non-B2B", ~b2b)):
        d09 = np.nanmean(np.abs(tk.mkt_open - tk.k_utc09)[g]) * 100
        d24 = np.nanmean(np.abs(tk.mkt_open - tk.k_h24)[g]) * 100
        dpv = np.nanmean(np.abs(tk.mkt_open - tk.k_prev_utc09)[g]) * 100
        print(f"  {nm} ({g.sum()}): |open - kalshi D 09UTC| {d09:.2f} pts, |open - kalshi T-24h| {d24:.2f}, |open - kalshi D-1 09UTC| {dpv:.2f}")
        dll = v.ll(yk, tk.blend.to_numpy()) - v.ll(yk, km)
        print(f"     blend(own,open) - kalshi09 log-loss {dll[g].mean():+.5f}; open - kalshi09 {(v.ll(yk, tk.mkt_open.to_numpy()) - v.ll(yk, km))[g].mean():+.5f}")


if __name__ == "__main__":
    main()
