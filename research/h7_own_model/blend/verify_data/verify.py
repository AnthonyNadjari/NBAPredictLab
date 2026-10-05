"""Independent re-check of blend/run.py + roi.py (data / leakage lens). Read-only on existing files.

    python research/h7_own_model/blend/verify_data/verify.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[3]          # research/
DATA = R / "data"
PREDS = R / "h7_own_model" / "ensemble" / "preds.csv"
MAP = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
W = np.round(np.arange(0, 1.0001, 0.05), 2)


def am_p(a):
    a = pd.to_numeric(a, errors="coerce").astype(float)
    return np.where(a < 0, -a / (100 - a), 100 / (a + 100))


def am_dec(a):
    a = pd.to_numeric(a, errors="coerce").astype(float)
    return np.where(a < 0, 1 + 100 / -a, 1 + a / 100)


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bl(o, m, w):
    return 1 / (1 + np.exp(-(w * lg(o) + (1 - w) * lg(m))))


def boot_ratio(num, den, days, n=4000, seed=1):
    u, inv = np.unique(days, return_inverse=True)
    sn, sd = np.bincount(inv, num), np.bincount(inv, den)
    rng = np.random.default_rng(seed)
    st = []
    for _ in range(n):
        k = rng.integers(0, len(u), len(u))
        st.append(sn[k].sum() / max(sd[k].sum(), 1e-9))
    return np.percentile(st, [2.5, 97.5])


def roi_line(name, sh, sa, y, dh, da, days):
    pnl = sh * np.where(y == 1, dh - 1, -1) + sa * np.where(y == 0, da - 1, -1)
    n = sh + sa
    if n.sum() == 0:
        print(f"  {name}: no bets"); return
    lo, hi = boot_ratio(pnl, n, days)
    print(f"  {name}: {int(n.sum())} bets ROI {100 * pnl.sum() / n.sum():+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]")


def value(p, dh, da, thr=0.0):
    evh, eva = p * dh - 1, (1 - p) * da - 1
    return ((evh > thr) & (evh >= eva)).astype(float), ((eva > thr) & (eva > evh)).astype(float)


def main():
    u = pd.read_csv(DATA / "upsets_dataset.csv", usecols=["game_id", "game_date", "season", "season_type", "home", "away",
                                                           "home_pts", "away_pts", "home_win", "mkt_open", "mkt_close"])
    pr = pd.read_csv(PREDS)
    print("== 1. join audit")
    print(f"upsets {len(u)} rows, dup game_id {u.game_id.duplicated().sum()}, dup (date,home,away) "
          f"{u.duplicated(['game_date', 'home', 'away']).sum()}; preds {len(pr)} rows, dup {pr.GAME_ID.duplicated().sum()}")
    chk = u.merge(pr, left_on="game_id", right_on="GAME_ID", suffixes=("", "_p"))
    bad = (chk.game_date != chk.date) | (chk.home != chk.home_p) | (chk.away != chk.away_p) | (chk.home_win != chk.home_win_p)
    print(f"preds vs upsets id/date/team/outcome mismatches: {int(bad.sum())} of {len(chk)}")
    print("season_type in upsets:", u.season_type.value_counts().to_dict())

    o = pd.concat([pd.read_csv(DATA / f"espn_{s}.csv") for s in ("2023-24", "2024-25", "2025-26")], ignore_index=True)
    o["home"], o["away"] = o.home.replace(MAP), o.away.replace(MAP)
    o["tip"] = pd.to_datetime(o.date_utc, utc=True)
    o["game_date"] = o.tip.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    nba = set(u.home) | set(u.away)
    o_nba = o[o.home.isin(nba) & o.away.isin(nba)]
    print(f"espn 2023-26 rows {len(o)}, non-NBA (all-star etc.) {len(o) - len(o_nba)}, "
          f"dup event_id {o.event_id.duplicated().sum()}, dup (date,home,away) {o_nba.duplicated(['game_date', 'home', 'away']).sum()}")
    j = u[u.season >= "2023-24"].merge(o_nba, on=["game_date", "home", "away"], how="left", indicator=True, suffixes=("", "_e"))
    print(f"upsets 2023-26 games {int((u.season >= '2023-24').sum())}: matched to ESPN {int((j._merge == 'both').sum())}, unmatched {int((j._merge == 'left_only').sum())}")
    # flipped home/away check
    fl = u[u.season >= "2023-24"].merge(o_nba, left_on=["game_date", "home", "away"], right_on=["game_date", "away", "home"])
    print(f"games found only with home/away flipped: {len(fl)}")
    m = j[j._merge == "both"].copy()
    sc_bad = (pd.to_numeric(m.home_score) != m.home_pts) | (pd.to_numeric(m.away_score) != m.away_pts)
    print(f"score mismatches ESPN vs upsets on matched games: {int(sc_bad.sum())}")
    st_map = {"Regular Season": 2, "Playoffs": 3, "PlayIn": 5, "Play-In": 5}
    print("season_type cross-tab:", pd.crosstab(m.season_type, m.season_type_e).to_dict())

    print("\n== 2. opening price quality")
    m["p_open_re"] = am_p(m.home_ml_open) / (am_p(m.home_ml_open) + am_p(m.away_ml_open))
    m["vig_open"] = am_p(m.home_ml_open) + am_p(m.away_ml_open) - 1
    d = (m.p_open_re - m.mkt_open).abs()
    print(f"mkt_open recomputed vs upsets: max abs diff {np.nanmax(d):.2e}; open missing {int(m.home_ml_open.isna().sum())}")
    print("vig_open quantiles:", np.nanpercentile(m.vig_open, [0, 1, 5, 50, 95, 99, 100]).round(4).tolist())
    print(f"vig_open <= 0: {int((m.vig_open <= 0).sum())}, > 0.10: {int((m.vig_open > 0.10).sum())}")
    raw_nonnum = m.home_ml_open.dropna().astype(str).str.match(r"^[+-]?\d+(\.\d+)?$")
    print(f"non-numeric open strings: {int((~raw_nonnum).sum())}  e.g. {m.home_ml_open.dropna().astype(str)[~raw_nonnum].unique()[:5]}")
    same = (pd.to_numeric(m.home_ml_open) == pd.to_numeric(m.home_ml_close)) & (pd.to_numeric(m.away_ml_open) == pd.to_numeric(m.away_ml_close))
    print(f"open == close exactly (both sides): {int(same.sum())} of {int(m.home_ml_open.notna().sum())}; by season "
          f"{m[same].season.value_counts().to_dict()}")
    m["mo"] = m.tip.dt.strftime("%Y-%m")
    m["absmove"] = (m.mkt_close - m.mkt_open).abs()
    print("book x season:", m.groupby(["season", "book"]).size().to_dict())
    print("mean |open-close| by book (2025-26):", m[m.season == "2025-26"].groupby("book").absmove.mean().round(4).to_dict())

    print("\n== 3. recompute headline (own = preds.csv, open = upsets.mkt_open, prices = espn open ML)")
    e = m.merge(pr[["GAME_ID", "p"]].rename(columns={"GAME_ID": "game_id", "p": "own"}), on="game_id")
    e = e[e.mkt_open.notna() & e.home_ml_open.notna()].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    e["dh"], e["da"] = am_dec(e.home_ml_open), am_dec(e.away_ml_open)
    # previous-day games (either team played on D-1): own model knows that result, the ~T-24h open may not
    allg = pd.concat([u[["game_date", "home"]].rename(columns={"home": "t"}), u[["game_date", "away"]].rename(columns={"away": "t"})])
    played = set(zip(allg.game_date, allg.t))
    prev = (pd.to_datetime(e.game_date) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    e["b2b"] = [((pv, h) in played) or ((pv, a) in played) for pv, h, a in zip(prev, e.home, e.away)]
    p = np.full(len(e), np.nan)
    for s in ("2024-25", "2025-26"):
        past, cur = (e.season < s).to_numpy(), (e.season == s).to_numpy()
        yp = e.home_win.to_numpy()[past]
        w = min(W, key=lambda w: ll(yp, bl(e.own.to_numpy()[past], e.mkt_open.to_numpy()[past], w)).mean())
        p[cur] = bl(e.own.to_numpy()[cur], e.mkt_open.to_numpy()[cur], w)
        print(f"  w(own) for {s} = {w} (fit on {past.sum()} games, seasons {sorted(e.season[past].unique())})")
    e["blend"] = p
    t = e[e.blend.notna()].reset_index(drop=True)
    y, days = t.home_win.to_numpy(), t.game_date.to_numpy()
    dll = ll(y, t.blend.to_numpy()) - ll(y, t.mkt_open.to_numpy())
    lo, hi = boot_ratio(dll, np.ones(len(dll)), days)
    print(f"  {len(t)} test games: blend-open log-loss {dll.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
    dh, da = t.dh.to_numpy(), t.da.to_numpy()
    fav = (t.mkt_open.to_numpy() >= .5).astype(float)
    roi_line("always open favourite", fav, 1 - fav, y, dh, da, days)
    sh, sa = value(t.blend.to_numpy(), dh, da)
    roi_line("blend value>0 (headline)", sh, sa, y, dh, da, days)

    print("\n== 4. timing: games where either team played the day before (open set ~T-24h, before that result)")
    for name, g in (("either team on B2B", t.b2b.to_numpy()), ("neither team on B2B", ~t.b2b.to_numpy())):
        lo, hi = boot_ratio(dll[g], np.ones(g.sum()), days[g])
        print(f" {name}: {g.sum()} games, blend-open {dll[g].mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
        roi_line("   blend value>0", sh * g, sa * g, y, dh, da, days)
    for s in ("2024-25", "2025-26"):
        g = (t.season == s).to_numpy()
        print(f" {s}: blend-open {dll[g].mean():+.5f}")
        roi_line("   blend value>0", sh * g, sa * g, y, dh, da, days)
    for st in sorted(t.season_type.unique()):
        g = (t.season_type == st).to_numpy()
        roi_line(f" {st} value>0", sh * g, sa * g, y, dh, da, days)
    g = (t.season == "2025-26").to_numpy()
    for b in t.book.unique():
        gb = g & (t.book == b).to_numpy()
        roi_line(f" 2025-26 book={b}", sh * gb, sa * gb, y, dh, da, days)
    # side / price buckets
    dogh = (t.mkt_open.to_numpy() < .5)
    on_dog = sh * dogh + sa * (~dogh)
    roi_line(" value bets on underdog", sh * dogh, sa * (~dogh), y, dh, da, days)
    roi_line(" value bets on favourite", sh * (~dogh), sa * dogh, y, dh, da, days)
    big = np.where(sh == 1, dh, np.where(sa == 1, da, 0)) >= 3.0
    roi_line(" value bets at decimal >= 3.0", sh * big, sa * big, y, dh, da, days)
    roi_line(" value bets at decimal < 3.0", sh * (~big), sa * (~big), y, dh, da, days)
    # what the open-to-close move did on our bets (CLV)
    clv_side = np.where(sh == 1, t.mkt_close - t.mkt_open, np.where(sa == 1, t.mkt_open - t.mkt_close, np.nan))
    print(f"  close moved toward our side on {np.nanmean(clv_side > 0) * 100:.1f}% of bets, away {np.nanmean(clv_side < 0) * 100:.1f}%, "
          f"mean {np.nanmean(clv_side) * 100:+.2f} pts")
    sh_c, sa_c = value(t.blend.to_numpy(), am_dec(t.home_ml_close), am_dec(t.away_ml_close))
    roi_line(" same blend, value>0 at CLOSING prices (not bettable at 09 UTC either)", sh_c, sa_c, y,
             am_dec(t.home_ml_close), am_dec(t.away_ml_close), days)

    print("\n== 5. bettable morning price: Kalshi 09:00 UTC mid (2025-26 + 24-25 playoffs)")
    k = pd.read_csv(DATA / "h6_timing" / "kalshi_snapshots.csv")[["event_id", "k_utc09", "k_close"]]
    tk = t.merge(k, on="event_id", how="inner")
    tk = tk[tk.k_utc09.notna()].reset_index(drop=True)
    yk, dk = tk.home_win.to_numpy(), tk.game_date.to_numpy()
    kb = tk.blend.to_numpy()
    d1 = ll(yk, kb) - ll(yk, tk.k_utc09.to_numpy())
    d2 = ll(yk, tk.mkt_open.to_numpy()) - ll(yk, tk.k_utc09.to_numpy())
    lo, hi = boot_ratio(d1, np.ones(len(d1)), dk)
    print(f"  {len(tk)} games: blend(own,open) - kalshi09 {d1.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]; open - kalshi09 {d2.mean():+.5f}")
    for w in (0.25, 0.45):
        b2 = bl(tk.own.to_numpy(), tk.k_utc09.to_numpy(), w)
        d3 = ll(yk, b2) - ll(yk, tk.k_utc09.to_numpy())
        lo, hi = boot_ratio(d3, np.ones(len(d3)), dk)
        print(f"  blend(own, kalshi09, w={w}) - kalshi09 {d3.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
    curve = {float(w): ll(yk, bl(tk.own.to_numpy(), tk.k_utc09.to_numpy(), w)).mean() for w in W}
    print(f"  in-sample best w(own) vs kalshi09 = {min(curve, key=curve.get)} (gain {min(curve.values()) - curve[0.0]:+.5f})")
    # value bets vs Kalshi 09 mid: fair odds 1/mid (no fee); with 1c half-spread + 0.07*p(1-p) fee
    km = tk.k_utc09.to_numpy()
    for lab, cost in (("at mid, no cost", 0.0), ("ask = mid+1c, +7%*p(1-p) fee", 1.0)):
        ph = np.clip(km + cost * (0.01 + 0.07 * km * (1 - km)), 0.01, 0.99)
        pa = np.clip((1 - km) + cost * (0.01 + 0.07 * km * (1 - km)), 0.01, 0.99)
        sh2, sa2 = value(bl(tk.own.to_numpy(), km, 0.45), 1 / ph, 1 / pa)
        roi_line(f" blend(own,kalshi09,w=.45) value>0 {lab}", sh2, sa2, yk, 1 / ph, 1 / pa, dk)
    # headline bets restricted to Kalshi games, priced at open vs at kalshi 09
    sh3, sa3 = value(kb, tk.dh.to_numpy(), tk.da.to_numpy())
    roi_line(" headline bets (Kalshi games) at OPEN price", sh3, sa3, yk, tk.dh.to_numpy(), tk.da.to_numpy(), dk)
    roi_line(" same bets at KALSHI 09 mid (no cost)", sh3, sa3, yk, 1 / km, 1 / (1 - km), dk)


if __name__ == "__main__":
    main()
