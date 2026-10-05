"""Skeptical re-check of blend/roi.py: CLV, robustness, model-free baselines, executable-price test.

    python research/h7_own_model/blend/verify_stats/verify.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BLEND = HERE.parent
sys.path.insert(0, str(BLEND))
from run import H7, DATA, W, ll, blend, logit  # noqa: E402
from roi import ESPN_TO_NBA, dec  # noqa: E402

RNG = np.random.default_rng(123)
NB = 2000


def am_p(ml):
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def power_devig(ph, pa):
    from scipy.optimize import brentq
    out = np.full(len(ph), np.nan)
    for i, (x, y) in enumerate(zip(ph, pa)):
        if np.isnan(x) or np.isnan(y):
            continue
        if x + y <= 1:
            out[i] = x / (x + y)
            continue
        k = brentq(lambda k: x ** k + y ** k - 1, 1, 10)
        out[i] = x ** k
    return out


def load():
    u = pd.read_csv(DATA / "upsets_dataset.csv", usecols=["game_id", "game_date", "season", "season_type", "home", "away",
                                                           "home_win", "mkt_open", "mkt_close",
                                                           "home_back_to_back", "away_back_to_back"])
    own = pd.read_csv(H7 / "ensemble" / "preds.csv")[["GAME_ID", "p"]].rename(columns={"GAME_ID": "game_id", "p": "own"})
    odds = pd.concat([pd.read_csv(DATA / f"espn_{s}.csv") for s in ("2023-24", "2024-25", "2025-26")])
    odds["game_date"] = pd.to_datetime(odds.date_utc).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    for c in ("home", "away"):
        odds[c] = odds[c].replace(ESPN_TO_NBA)
    odds = odds.dropna(subset=["home_ml_open", "away_ml_open"])[
        ["event_id", "game_date", "home", "away", "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close", "book"]]
    ev = u.merge(own, on="game_id").merge(odds, on=["game_date", "home", "away"], how="inner")
    ev = ev[ev.mkt_open.notna()].sort_values("game_date").reset_index(drop=True)
    ev["dh"], ev["da"] = dec(ev.home_ml_open), dec(ev.away_ml_open)
    ev["dhc"], ev["dac"] = dec(ev.home_ml_close), dec(ev.away_ml_close)
    ph, pa = am_p(ev.home_ml_close), am_p(ev.away_ml_close)
    ev["pc_mult"] = ph / (ph + pa)
    ev["pc_pow"] = power_devig(ph, pa)
    ph, pa = am_p(ev.home_ml_open), am_p(ev.away_ml_open)
    ev["po_mult"] = ph / (ph + pa)
    ev["po_pow"] = power_devig(ph, pa)
    return ev


def wf_blend(ev, mk="mkt_open", own="own"):
    p, ws = np.full(len(ev), np.nan), {}
    for s in ("2024-25", "2025-26"):
        past, cur = (ev.season < s).to_numpy(), (ev.season == s).to_numpy()
        yp = ev.home_win.to_numpy()[past]
        w = min(W, key=lambda w: ll(yp, blend(ev[own].to_numpy()[past], ev[mk].to_numpy()[past], w)).mean())
        p[cur], ws[s] = blend(ev[own].to_numpy()[cur], ev[mk].to_numpy()[cur], w), float(w)
    return p, ws


def stakes(p, dh, da, thr=0.0):
    evh, eva = p * dh - 1, (1 - p) * da - 1
    return ((evh > thr) & (evh >= eva)).astype(float), ((eva > thr) & (eva > evh)).astype(float)


def boot_ratio(num, den, days, n=NB):
    uniq, inv = np.unique(days, return_inverse=True)
    sn, sd = np.bincount(inv, num, len(uniq)), np.bincount(inv, den, len(uniq))
    k = RNG.integers(0, len(uniq), (n, len(uniq)))
    r = sn[k].sum(1) / np.maximum(sd[k].sum(1), 1e-9)
    return np.percentile(r, [2.5, 97.5]), r


def roi_line(e, sh, sa, label, show=True):
    y = e.home_win.to_numpy()
    pnl = sh * np.where(y == 1, e.dh - 1, -1) + sa * np.where(y == 0, e.da - 1, -1)
    n = sh + sa
    if n.sum() == 0:
        return None
    (lo, hi), _ = boot_ratio(pnl, n, e.game_date.to_numpy())
    r = pnl.sum() / n.sum()
    if show:
        print(f"  {label:48s} n={int(n.sum()):5d} ROI {100*r:+6.1f}% [{100*lo:+.1f}, {100*hi:+.1f}]")
    return r, pnl


def main():
    ev = load()
    p, ws = wf_blend(ev)
    m = ~np.isnan(p)
    e, p = ev[m].reset_index(drop=True), p[m]
    y = e.home_win.to_numpy()
    days = e.game_date.to_numpy()
    sh, sa = stakes(p, e.dh, e.da, 0.0)
    bet = (sh + sa) > 0
    side_home = sh > 0
    print(f"walk-forward w(own) {ws}; games {len(e)}; bets {int(bet.sum())}")
    print("books:", e.book.value_counts().to_dict())
    base = roi_line(e, sh, sa, "REPRODUCE blend value > 0%")

    # ---------------------------------------------------------------- 1. CLV
    print("\n=== 1. Closing line value (bets at open, vs main-book close of the same side) ===")
    has_c = e.dhc.notna() & e.dac.notna() & e.pc_mult.notna()
    b = bet & has_c.to_numpy()
    d_open = np.where(side_home, e.dh, e.da)
    d_close = np.where(side_home, e.dhc, e.dac)
    for meth in ("pc_mult", "pc_pow"):
        pcs = np.where(side_home, e[meth], 1 - e[meth])
        clv = pcs * d_open - 1  # EV of the bet at the open price if the close is the true prob
        (lo, hi), _ = boot_ratio(clv[b], np.ones(b.sum()), days[b])
        beat_fair = (d_open > 1 / pcs)[b].mean()
        print(f"  CLV (close de-vig {meth[3:]}): mean EV vs fair close {100*clv[b].mean():+.2f}% [{100*lo:+.2f}, {100*hi:+.2f}],"
              f" beat fair close {100*beat_fair:.1f}% of bets")
    print(f"  raw price: open dec > close dec on {100*(d_open > d_close)[b].mean():.1f}%, equal {100*(d_open == d_close)[b].mean():.1f}%,"
          f" worse {100*(d_open < d_close)[b].mean():.1f}%  (n={int(b.sum())})")
    po_side = np.where(side_home, e.mkt_open, 1 - e.mkt_open)
    pc_side = np.where(side_home, e.pc_mult, 1 - e.pc_mult)
    mv = (logit(pc_side) - logit(po_side))[b]
    print(f"  logit move open->close toward our side: mean {mv.mean():+.4f}, moved our way {100*(mv>0).mean():.1f}%, against {100*(mv<0).mean():.1f}%")
    # realised outcome vs close-implied: did bets win more than the close said?
    won = np.where(side_home, y == 1, y == 0)[b]
    print(f"  hit {100*won.mean():.1f}% vs fair-open {100*po_side[b].mean():.1f}% vs fair-close {100*pc_side[b].mean():.1f}% vs blend {100*np.where(side_home,p,1-p)[b].mean():.1f}%")
    # ROI decomposition: realised ROI = CLV + (luck vs close)
    pnl_b = np.where(won, d_open[b] - 1, -1)
    print(f"  realised ROI on these {int(b.sum())} bets {100*pnl_b.mean():+.2f}%; ROI if settled at close-fair EV {100*(pc_side[b]*d_open[b]-1).mean():+.2f}%;"
          f" remainder (outcome luck vs close) {100*(pnl_b.mean()-(pc_side[b]*d_open[b]-1).mean()):+.2f}%")
    # comparator: CLV of a random/all-underdog strategy, to see the baseline drift of open->close
    dog_home = (e.mkt_open < .5).to_numpy()
    pcd = np.where(dog_home, e.pc_mult, 1 - e.pc_mult)
    dod = np.where(dog_home, e.dh, e.da)
    hc = has_c.to_numpy()
    print(f"  baseline: all opening underdogs EV vs fair close {100*(pcd*dod-1)[hc].mean():+.2f}%; all favourites "
          f"{100*((1-pcd)*np.where(dog_home, e.da, e.dh)-1)[hc].mean():+.2f}%")
    # CLV by season
    for s in ("2024-25", "2025-26"):
        bs = b & (e.season == s).to_numpy()
        clv = (pc_side * d_open - 1)[bs]
        (lo, hi), _ = boot_ratio(clv, np.ones(bs.sum()), days[bs])
        print(f"    {s}: n={bs.sum()}, CLV {100*clv.mean():+.2f}% [{100*lo:+.2f}, {100*hi:+.2f}], realised ROI "
              f"{100*np.where(np.where(side_home,y==1,y==0), d_open-1, -1)[bs].mean():+.1f}%")

    # Kalshi as independent close (2025-26 subset)
    k = pd.read_csv(DATA / "h6_timing" / "kalshi_snapshots.csv")
    ek = e.merge(k[["event_id", "k_close", "k_utc09", "k_utc17", "k_prev_utc09", "k_h24"]], on="event_id", how="left")
    bk = b & ek.k_close.notna().to_numpy()
    pk = np.where(side_home, ek.k_close, 1 - ek.k_close)
    clvk = (pk * d_open - 1)[bk]
    (lo, hi), _ = boot_ratio(clvk, np.ones(bk.sum()), days[bk])
    print(f"  CLV vs Kalshi close (n={bk.sum()}): {100*clvk.mean():+.2f}% [{100*lo:+.2f}, {100*hi:+.2f}]")

    # ---------------------------------------------------------------- 2. robustness
    print("\n=== 2. Robustness of the value>0 strategy ===")
    pnl = np.where(bet, np.where(np.where(side_home, y == 1, y == 0), d_open - 1, -1), 0.0)
    for s in ("2024-25", "2025-26"):
        ms = (e.season == s).to_numpy()
        roi_line(e[ms], sh[ms], sa[ms], f"season {s}")
    print("  by month:")
    mon = pd.to_datetime(e.game_date).dt.strftime("%Y-%m").to_numpy()
    for mo in sorted(set(mon)):
        mm = (mon == mo) & bet
        if mm.sum():
            print(f"    {mo}: n={mm.sum():4d} ROI {100*pnl[mm].mean():+6.1f}% profit {pnl[mm].sum():+6.1f}u")
    fav_side = np.where(side_home, e.mkt_open >= .5, e.mkt_open < .5)
    for lab, mask in (("bets on opening favourite", fav_side), ("bets on opening underdog", ~fav_side)):
        mm = bet & mask
        (lo, hi), _ = boot_ratio(pnl[mm], np.ones(mm.sum()), days[mm])
        print(f"  {lab:28s} n={mm.sum():4d} ({100*mm.sum()/bet.sum():.0f}%) ROI {100*pnl[mm].mean():+.1f}% [{100*lo:+.1f}, {100*hi:+.1f}] profit {pnl[mm].sum():+.1f}u")
    for lab, mask in (("home", side_home), ("away", ~side_home)):
        mm = bet & mask
        print(f"  bets on {lab}: n={mm.sum()} ROI {100*pnl[mm].mean():+.1f}%")
    print("  by decimal price band:")
    bands = [1, 1.5, 2.0, 2.5, 3.0, 4.0, 100]
    tot = pnl[bet].sum()
    for lo_, hi_ in zip(bands[:-1], bands[1:]):
        mm = bet & (d_open >= lo_) & (d_open < hi_)
        if mm.sum():
            print(f"    [{lo_:.1f},{hi_:.1f}): n={mm.sum():4d} ROI {100*pnl[mm].mean():+6.1f}% profit {pnl[mm].sum():+6.1f}u ({100*pnl[mm].sum()/tot:+.0f}% of total)")
    mm = bet & (d_open >= 3.0)
    print(f"  share of profit from bets at >= 3.0: {100*pnl[mm].sum()/tot:.0f}%; profit excluding them: {pnl[bet & ~(d_open >= 3.0)].sum():+.1f}u on {(bet & ~(d_open >= 3.0)).sum()} bets")
    # top-k wins
    srt = np.sort(pnl[bet])[::-1]
    print(f"  total profit {tot:+.1f}u; top 10 wins sum {srt[:10].sum():+.1f}u; top 20 {srt[:20].sum():+.1f}u")
    # edge size bands
    edge = np.where(side_home, p * e.dh - 1, (1 - p) * e.da - 1)
    print("  by model edge:")
    for lo_, hi_ in ((0, .02), (.02, .04), (.04, .08), (.08, 1)):
        mm = bet & (edge > lo_) & (edge <= hi_)
        print(f"    edge ({lo_:.0%},{hi_:.0%}]: n={mm.sum():4d} ROI {100*pnl[mm].mean():+6.1f}%")
    # threshold grid + multiple testing: max-statistic over the grid under day-bootstrap null
    print("  threshold grid (multiple testing):")
    grid = [0, .01, .02, .03, .04, .05, .06, .08, .10]
    uniq, inv = np.unique(days, return_inverse=True)
    K = RNG.integers(0, len(uniq), (NB, len(uniq)))
    zs, boots = [], []
    for t in grid:
        a, bb = stakes(p, e.dh, e.da, t)
        r, pn = roi_line(e, a, bb, f"blend value > {t:.0%}")
        n = a + bb
        sp, sn = np.bincount(inv, pn, len(uniq)), np.bincount(inv, n, len(uniq))
        rb = sp[K].sum(1) / sn[K].sum(1)
        boots.append(rb - r)  # centred -> null distribution
        zs.append(r / rb.std())
    boots = np.array(boots)
    sds = boots.std(1, keepdims=True)
    maxnull = (boots / sds).max(0)
    zmax = max(zs)
    print(f"  best z over grid {zmax:.2f}; family-wise p (max-stat, day bootstrap) = {(maxnull >= zmax).mean():.3f};"
          f" single-test p for thr=0: {(boots[0]/sds[0] >= zs[0]).mean():.3f}")

    # ---------------------------------------------------------------- 3. alternative explanations
    print("\n=== 3. Model-free baselines at the same opening prices (2024-25+2025-26) ===")
    fav_home = (e.mkt_open >= .5).astype(float).to_numpy()
    roi_line(e, fav_home, 1 - fav_home, "always opening favourite")
    roi_line(e, 1 - fav_home, fav_home, "always opening underdog")
    roi_line(e, np.ones(len(e)), np.zeros(len(e)), "always home")
    roi_line(e, np.zeros(len(e)), np.ones(len(e)), "always away")
    for lo_ in (2.0, 2.5, 3.0, 4.0):
        hh = ((e.dh >= lo_) & (e.mkt_open < .5)).astype(float).to_numpy()
        aa = ((e.da >= lo_) & (e.mkt_open >= .5)).astype(float).to_numpy()
        roi_line(e, hh, aa, f"all underdogs at dec >= {lo_}")
    # mimic the blend's side mix with no model: dogs with the same price distribution
    # also same on 2023-24 (out of sample for any FLB story)
    e23 = ev[ev.season == "2023-24"].reset_index(drop=True)
    fh = (e23.mkt_open >= .5).astype(float).to_numpy()
    roi_line(e23, fh, 1 - fh, "2023-24 always opening favourite")
    roi_line(e23, 1 - fh, fh, "2023-24 always opening underdog")
    p_own = e.own.to_numpy()
    a, bb = stakes(p_own, e.dh, e.da, 0)
    roi_line(e, a, bb, "own model alone value > 0")
    # placebo: market open with the walk-forward shrink but own replaced by fallback-like Elo? use own shuffled within day
    print("  placebo: own prob shuffled across games of the same month (keeps dog-tilt of the blend weights):")
    rois = []
    for _ in range(200):
        sh_own = e.own.to_numpy().copy()
        for mo in set(mon):
            ix = np.where(mon == mo)[0]
            sh_own[ix] = sh_own[RNG.permutation(ix)]
        pp = blend(sh_own, e.mkt_open.to_numpy(), 0.35)
        a, bb = stakes(pp, e.dh, e.da, 0)
        r = roi_line(e, a, bb, "", show=False)
        rois.append(r[0])
    rois = np.array(rois)
    print(f"    placebo ROI mean {100*rois.mean():+.1f}%, 95% range [{100*np.percentile(rois,2.5):+.1f}, {100*np.percentile(rois,97.5):+.1f}],"
          f" share >= observed {100*(rois >= base[0]).mean():.1f}%")
    print("  de-vig method for the market input of the blend:")
    for col in ("po_mult", "po_pow"):
        ev2 = ev.copy()
        ev2["mk"] = ev2[col]
        pp, w2 = wf_blend(ev2, "mk")
        pp = pp[m]
        a, bb = stakes(pp, e.dh, e.da, 0)
        roi_line(e, a, bb, f"blend with open de-vig {col[3:]} w={w2}")
    # Calibration / vig asymmetry: favourite-longshot pattern in the open
    print("  realised win rate vs de-vigged open, by open prob band (side = the team at that prob):")
    side_p = np.concatenate([e.po_mult, 1 - e.po_mult])
    side_win = np.concatenate([y == 1, y == 0])
    side_pow = np.concatenate([e.po_pow, 1 - e.po_pow])
    for lo_, hi_ in ((0, .25), (.25, .4), (.4, .5), (.5, .6), (.6, .75), (.75, 1)):
        mm = (side_p >= lo_) & (side_p < hi_)
        print(f"    [{lo_:.2f},{hi_:.2f}): n={mm.sum():4d} mult {side_p[mm].mean():.3f} power {side_pow[mm].mean():.3f} realised {side_win[mm].mean():.3f}")

    # stale info: open set ~T-24h, before the previous night's games ended
    print("\n=== 3b. Stale-open check: bets involving a team on a back-to-back (open set before its D-1 game ended) ===")
    b2b = ((e.home_back_to_back == 1) | (e.away_back_to_back == 1)).to_numpy()
    for lab, mm in (("any team on B2B", bet & b2b), ("no B2B", bet & ~b2b)):
        clv = (pc_side * d_open - 1)[mm & hc]
        print(f"  {lab:16s} n={mm.sum():4d} ROI {100*pnl[mm].mean():+6.1f}% CLV {100*clv.mean():+.2f}%")
    # Executable test: same blend prob, priced at Kalshi 09:00 UTC (after D-1 games, when the bot runs)
    print("\n=== 3c. Executable price check: Kalshi at 09:00 UTC on game day (2025-26 regular + PO) ===")
    for col in ("k_prev_utc09", "k_h24", "k_utc09", "k_utc17"):
        kk = ek[col].to_numpy()
        ok = ~np.isnan(kk)
        # buy at mid + 1c, Kalshi taker fee 0.07*P*(1-P) per contract
        costh = np.clip(kk + 0.01, 0.01, 0.99)
        costa = np.clip(1 - kk + 0.01, 0.01, 0.99)
        costh = costh + 0.07 * costh * (1 - costh)
        costa = costa + 0.07 * costa * (1 - costa)
        dkh, dka = 1 / costh, 1 / costa
        eh_, ea_ = p * dkh - 1, (1 - p) * dka - 1
        hb = ok & (eh_ > 0) & (eh_ >= ea_)
        ab = ok & (ea_ > 0) & (ea_ > eh_)
        pn = np.where(hb, np.where(y == 1, dkh - 1, -1), 0) + np.where(ab, np.where(y == 0, dka - 1, -1), 0)
        nb = (hb | ab)
        (lo, hi), _ = boot_ratio(pn[nb], np.ones(nb.sum()), days[nb])
        # same games, at the ESPN open price
        po = pnl[ok & bet]
        print(f"  {col:13s} games {ok.sum():4d} bets {nb.sum():4d} ROI {100*pn[nb].mean():+6.1f}% [{100*lo:+.1f}, {100*hi:+.1f}]"
              f" | ESPN-open strategy on same games n={(ok & bet).sum()} ROI {100*po.mean():+.1f}%")
    # Kalshi utc09 vs ESPN open distance
    kk = ek.k_utc09.to_numpy()
    ok = ~np.isnan(kk)
    print(f"  mean |Kalshi 09UTC - ESPN open| = {np.abs(kk - e.mkt_open.to_numpy())[ok].mean():.4f};"
          f" log-loss open {ll(y[ok], e.mkt_open.to_numpy()[ok]).mean():.4f}, k09 {ll(y[ok], kk[ok]).mean():.4f}, blend {ll(y[ok], p[ok]).mean():.4f}")

    # ---------------------------------------------------------------- 4. selection-bias shrink
    print("\n=== 4. Shrink the own model toward the fallback (selection bias) and re-run ===")
    spec = importlib.util.spec_from_file_location("ens_run", H7 / "ensemble" / "run.py")
    ens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ens)
    fb = ens.fallback_and_elo()[["GAME_ID", "fallback"]].rename(columns={"GAME_ID": "game_id"})
    ev3 = ev.merge(fb, on="game_id", how="left")
    assert len(ev3) == len(ev)
    for keep in (1.0, 0.6, 0.4, 0.0):
        ev3["own_s"] = 1 / (1 + np.exp(-(logit(ev3.fallback) + keep * (logit(ev3.own) - logit(ev3.fallback)))))
        pp, w3 = wf_blend(ev3, "mkt_open", "own_s")
        pp = pp[m]
        a, bb = stakes(pp, e.dh, e.da, 0)
        r, _ = roi_line(e, a, bb, "", show=False)
        dll = ll(y, pp).mean() - ll(y, e.mkt_open.to_numpy()).mean()
        own_ll = ll(y, ev3.own_s.to_numpy()[m]).mean() - ll(y, e.mkt_open.to_numpy()).mean()
        print(f"  keep {keep:.0%} of own gain over fallback: own-open {own_ll:+.4f}, w={w3}, blend-open {dll:+.4f}, "
              f"value bets {int((a+bb).sum())} ROI {100*r:+.1f}%")


if __name__ == "__main__":
    main()
