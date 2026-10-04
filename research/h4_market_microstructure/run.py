"""H4 - market microstructure. End-to-end:  python research/h4_market_microstructure/run.py

  1. (cached) re-download full per-book ESPN odds  -> research/data/h4_market_microstructure/books/
  2. build game + book tables                        -> research/data/h4_market_microstructure/*.csv
  3. run (a) longshot bias, (b) cross-book, (c) ML vs spread, (d) line movement,
     (e) popular teams / overreaction, (open) same ideas against the opening line
  4. write results/*.csv + results/summary.json and print the headline tables
Flags: --rebuild (rebuild cached game/book tables), --no-fetch (skip the download step)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import common as C  # noqa: E402
import a_longshot, b_books, c_ml_vs_spread, d_line_move, e_popular, o_open  # noqa: E402,E401


def all_features(g, books):
    """Kitchen-sink check over the 4 test seasons: every 4-season feature at once."""
    g = c_ml_vs_spread.add_spread_cols(e_popular.add_cols(g), books)
    g = g[g.spread.notna() & g.total.notna()].reset_index(drop=True)
    t = g[g.season.isin(C.TEST_SEASONS)]
    fx = lambda d: np.c_[np.abs(C.logit(d.p_close)), -d.spread_eff, -d.spread_eff * d.tot_c, d.pop10_diff,
                         d.pop10_diff * d.fav_sign, d.bw_diff, d.bl_diff, d.hot_diff, d.cold_diff]
    p = C.walk_forward(g, fx)
    platt = C.walk_forward(g, None)
    r = C.compare(t, p[t.index].to_numpy(), t.p_close.to_numpy(), "ALL 4-season features together")
    r["delta_vs_platt"] = C.compare(t, p[t.index].to_numpy(), platt[t.index].to_numpy(), "x")["delta"]
    bets = [C.bet_roi(t, p[t.index].to_numpy(), th, label=f"ALL-features model edge>{th}") for th in (0.0, 0.02)]
    return [{"part": "all", **r}], [{"part": "all", **b} for b in bets]


def main():
    t0 = time.time()
    args = set(sys.argv[1:])
    if "--rebuild" in args:
        for f in ("games.csv", "books.csv"):
            (C.CACHE / f).unlink(missing_ok=True)
    if "--no-fetch" not in args:
        import fetch_books
        fetch_books.main()
    g = C.load_games()
    books = C.load_books(g)
    print(f"games {len(g)} | book rows {len(books)} | juice fields: {'h_so' in books.columns}")

    LL, BETS, TABLES = [], [], {}
    for name, fn in [("a", lambda: a_longshot.run(g)), ("b", lambda: b_books.run(g, books)),
                     ("c", lambda: c_ml_vs_spread.run(g, books)), ("d", lambda: d_line_move.run(g, books)),
                     ("e", lambda: e_popular.run(g)), ("open", lambda: o_open.run(g, books))]:
        o = fn()
        LL += o["logloss"]
        BETS += o["bets"]
        TABLES.update(o["tables"])
        print(f"part {name} done ({time.time() - t0:.0f}s)", flush=True)
    ll2, b2 = all_features(g, books)
    LL += ll2
    BETS += b2

    ll = pd.DataFrame(LL)
    bets = pd.DataFrame(BETS)
    # reference rows (not hypotheses): close-vs-open is a sanity comparison
    ll["is_reference"] = ll.variant.str.contains("reference")
    hyp = ll[~ll.is_reference]
    real_bets = bets[bets.get("oracle", pd.Series(False, index=bets.index)).fillna(False) != True]  # noqa: E712
    for name, df in TABLES.items():
        df.to_csv(C.RES / f"{name}.csv", index=False)
    ll.to_csv(C.RES / "logloss_variants.csv", index=False)
    bets.to_csv(C.RES / "betting_rules.csv", index=False)

    close_hyp = hyp[hyp.get("baseline", pd.Series(np.nan, index=hyp.index)).isna()]
    open_hyp = hyp[hyp.get("baseline", pd.Series(np.nan, index=hyp.index)) == "open"]
    best = close_hyp.sort_values("delta").iloc[0]
    bestb = real_bets[real_bets.bets >= 100].sort_values("roi", ascending=False).iloc[0]
    summary = {
        "n_logloss_variants": int(len(hyp)), "n_betting_rules": int(len(real_bets)),
        "bonferroni_k_used": {"logloss": C.BONF_K_LL, "bets": C.BONF_K_BET},
        "n_passing_protocol": int(hyp.passes.sum()), "n_passing_bonferroni": int(hyp.passes_bonf.sum()),
        "best_close_variant": best.dropna().to_dict(),
        "median_close_variant_delta": float(close_hyp.delta.median()),
        "best_open_variant": open_hyp.sort_values("delta").iloc[0].dropna().to_dict() if len(open_hyp) else None,
        "median_open_variant_delta": float(open_hyp.delta.median()) if len(open_hyp) else None,
        "bets_with_ci_above_zero": real_bets[real_bets.roi_lo > 0][["part", "rule", "bets", "roi", "roi_lo", "roi_hi"]]
        .to_dict("records"),
        "best_bet_rule_100plus": bestb.dropna().to_dict(),
    }
    (C.RES / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    cols = ["part", "variant", "n", "ll_base", "delta", "ci_lo", "ci_hi", "ci_lo_bonf", "ci_hi_bonf",
            "seasons_neg", "seasons", "delta_vs_platt", "passes", "passes_bonf"]
    print("\n=== log-loss: model minus market (negative = better than market) ===")
    print(ll[[c for c in cols if c in ll.columns]].round(5).to_string(index=False))
    print("\n=== per-season deltas ===")
    print(ll[["part", "variant"] + [c for c in ll.columns if c.startswith("d_")]].round(5).to_string(index=False))
    bcols = ["part", "rule", "bets", "roi", "roi_lo", "roi_hi", "roi_lo_bonf", "roi_hi_bonf"] + \
        [c for c in bets.columns if c.startswith("roi_20")]
    print("\n=== betting rules (flat 1u at actual odds incl. vig) ===")
    print(bets[bcols].round(4).to_string(index=False))
    print(f"\n{len(hyp)} log-loss variants, {len(real_bets)} betting rules | passing protocol: "
          f"{int(hyp.passes.sum())} | passing Bonferroni: {int(hyp.passes_bonf.sum())} | "
          f"bets with 95% CI > 0: {len(summary['bets_with_ci_above_zero'])}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
