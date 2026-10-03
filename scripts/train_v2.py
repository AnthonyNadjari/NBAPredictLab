#!/usr/bin/env python3
"""Train the v2 stats model on data/games_history.csv and report a walk-forward backtest.

Usage:
    python scripts/train_v2.py            # backtest + train on all seasons + save
    python scripts/train_v2.py --no-save  # backtest only
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import features, history, model  # noqa: E402

MIN_SEASON = "2019-20"


def metrics(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"n": len(y), "acc": float(((p > .5) == y).mean()), "brier": float(((p - y) ** 2).mean()),
            "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    g = features.build(history.load())
    g = g[g.home_win.notna() & (g.season >= MIN_SEASON)]
    seasons = sorted(g.season.unique())
    print("Walk-forward (train on earlier seasons, test on next):")
    for s in seasons[-4:]:
        m = model.fit(g[g.season < s])
        te = g[g.season == s]
        r = metrics(te.home_win.to_numpy(), model.predict_proba(m, te))
        print(f"  {s}: n={r['n']} acc={r['acc']:.3f} brier={r['brier']:.4f} logloss={r['logloss']:.4f}")

    final = model.fit(g)
    print("Coefficients:", dict(zip(final["features"], np.round(final["coef"], 3))))
    if not args.no_save:
        model.save(final)
        print(f"Saved {model.MODEL_PATH}")


if __name__ == "__main__":
    main()
