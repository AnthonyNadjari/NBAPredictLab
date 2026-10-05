"""Win-probability model.

Stats model: standardised logistic regression on a few robust features,
stored as plain JSON (no pickles, no library-version coupling).

Final probability: the betting market's de-vigged consensus when odds are
available. Backtests 2022-23..2025-26 (see research/): market 68.6% accuracy /
Brier 0.203 vs stats model 65.5% / 0.214; a fitted blend gave the stats model
~0 weight, so the market is used as-is. Without a market price the published
number is our own model (own_model.py, research H7); this logit is the last resort.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .features import MODEL_FEATURES

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "v2_model.json"


def fit(games: pd.DataFrame, C: float = 1.0) -> dict:
    from sklearn.linear_model import LogisticRegression

    d = games.dropna(subset=MODEL_FEATURES + ["home_win"])
    X = d[MODEL_FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=C, max_iter=5000).fit((X - mu) / sd, d.home_win.astype(int))
    return {
        "features": MODEL_FEATURES, "mean": mu.tolist(), "std": sd.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "n_games": int(len(d)), "seasons": sorted(d.season.unique().tolist()),
        "trained_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def save(model: dict, path: Path = MODEL_PATH) -> None:
    path.write_text(json.dumps(model, indent=2))


def load(path: Path = MODEL_PATH) -> dict:
    return json.loads(path.read_text())


def predict_proba(model: dict, rows: pd.DataFrame) -> np.ndarray:
    X = rows[model["features"]].to_numpy(float)
    X = np.nan_to_num((X - np.array(model["mean"])) / np.array(model["std"]))
    z = X @ np.array(model["coef"]) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def final_probability(model_prob: float, market_prob: Optional[float],
                      own_prob: Optional[float] = None) -> tuple:
    """(home win probability, source) used for the published pick.

    The market price when there is one; otherwise our own model (src/engine/own_model.py,
    research H7: clearly better than this logit, -0.011 log-loss over 4 seasons); the stats
    logit only if the own model could not run."""
    if market_prob is not None and 0.01 < market_prob < 0.99:
        return float(market_prob), "market"
    if own_prob is not None and 0.0 < own_prob < 1.0:
        return float(own_prob), "own"
    return float(model_prob), "model"
