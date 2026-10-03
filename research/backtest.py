"""Walk-forward backtest: train on all seasons before S, test on season S."""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from features import build  # noqa: E402

FEATS = ["elo_diff", "net_diff", "form_diff", "rest_diff", "b2b_diff", "g5_diff"]
TEST_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


def score(name, y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"model": name, "n": len(y), "acc": ((p > .5) == y).mean(),
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p)}


def run(g, feats=FEATS, C=1.0):
    rows = []
    for s in TEST_SEASONS:
        tr = g[(g.season < s) & (g.season >= "2019-20")]
        te = g[g.season == s]
        m = LogisticRegression(C=C, max_iter=2000).fit(tr[feats], tr.home_win)
        p = m.predict_proba(te[feats])[:, 1]
        rows.append({**score("logit", te.home_win, p), "season": s})
        rows.append({**score("elo", te.home_win, te.elo_prob), "season": s})
        rows.append({**score("home", te.home_win, np.full(len(te), tr.home_win.mean())), "season": s})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logs = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "research/data/team_logs.csv")
    g = build(logs)
    g.to_csv("research/data/games_features.csv", index=False)
    r = run(g)
    print(r.groupby("model")[["acc", "brier", "logloss"]].mean().round(4))
    print(r.pivot(index="season", columns="model", values="acc").round(3))
