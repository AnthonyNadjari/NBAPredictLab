"""Black-box leak test of the three H7 families (and therefore of the ensemble built on them).

    python research/h7_own_model/ensemble/leak_test.py [--date 2024-01-15] [--tmp DIR]   (~8 min)

Idea: a pre-game model's predictions for every game on or before date D must not move when
everything that happens ON or AFTER D is replaced by garbage. So we build a scrambled copy of
the data and re-run each family's own, unmodified run.py on it (only its DATA / ROOT paths are
redirected to the copy), then compare its preds.csv with the published one:

  * team_logs.csv: every game on/after D gets the box score (both team rows) of another random
    game on/after D -> scores, margins, possessions, 3PM, W/L of D and later are wrong;
  * games_features_odds.csv: REBUILT from the scrambled team logs with research/features.py,
    so the pre-game features of games after D are consistent with the scrambled outcomes (a model
    reading a later game's Elo or a season aggregate would be caught);
  * player logs: on/after D, 15% of player rows dropped (who played changes) and all stat
    columns (MIN, PTS, +/-, ...) shuffled across rows;
  * rotation_player_games.csv (H1, used by margin_ratings) rebuilt from the scrambled logs with
    the H1 builder; elo_plus rebuilds its own availability cache (--rebuild);
  * ALL odds columns (games_features_odds mkt/spread, ESPN moneylines, upsets_dataset
    mkt_open/mkt_close) are replaced by random numbers everywhere, NaN pattern kept: if any odds
    column were an input, predictions before D would move too.
Pass = max |p_scrambled - p_published| < TOL on all games dated <= D, while games after D do
move (proof that the scramble reached the models).

Why TOL = 1e-3 and not 0: player_impact refits sklearn LogisticRegression (lbfgs, tol 1e-4) on
float inputs. Re-deriving games_features_odds with features.py on the UNSCRAMBLED logs (values
equal to 1e-14) already moves its published probabilities by up to 5e-4; on the scrambled run
its pre-D player composites differ from the clean ones by <= 4e-14 (row order of same-date
rows in a non-stable sort) and its Kalman ratings, player rates and P(plays) table are bitwise
identical before D. An information leak shows up as differences of 1e-2..1e-1 (games after D:
~0.15). margin_ratings and elo_plus are bitwise identical (0.0) before D.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
H7 = HERE.parent
ROOT = H7.parents[1]
DATA = ROOT / "research" / "data"
FAMILIES = ["player_impact", "margin_ratings", "elo_plus"]
TOL = 1e-3   # optimiser-tolerance noise, see docstring
STAT_TEAM = ["WL", "MIN", "PTS", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
             "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PLUS_MINUS"]
STAT_PLAYER = ["MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB",
               "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "FANTASY_PTS"]


def scramble(tmp: Path, D: pd.Timestamp, rng):
    out = tmp / "data"
    (out / "h1_player_availability").mkdir(parents=True, exist_ok=True)

    # ---- team logs: whole-game box-score swap among games on/after D
    t = pd.read_csv(DATA / "team_logs.csv", dtype={"GAME_ID": str})
    t = t.drop_duplicates(["TEAM_ID", "GAME_ID"]).reset_index(drop=True)
    t["_d"] = pd.to_datetime(t.GAME_DATE)
    t["_home"] = t.MATCHUP.str.contains(" vs. ")
    t["_slot"] = t.sort_values(["_home", "TEAM_ID"], ascending=[False, True]).groupby("GAME_ID").cumcount()
    n_rows = t.groupby("GAME_ID").GAME_ID.transform("size")
    idx = (t._d >= D) & (n_rows == 2)
    late = t[idx]
    gids = late.GAME_ID.unique()
    src = dict(zip(gids, rng.permutation(gids)))
    stats = late.set_index(["GAME_ID", "_slot"])[STAT_TEAM]
    keys = list(zip(t.loc[idx, "GAME_ID"].map(src), t.loc[idx, "_slot"]))
    t[STAT_TEAM] = t[STAT_TEAM].astype(object)
    t.loc[idx, STAT_TEAM] = stats.reindex(keys).to_numpy()
    assert t.loc[idx, ["PTS", "MIN", "PLUS_MINUS"]].notna().all().all()
    t.drop(columns=["_d", "_home", "_slot"]).to_csv(out / "team_logs.csv", index=False)
    print(f"  team logs: {idx.sum()} rows scrambled ({len(gids)} games on/after {D.date()})")

    # ---- games_features_odds rebuilt from the scrambled logs (leak-free builder), odds randomised
    sys.path.insert(0, str(ROOT / "research"))
    from features import build
    g = build(pd.read_csv(out / "team_logs.csv"))
    o = pd.read_csv(DATA / "games_features_odds.csv", usecols=["GAME_ID", "mkt", "spread", "book"])
    g = g.merge(o, on="GAME_ID", how="left")
    g["mkt"] = np.where(g.mkt.notna(), rng.uniform(0.05, 0.95, len(g)), np.nan)
    g["spread"] = np.where(g.spread.notna(), rng.normal(0, 8, len(g)).round(1), np.nan)
    ref = pd.read_csv(DATA / "games_features_odds.csv", nrows=1).columns
    g[list(ref)].to_csv(out / "games_features_odds.csv", index=False)

    # ---- upsets_dataset: market columns randomised (evaluation set definition unchanged)
    u = pd.read_csv(DATA / "upsets_dataset.csv")
    for c in ("mkt", "mkt_open", "mkt_close", "p", "fav_p"):
        if c in u:
            u[c] = np.where(u[c].notna(), rng.uniform(0.05, 0.95, len(u)), np.nan)
    u.to_csv(out / "upsets_dataset.csv", index=False)

    # ---- ESPN odds: moneylines randomised
    for f in sorted(DATA.glob("espn_*.csv")):
        e = pd.read_csv(f)
        for c in ("home_ml", "away_ml", "home_ml_open", "away_ml_open", "home_ml_close", "away_ml_close"):
            v = pd.to_numeric(e[c], errors="coerce")
            e[c] = np.where(v.notna(), rng.choice([-400, -250, -150, -110, 120, 180, 300], len(e)), np.nan)
        e.to_csv(out / f.name, index=False)

    # ---- player logs on/after D: drop 15% of rows, shuffle stats across rows
    for f in sorted((DATA / "h1_player_availability").glob("player_logs_*.csv")):
        p = pd.read_csv(f, dtype={"GAME_ID": str})
        late = pd.to_datetime(p.GAME_DATE) >= D
        if late.any():
            keep = ~late | (rng.random(len(p)) > 0.15)
            p = p[keep].copy()
            late = pd.to_datetime(p.GAME_DATE) >= D
            perm = rng.permutation(np.flatnonzero(late.to_numpy()))
            p.loc[late, STAT_PLAYER] = p.iloc[perm][STAT_PLAYER].to_numpy()
        p.to_csv(out / "h1_player_availability" / f.name, index=False)

    # ---- H1 rotation table rebuilt from the scrambled logs (same builder as the original)
    sys.path.insert(0, str(ROOT / "research" / "h1_player_availability"))
    import build as h1
    h1.CACHE, h1.DATA = out / "h1_player_availability", out
    p, tl = h1.player_logs(), h1.team_logs()
    h1.REPL = h1.replacement_rate(p)
    a = h1.availability(p, tl, h1.player_values(p, tl, h1.REPL))
    a.to_csv(out / "h1_player_availability" / "rotation_player_games.csv", index=False)
    print("  player logs + H1 rotation table scrambled / rebuilt")
    return out


def patched_copy(fam: str, tmp: Path, data: Path) -> Path:
    src = (H7 / fam / "run.py").read_text(encoding="utf-8")
    rep = {'ROOT = HERE.parents[2]': f'ROOT = Path(r"{ROOT}")',
           'DATA = ROOT / "research" / "data"': f'DATA = Path(r"{data}")'}
    for a, b in rep.items():
        assert src.count(a) == 1, (fam, a)
        src = src.replace(a, b)
    d = tmp / fam
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(src, encoding="utf-8")
    return d / "run.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2024-01-15")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    D = pd.Timestamp(args.date)
    tmp = Path(args.tmp) if args.tmp else Path(tempfile.mkdtemp(prefix="h7_leak_"))
    t0 = time.time()
    print(f"scrambling everything on/after {D.date()} into {tmp}")
    data = scramble(tmp, D, np.random.default_rng(11))
    procs = {}
    for fam in FAMILIES:
        script = patched_copy(fam, tmp, data)
        cmd = [sys.executable, str(script)] + (["--rebuild"] if fam == "elo_plus" else [])
        procs[fam] = subprocess.Popen(cmd, stdout=open(tmp / f"{fam}.log", "w", encoding="utf-8"),
                                      stderr=subprocess.STDOUT)
    for fam, pr in procs.items():
        pr.wait()
        print(f"  {fam}: exit {pr.returncode} ({time.time() - t0:.0f}s)")
    lines = []
    for fam in FAMILIES:
        a = pd.read_csv(H7 / fam / "preds.csv")
        b = pd.read_csv(tmp / fam / "preds.csv")
        m = a.merge(b, on="GAME_ID", suffixes=("", "_s"))
        d = (m.p - m.p_s).abs()
        pre = pd.to_datetime(m.date) <= D
        lines.append(f"{fam:15s} games<=D {pre.sum():5d}  max|diff| {d[pre].max():.2e}   "
                     f"games>D {(~pre).sum():5d}  mean|diff| {d[~pre].mean():.4f}  "
                     f"{'PASS' if d[pre].max() < TOL and d[~pre].mean() > 1e-2 else 'FAIL'}")
    print("\n".join(lines))
    (HERE / "leak_test_result.txt").write_text(f"scramble date {D.date()}\n" + "\n".join(lines) + "\n",
                                                encoding="utf-8")
    if not args.tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
