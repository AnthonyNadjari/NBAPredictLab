#!/usr/bin/env python3
"""One-off backfill of data/player_games.csv (player box scores, ESPN ids) for the own model.

    python scripts/backfill_players.py [--seasons 2024-25 2025-26] [--sleep 0.6]
                                       [--budget-minutes 150] [--commit]

Scans every US date of the given seasons (first to last game date in data/games_history.csv),
then the current season from September 25 to yesterday (preseason included: it shows who moved
team over the summer). For each finished game not in the store yet: one ESPN `summary` call.
About 480 scoreboard + 2,700 summary calls for two seasons, ~50 min at the default pace.

Resumable: games already in the store are skipped, days already covered are not re-scanned
(except the last one), progress is saved every 100 games, and the run stops cleanly when the
time budget is spent; re-run the workflow to continue. --commit commits and pushes the csv
(merging with the remote's rows if the daily run pushed in between).
Run by .github/workflows/backfill_players.yml (manual).
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.engine import history, player_store  # noqa: E402
from src.engine.espn import season_label  # noqa: E402
from src.engine.teams import TEAMS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("backfill_players")
STORE = "data/player_games.csv"


def days_to_scan(seasons, store: pd.DataFrame, today: date):
    hist = history.load()
    days = []
    for s in seasons:
        d = hist[hist.season == s].game_date
        if d.empty:
            log.warning("No %s games in data/games_history.csv: season skipped", s)
            continue
        lo, hi = date.fromisoformat(d.min()), date.fromisoformat(d.max())
        days += [lo + timedelta(days=i) for i in range((hi - lo).days + 1)]
    cur = season_label(today)
    start = date(int(cur[:4]), 9, 25)
    days += [start + timedelta(days=i) for i in range((today - start).days)]
    days = sorted(set(days))
    # a day is covered when the store has as many games as the history for it (a game whose
    # ESPN summary failed on an earlier run is retried); days the history does not know
    # (preseason) when the store has any game of them
    nba = hist[hist.home.isin(TEAMS) & hist.away.isin(TEAMS)]
    want = nba.groupby("game_date").size()
    st = store[store.season_type.astype(str) != "Preseason"]
    have = st.drop_duplicates("game_id").groupby(st.game_date.astype(str)).size() if len(st) else pd.Series(dtype=int)
    any_rows = set(store.game_date.astype(str))
    covered = {d for d in any_rows if d not in want.index or have.get(d, 0) >= want[d]}
    last = max((d for d in days if d.isoformat() in covered), default=None)
    todo = [d for d in days if d.isoformat() not in covered or d == last]
    log.info("%d days in range, %d to scan (%d already covered)", len(days), len(todo), len(days) - len(todo))
    return todo


def git(*args, check=False):
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r


def commit_and_push(message: str) -> bool:
    if os.environ.get("GITHUB_ACTIONS"):
        git("config", "user.name", "GitHub Actions Bot")
        git("config", "user.email", "actions@github.com")
    for attempt in range(1, 4):
        git("add", STORE)
        if git("diff", "--cached", "--quiet").returncode == 0:
            log.info("Nothing to commit")
            return True
        git("commit", "-m", message, check=True)
        if git("push", "origin", "HEAD:main").returncode == 0:
            log.info("Pushed %s", STORE)
            return True
        log.info("Push rejected (attempt %d): merging with the remote rows", attempt)
        tmp = Path(tempfile.mkdtemp()) / "player_games.csv"
        shutil.copy2(PROJECT_ROOT / STORE, tmp)
        git("fetch", "origin", "main", check=True)
        git("reset", "--hard", "origin/main", check=True)
        player_store.save(pd.concat([player_store.load(), player_store.load(tmp)], ignore_index=True))
    log.error("Could not push after 3 attempts")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    ap.add_argument("--sleep", type=float, default=0.6, help="seconds between ESPN calls")
    ap.add_argument("--budget-minutes", type=float, default=150.0)
    ap.add_argument("--commit", action="store_true", help="commit + push the csv at the end")
    args = ap.parse_args()

    t0 = time.time()
    store = player_store.load()
    n0 = store.game_id.nunique()
    todo = days_to_scan(args.seasons, store, history.espn_today())
    store = player_store.fetch_days(store, todo, sleep=args.sleep, deadline=t0 + 60 * args.budget_minutes,
                                    checkpoint=player_store.save)
    player_store.save(store)
    store = player_store.load()
    done = store.game_id.nunique()
    log.info("Store: %d games (+%d), %d player rows, %s .. %s, %.0f min", done, done - n0, len(store),
             store.game_date.min(), store.game_date.max(), (time.time() - t0) / 60)
    per = store.drop_duplicates("game_id").groupby(["season", "season_type"]).size()
    log.info("Games per season / type:\n%s", per.to_string())
    if args.commit:
        ok = commit_and_push(f"Player box scores backfill: {done} games ({store.game_date.min()} .. "
                             f"{store.game_date.max()})")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
