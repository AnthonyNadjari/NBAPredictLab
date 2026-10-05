"""Player box-score store: data/player_games.csv, one row per player per finished game.

ESPN ids (game_id 'espn_<event>', player_id = ESPN athlete id). Fed by the daily run (games
finished in the last few days) and, once, by scripts/backfill_players.py. Preseason games are
kept (season_type 'Preseason'): the own model only reads them to see who changed team.
Plain CSV, like the game history, so it diffs and merges in git.
"""
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from . import espn
from .teams import TEAMS

log = logging.getLogger(__name__)

PLAYERS_PATH = Path(__file__).resolve().parents[2] / "data" / "player_games.csv"
COLUMNS = (["game_id", "game_date", "season", "season_type", "team", "opp", "home", "player_id", "player",
            "starter"] + espn.PLAYER_STATS)


def load(path: Path = PLAYERS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"game_id": str, "player_id": str})
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS]


def save(df: pd.DataFrame, path: Path = PLAYERS_PATH) -> None:
    df = df.drop_duplicates(["game_id", "player_id"], keep="last")
    df = df.sort_values(["game_date", "game_id", "team", "player_id"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path, index=False)


def fetch_days(df: pd.DataFrame, days: Iterable[date], sleep: float = 0.0,
               deadline: Optional[float] = None,
               checkpoint: Optional[Callable[[pd.DataFrame], None]] = None) -> pd.DataFrame:
    """Add the player lines of every finished game (preseason included) of the given US dates
    that is not in the store yet. Unreachable days / games are skipped (retried next run).

    sleep: pause after each ESPN call (polite rate limit); deadline: time.time() after which
    to stop early; checkpoint(df): called every 100 new games (backfill progress)."""
    known = set(df.game_id.astype(str))
    new, n_games, missing = [], 0, []

    def flush(cur):
        return pd.concat([cur, pd.DataFrame(new, columns=COLUMNS)], ignore_index=True) if new else cur

    for day in days:
        if deadline and time.time() > deadline:
            log.info("Players: time budget reached at %s", day)
            break
        try:
            games = espn.scoreboard(day, include_preseason=True)
        except RuntimeError as e:
            log.warning("%s", e)
            continue
        finally:
            time.sleep(sleep)
        for g in games:
            gid = f"espn_{g['event_id']}"
            if not g["completed"] or gid in known or g["home"] not in TEAMS or g["away"] not in TEAMS:
                continue     # (All-Star games and exhibitions vs non-NBA clubs are left out)
            rows = espn.player_box(g["event_id"])
            time.sleep(sleep)
            if rows is None:
                log.warning("Players: ESPN summary unavailable for %s (%s %s@%s)", gid, g["game_date"],
                            g["away"], g["home"])
                continue
            if not rows or len({r["team"] for r in rows}) < 2:
                missing.append(f"{g['game_date']} {g['away']}@{g['home']}")
                continue
            new.extend(rows)
            known.add(gid)
            n_games += 1
            if checkpoint and n_games % 100 == 0:
                df, new = flush(df), []
                checkpoint(df)
    if missing:
        # a renamed / broken player box score silently removes the player terms of the own model
        log.warning("Players: finished game(s) without player lines: %s", ", ".join(missing))
    if n_games:
        log.info("Players: +%d finished game(s) from ESPN", n_games)
    return flush(df)


def update_from_espn(df: pd.DataFrame, days_back: int = 7, today: Optional[date] = None) -> pd.DataFrame:
    """Daily update: finished games of the last `days_back` US dates missing from the store."""
    from .history import espn_today
    today = today or espn_today()
    return fetch_days(df, [today - timedelta(days=i) for i in range(days_back, 0, -1)])
