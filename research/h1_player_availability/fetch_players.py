"""Download player game logs (one row per player-game actually played) with nba_api.

One LeagueGameLog call per season x season type, cached as CSV under
research/data/h1_player_availability/. Reruns skip cached files.
"""
import time
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "h1_player_availability"
OUT.mkdir(parents=True, exist_ok=True)
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TYPES = ["Regular Season", "PlayIn", "Playoffs"]


def fetch(season, stype, tries=4):
    from nba_api.stats.endpoints import leaguegamelog
    for k in range(tries):
        try:
            return leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star=stype, player_or_team_abbreviation="P",
                league_id="00", timeout=90).get_data_frames()[0]
        except Exception as e:  # network hiccups / throttling
            print(f"  retry {k + 1} after {type(e).__name__}: {e}")
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"failed {season} {stype}")


def main():
    for season in SEASONS:
        for stype in TYPES:
            f = OUT / f"player_logs_{season}_{stype.replace(' ', '')}.csv"
            if f.exists():
                continue
            df = fetch(season, stype)
            df["SEASON"], df["SEASON_TYPE"] = season, stype
            df.to_csv(f, index=False)
            print(season, stype, len(df), flush=True)
            time.sleep(1.5)
    frames = [pd.read_csv(f, dtype={"GAME_ID": str}) for f in sorted(OUT.glob("player_logs_*.csv"))]
    allp = pd.concat(frames, ignore_index=True)
    print("total player-games", len(allp))


if __name__ == "__main__":
    main()
