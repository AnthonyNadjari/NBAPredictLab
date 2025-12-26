"""
Pre-fill Player Stats Cache
Fetches and caches all player stats before training to speed up the process
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_fetcher import NBADataFetcher
from nba_api.stats.static import teams
import time


def prefill_cache():
    """Pre-fill player stats cache for all NBA teams"""
    print("=" * 60)
    print("PLAYER STATS CACHE PRE-FILL")
    print("=" * 60)
    print()

    fetcher = NBADataFetcher()
    all_teams = teams.get_teams()
    season = "2024-25"

    total_teams = len(all_teams)
    print(f"Pre-filling cache for {total_teams} teams...")
    print()

    for idx, team in enumerate(all_teams, 1):
        team_id = team['id']
        team_name = team['full_name']

        print(f"[{idx}/{total_teams}] {team_name}...", end=" ", flush=True)

        try:
            # This will fetch and cache all player stats for the team
            stats = fetcher.get_team_player_aggregated_stats(team_id, season)

            if stats and stats.get('active_players', 0) > 0:
                print(f"OK - Cached {stats['active_players']} players")
            else:
                print("WARN - No players found")

        except Exception as e:
            print(f"ERROR - {e}")

        # Rate limiting
        if idx < total_teams:
            time.sleep(1)  # Small delay between teams

    print()
    print("=" * 60)
    print("CACHE PRE-FILL COMPLETE")
    print("=" * 60)

    # Print cache statistics
    cache_stats = fetcher.player_cache.get_cache_stats()
    print(f"Players cached: {cache_stats['total_players']}")
    print(f"Team rosters cached: {cache_stats['total_rosters']}")
    print(f"Team stats cached: {cache_stats['total_team_stats']}")
    print()
    print("Ready for model training with player stats!")
    print()


if __name__ == "__main__":
    prefill_cache()
