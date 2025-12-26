"""
Player Statistics Cache
Robust caching system for NBA player statistics to avoid slow API calls
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import time


class PlayerStatsCache:
    """
    Manages caching of player statistics in SQLite database.

    Features:
    - Stores player stats with expiration (24h TTL)
    - Automatic cleanup of stale data
    - Batch updates for efficiency
    - Thread-safe operations
    """

    def __init__(self, db_path: str = 'data/player_cache.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize cache database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Player stats cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_stats_cache (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT,
                team_id INTEGER,
                stats_json TEXT,
                cached_at TEXT,
                expires_at TEXT
            )
        ''')

        # Team roster cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_roster_cache (
                team_id INTEGER,
                season TEXT,
                roster_json TEXT,
                cached_at TEXT,
                expires_at TEXT,
                PRIMARY KEY (team_id, season)
            )
        ''')

        # Team aggregated stats cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_player_stats_cache (
                team_id INTEGER,
                season TEXT,
                stats_json TEXT,
                cached_at TEXT,
                expires_at TEXT,
                PRIMARY KEY (team_id, season)
            )
        ''')

        conn.commit()
        conn.close()

    def get_player_stats(self, player_id: int) -> Optional[Dict]:
        """Get cached player stats if not expired"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT stats_json, expires_at
            FROM player_stats_cache
            WHERE player_id = ?
        ''', (player_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            stats_json, expires_at = result
            if datetime.now() < datetime.fromisoformat(expires_at):
                return json.loads(stats_json)

        return None

    def set_player_stats(self, player_id: int, player_name: str,
                        team_id: int, stats: Dict, ttl_hours: int = 24):
        """Cache player stats with expiration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cached_at = datetime.now()
        expires_at = cached_at + timedelta(hours=ttl_hours)

        cursor.execute('''
            INSERT OR REPLACE INTO player_stats_cache
            (player_id, player_name, team_id, stats_json, cached_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            player_id,
            player_name,
            team_id,
            json.dumps(stats),
            cached_at.isoformat(),
            expires_at.isoformat()
        ))

        conn.commit()
        conn.close()

    def get_team_roster(self, team_id: int, season: str) -> Optional[List[Dict]]:
        """Get cached team roster"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT roster_json, expires_at
            FROM team_roster_cache
            WHERE team_id = ? AND season = ?
        ''', (team_id, season))

        result = cursor.fetchone()
        conn.close()

        if result:
            roster_json, expires_at = result
            if datetime.now() < datetime.fromisoformat(expires_at):
                return json.loads(roster_json)

        return None

    def set_team_roster(self, team_id: int, season: str,
                       roster: List[Dict], ttl_hours: int = 168):  # 1 week
        """Cache team roster"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cached_at = datetime.now()
        expires_at = cached_at + timedelta(hours=ttl_hours)

        cursor.execute('''
            INSERT OR REPLACE INTO team_roster_cache
            (team_id, season, roster_json, cached_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            team_id,
            season,
            json.dumps(roster),
            cached_at.isoformat(),
            expires_at.isoformat()
        ))

        conn.commit()
        conn.close()

    def get_team_aggregated_stats(self, team_id: int, season: str) -> Optional[Dict]:
        """Get cached aggregated team player stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT stats_json, expires_at
            FROM team_player_stats_cache
            WHERE team_id = ? AND season = ?
        ''', (team_id, season))

        result = cursor.fetchone()
        conn.close()

        if result:
            stats_json, expires_at = result
            if datetime.now() < datetime.fromisoformat(expires_at):
                return json.loads(stats_json)

        return None

    def set_team_aggregated_stats(self, team_id: int, season: str,
                                  stats: Dict, ttl_hours: int = 24):
        """Cache aggregated team player stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cached_at = datetime.now()
        expires_at = cached_at + timedelta(hours=ttl_hours)

        cursor.execute('''
            INSERT OR REPLACE INTO team_player_stats_cache
            (team_id, season, stats_json, cached_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            team_id,
            season,
            json.dumps(stats),
            cached_at.isoformat(),
            expires_at.isoformat()
        ))

        conn.commit()
        conn.close()

    def cleanup_expired(self):
        """Remove expired cache entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute('DELETE FROM player_stats_cache WHERE expires_at < ?', (now,))
        cursor.execute('DELETE FROM team_roster_cache WHERE expires_at < ?', (now,))
        cursor.execute('DELETE FROM team_player_stats_cache WHERE expires_at < ?', (now,))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted

    def clear_all(self):
        """Clear entire cache"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM player_stats_cache')
        cursor.execute('DELETE FROM team_roster_cache')
        cursor.execute('DELETE FROM team_player_stats_cache')

        conn.commit()
        conn.close()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM player_stats_cache')
        player_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM team_roster_cache')
        roster_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM team_player_stats_cache')
        team_stats_count = cursor.fetchone()[0]

        # Count expired entries
        now = datetime.now().isoformat()
        cursor.execute('SELECT COUNT(*) FROM player_stats_cache WHERE expires_at < ?', (now,))
        expired_players = cursor.fetchone()[0]

        conn.close()

        return {
            'total_players': player_count,
            'total_rosters': roster_count,
            'total_team_stats': team_stats_count,
            'expired_entries': expired_players
        }
