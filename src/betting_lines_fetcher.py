"""
Betting Lines Fetcher
Fetches historical and live betting lines to use as features
"""

from typing import Dict, Optional
from datetime import datetime
import sqlite3
from pathlib import Path
from src.odds_api_client import OddsAPIClient


class BettingLinesFetcher:
    """
    Fetches betting lines (spreads, totals, moneylines) for use as features.
    Uses The Odds API for live data.
    """
    
    def __init__(self, db_path: str = "data/nba_predictor.db"):
        self.db_path = Path(db_path)
        self.odds_client = OddsAPIClient()
        self._init_betting_lines_table()
    
    def _init_betting_lines_table(self):
        """Initialize betting lines cache table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS betting_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date DATE,
                home_team TEXT,
                away_team TEXT,
                market_spread REAL,
                market_total REAL,
                market_home_ml REAL,
                market_away_ml REAL,
                market_implied_prob REAL,
                bookmaker TEXT,
                fetched_at TIMESTAMP,
                UNIQUE(game_date, home_team, away_team)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_betting_lines(self, home_team: str, away_team: str, 
                          game_date: str = None) -> Dict[str, float]:
        """
        Get betting lines for a game.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date (YYYY-MM-DD), if None uses today
            
        Returns:
            Dictionary with betting line features:
            - market_spread: Point spread (negative = home favored)
            - market_total: Over/Under total points
            - market_home_ml: Home moneyline (decimal odds)
            - market_away_ml: Away moneyline (decimal odds)
            - market_implied_prob: Implied probability from moneyline
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache first
        cached = self._get_cached_lines(home_team, away_team, game_date)
        if cached:
            return cached
        
        # Try to fetch live odds
        try:
            odds_data = self.odds_client.find_game_odds(home_team, away_team)
            
            if odds_data and odds_data.get('bookmakers'):
                lines = self._parse_odds_to_lines(odds_data)
                
                # Cache it
                self._cache_betting_lines(home_team, away_team, game_date, lines)
                
                return lines
        except Exception as e:
            print(f"Error fetching betting lines: {e}")
        
        # Return defaults if not available
        return self._default_lines()
    
    def _get_cached_lines(self, home_team: str, away_team: str, 
                          game_date: str) -> Optional[Dict]:
        """Get betting lines from cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT market_spread, market_total, market_home_ml, 
                   market_away_ml, market_implied_prob
            FROM betting_lines
            WHERE home_team = ? AND away_team = ? AND game_date = ?
        """, (home_team, away_team, game_date))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            'market_spread': row[0],
            'market_total': row[1],
            'market_home_ml': row[2],
            'market_away_ml': row[3],
            'market_implied_prob': row[4]
        }
    
    def _parse_odds_to_lines(self, odds_data: Dict) -> Dict[str, float]:
        """
        Parse odds API data into betting line features.
        
        Note: The Odds API primarily provides moneylines (h2h).
        Spreads and totals require additional markets.
        """
        # Get best home/away odds (from nested dict structure)
        home_ml = odds_data.get('best_home_odds', {}).get('odds', 2.0)
        away_ml = odds_data.get('best_away_odds', {}).get('odds', 2.0)
        
        # Calculate implied probability from moneyline
        home_implied = 1 / home_ml if home_ml > 0 else 0.5
        away_implied = 1 / away_ml if away_ml > 0 else 0.5
        
        # Normalize (remove vig)
        total_implied = home_implied + away_implied
        home_prob = home_implied / total_implied if total_implied > 0 else 0.5
        
        # Estimate spread from odds (rough approximation)
        # More negative = bigger favorite
        if home_ml < away_ml:
            # Home is favorite
            spread = -((away_ml - home_ml) * 2)  # Rough conversion
        else:
            # Away is favorite
            spread = ((home_ml - away_ml) * 2)
        
        # Estimate total (NBA average is around 220)
        # This is a placeholder - would need actual totals market
        estimated_total = 220.0
        
        return {
            'market_spread': round(spread, 1),
            'market_total': estimated_total,
            'market_home_ml': home_ml,
            'market_away_ml': away_ml,
            'market_implied_prob': round(home_prob, 3)
        }
    
    def _cache_betting_lines(self, home_team: str, away_team: str,
                            game_date: str, lines: Dict):
        """Save betting lines to cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO betting_lines
            (game_date, home_team, away_team, market_spread, market_total,
             market_home_ml, market_away_ml, market_implied_prob, 
             bookmaker, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_date, home_team, away_team,
            lines['market_spread'], lines['market_total'],
            lines['market_home_ml'], lines['market_away_ml'],
            lines['market_implied_prob'],
            'aggregated', datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _default_lines(self) -> Dict[str, float]:
        """Return default betting lines when not available."""
        return {
            'market_spread': 0.0,  # Pick'em
            'market_total': 220.0,  # NBA average
            'market_home_ml': 2.0,  # Even odds
            'market_away_ml': 2.0,
            'market_implied_prob': 0.5  # 50/50
        }
    
    def get_betting_features(self, home_team: str, away_team: str,
                            game_date: str = None) -> Dict[str, float]:
        """
        Get betting line features for a game.
        
        Returns 5 features ready for model input.
        """
        lines = self.get_betting_lines(home_team, away_team, game_date)
        
        return {
            'market_spread': lines['market_spread'],
            'market_total': lines['market_total'],
            'market_home_ml': lines['market_home_ml'],
            'market_implied_prob': lines['market_implied_prob'],
            'market_confidence': abs(lines['market_home_ml'] - lines['market_away_ml'])
        }


# Example usage
if __name__ == "__main__":
    fetcher = BettingLinesFetcher()
    
    # Example
    lines = fetcher.get_betting_lines("Los Angeles Lakers", "Boston Celtics")
    
    print("Lakers @ Celtics Betting Lines:")
    print(f"  Spread: {lines['market_spread']}")
    print(f"  Total: {lines['market_total']}")
    print(f"  Home ML: {lines['market_home_ml']}")
    print(f"  Away ML: {lines['market_away_ml']}")
    print(f"  Implied Prob: {lines['market_implied_prob']:.1%}")
