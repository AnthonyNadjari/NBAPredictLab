"""
Real Bookmaker Odds Fetcher
Fetches live odds from The Odds API (free tier: 500 requests/month)
"""

import requests
import os
from typing import Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path

class RealOddsFetcher:
    """
    Fetch real bookmaker odds from The Odds API.

    Free tier: 500 requests/month
    Sign up: https://the-odds-api.com/
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with API key.

        Get your free API key at: https://the-odds-api.com/
        Set in .env file: ODDS_API_KEY=your_key_here

        Args:
            api_key: The Odds API key (optional, will read from .env)
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"
        self.cache_dir = Path("data/odds_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_todays_odds(self) -> List[Dict]:
        """
        Get all NBA game odds for today.

        Returns:
            List of games with odds from multiple bookmakers
        """
        if not self.api_key:
            print("⚠️ No Odds API key found. Set ODDS_API_KEY in .env file")
            print("Get free key at: https://the-odds-api.com/")
            return []

        url = f"{self.base_url}/sports/{self.sport}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us,eu',  # US and European bookmakers
            'markets': 'h2h',     # Head-to-head (moneyline)
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            # Check remaining requests
            remaining = response.headers.get('x-requests-remaining', 'Unknown')
            used = response.headers.get('x-requests-used', 'Unknown')
            print(f"📊 Odds API Usage: {used} used, {remaining} remaining")

            if response.status_code == 200:
                games = response.json()

                # Cache the response
                cache_file = self.cache_dir / f"odds_{datetime.now().strftime('%Y%m%d')}.json"
                with open(cache_file, 'w') as f:
                    json.dump(games, f, indent=2)

                return games
            elif response.status_code == 401:
                print("❌ Invalid API key. Check your ODDS_API_KEY in .env")
                return []
            elif response.status_code == 429:
                print("❌ Rate limit exceeded. Using cached odds if available.")
                return self._load_cached_odds()
            else:
                print(f"❌ Error fetching odds: {response.status_code}")
                return self._load_cached_odds()

        except Exception as e:
            print(f"❌ Error fetching odds: {e}")
            return self._load_cached_odds()

    def get_game_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Get odds for a specific game.

        Args:
            home_team: Home team name (e.g., "Los Angeles Lakers")
            away_team: Away team name (e.g., "Boston Celtics")

        Returns:
            {
                'home_team': str,
                'away_team': str,
                'commence_time': str,
                'bookmakers': {
                    'draftkings': {'home': 1.85, 'away': 2.10},
                    'fanduel': {'home': 1.83, 'away': 2.15},
                    ...
                },
                'best_home_odds': {'bookmaker': 'fanduel', 'odds': 1.85},
                'best_away_odds': {'bookmaker': 'draftkings', 'odds': 2.15},
                'avg_home_odds': 1.84,
                'avg_away_odds': 2.125,
                'market_home_prob': 0.543,  # Implied probability
                'market_away_prob': 0.471,  # (sum > 1 due to vig)
                'source': 'the-odds-api',
                'last_update': '2024-12-28T10:30:00Z'
            }
        """
        all_games = self.get_todays_odds()

        if not all_games:
            return None

        # Normalize team names for matching
        home_normalized = self._normalize_team_name(home_team)
        away_normalized = self._normalize_team_name(away_team)

        for game in all_games:
            game_home = self._normalize_team_name(game.get('home_team', ''))
            game_away = self._normalize_team_name(game.get('away_team', ''))

            if game_home == home_normalized and game_away == away_normalized:
                return self._parse_game_odds(game)

        # Try fuzzy matching
        for game in all_games:
            game_home = self._normalize_team_name(game.get('home_team', ''))
            game_away = self._normalize_team_name(game.get('away_team', ''))

            if (home_normalized in game_home or game_home in home_normalized) and \
               (away_normalized in game_away or game_away in away_normalized):
                return self._parse_game_odds(game)

        print(f"⚠️ No odds found for {away_team} @ {home_team}")
        return None

    def _parse_game_odds(self, game: Dict) -> Dict:
        """Parse raw game data into structured odds"""
        bookmakers_data = {}
        home_odds_list = []
        away_odds_list = []

        for bookmaker in game.get('bookmakers', []):
            bookie_name = bookmaker.get('key', '').lower()

            # Get h2h market
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = market.get('outcomes', [])

                    home_odds = None
                    away_odds = None

                    for outcome in outcomes:
                        if outcome.get('name') == game.get('home_team'):
                            home_odds = outcome.get('price')
                        elif outcome.get('name') == game.get('away_team'):
                            away_odds = outcome.get('price')

                    if home_odds and away_odds:
                        bookmakers_data[bookie_name] = {
                            'home': home_odds,
                            'away': away_odds
                        }
                        home_odds_list.append(home_odds)
                        away_odds_list.append(away_odds)

        if not bookmakers_data:
            return None

        # Calculate averages
        avg_home = sum(home_odds_list) / len(home_odds_list)
        avg_away = sum(away_odds_list) / len(away_odds_list)

        # Find best odds
        best_home = {'bookmaker': '', 'odds': 0}
        best_away = {'bookmaker': '', 'odds': 0}

        for bookie, odds in bookmakers_data.items():
            if odds['home'] > best_home['odds']:
                best_home = {'bookmaker': bookie, 'odds': odds['home']}
            if odds['away'] > best_away['odds']:
                best_away = {'bookmaker': bookie, 'odds': odds['away']}

        # Calculate implied probabilities (removing vig approximately)
        market_home_prob = 1 / avg_home
        market_away_prob = 1 / avg_away

        return {
            'home_team': game.get('home_team'),
            'away_team': game.get('away_team'),
            'commence_time': game.get('commence_time'),
            'bookmakers': bookmakers_data,
            'best_home_odds': best_home,
            'best_away_odds': best_away,
            'avg_home_odds': round(avg_home, 2),
            'avg_away_odds': round(avg_away, 2),
            'market_home_prob': round(market_home_prob, 3),
            'market_away_prob': round(market_away_prob, 3),
            'source': 'the-odds-api',
            'last_update': datetime.now().isoformat()
        }

    def _normalize_team_name(self, team: str) -> str:
        """Normalize team name for matching"""
        # Convert to lowercase and remove common words
        team = team.lower().strip()
        team = team.replace('los angeles', 'la')
        team = team.replace('new york', 'ny')
        team = team.replace('golden state', 'gs')

        # Common abbreviations
        abbrev_map = {
            'lakers': 'lal', 'celtics': 'bos', 'warriors': 'gsw',
            'nets': 'bkn', 'knicks': 'nyk', 'bulls': 'chi',
            'cavaliers': 'cle', 'mavericks': 'dal', 'nuggets': 'den',
            'pistons': 'det', 'rockets': 'hou', 'pacers': 'ind',
            'clippers': 'lac', 'heat': 'mia', 'bucks': 'mil',
            'timberwolves': 'min', 'pelicans': 'nop', 'thunder': 'okc',
            'magic': 'orl', '76ers': 'phi', 'suns': 'phx',
            'blazers': 'por', 'kings': 'sac', 'spurs': 'sas',
            'raptors': 'tor', 'jazz': 'uta', 'wizards': 'was',
            'hawks': 'atl', 'hornets': 'cha', 'grizzlies': 'mem',
            'sixers': 'phi', 'trail blazers': 'por'
        }

        for full, abbrev in abbrev_map.items():
            if full in team:
                return abbrev

        return team

    def _load_cached_odds(self) -> List[Dict]:
        """Load most recent cached odds"""
        cache_files = sorted(self.cache_dir.glob("odds_*.json"), reverse=True)

        if cache_files:
            try:
                with open(cache_files[0], 'r') as f:
                    data = json.load(f)
                print(f"📁 Using cached odds from {cache_files[0].name}")
                return data
            except Exception as e:
                print(f"Error loading cache: {e}")

        return []

    def check_quota(self) -> Dict:
        """Check remaining API quota"""
        if not self.api_key:
            return {'error': 'No API key'}

        url = f"{self.base_url}/sports"
        params = {'apiKey': self.api_key}

        try:
            response = requests.get(url, params=params, timeout=5)
            return {
                'remaining': response.headers.get('x-requests-remaining', 'Unknown'),
                'used': response.headers.get('x-requests-used', 'Unknown'),
                'status': 'OK' if response.status_code == 200 else 'Error'
            }
        except Exception as e:
            return {'error': str(e)}


# Alternative: Use odds-api.com free endpoints (no key needed, limited data)
class FreeOddsFetcher:
    """
    Fallback fetcher using free endpoints.
    No API key needed but very limited data.
    """

    def __init__(self):
        self.base_url = "https://www.thesportsdb.com/api/v1/json/3"

    def get_game_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Try to get odds from free source.
        Very limited and unreliable.
        """
        # This is a placeholder - free odds sources are scarce and unreliable
        print("⚠️ Free odds sources have very limited data")
        print("Recommend getting The Odds API key (free tier: 500 requests/month)")
        return None


if __name__ == "__main__":
    # Test the fetcher
    print("Testing Real Odds Fetcher...")
    print("=" * 50)

    fetcher = RealOddsFetcher()

    # Check quota
    quota = fetcher.check_quota()
    print(f"API Quota: {quota}")
    print()

    # Get all today's games
    print("Fetching today's NBA odds...")
    games = fetcher.get_todays_odds()

    if games:
        print(f"\n✅ Found {len(games)} games with odds:")
        for game in games[:3]:  # Show first 3
            print(f"\n{game.get('away_team')} @ {game.get('home_team')}")
            print(f"Start: {game.get('commence_time')}")

            odds = fetcher.get_game_odds(game.get('home_team'), game.get('away_team'))
            if odds:
                print(f"Avg odds: Home {odds['avg_home_odds']}, Away {odds['avg_away_odds']}")
                print(f"Market prob: Home {odds['market_home_prob']:.1%}, Away {odds['market_away_prob']:.1%}")
    else:
        print("\n❌ No odds data available")
        print("\nTo use this feature:")
        print("1. Get free API key: https://the-odds-api.com/")
        print("2. Add to .env file: ODDS_API_KEY=your_key_here")
        print("3. Free tier: 500 requests/month")
