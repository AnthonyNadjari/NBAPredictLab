"""
Real-time odds fetcher using The Odds API (Free Tier).

Get your free API key at: https://the-odds-api.com/
Free tier: 500 requests/month (plenty for NBA predictions)

Setup:
1. Go to https://the-odds-api.com/
2. Sign up for free account
3. Copy your API key
4. Create a .env file in the project root with: ODDS_API_KEY=your_key_here
"""

import requests
from typing import Dict, List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OddsAPIClient:
    """Client for fetching real NBA odds from The Odds API (Free Tier)"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the odds API client.
        
        Args:
            api_key: Your API key from theoddsapi.com
                    If None, will try to load from ODDS_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"
        self.last_quota_check = None
        
    def get_upcoming_games_odds(self, regions: str = "us", markets: str = "h2h") -> List[Dict]:
        """
        Fetch odds for upcoming NBA games.
        
        Args:
            regions: Comma-separated regions (us, eu, uk, au)
            markets: Comma-separated markets (h2h=moneyline, spreads, totals)
            
        Returns:
            List of games with odds from multiple bookmakers
        """
        if not self.api_key:
            print("⚠️ No API key found. Get one free at https://the-odds-api.com/")
            return []
            
        url = f"{self.base_url}/sports/{self.sport}/odds"
        
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",  # European format (1.5, 2.0, etc.)
            "dateFormat": "iso"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Check remaining quota
            remaining = response.headers.get('x-requests-remaining', 'Unknown')
            used = response.headers.get('x-requests-used', 'Unknown')
            self.last_quota_check = {
                'remaining': remaining,
                'used': used,
                'timestamp': datetime.now()
            }
            print(f"📊 API Quota: {used} used, {remaining} remaining")
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching odds: {e}")
            return []
    
    def find_game_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Find odds for a specific matchup.
        
        Args:
            home_team: Home team name (e.g., "Los Angeles Lakers")
            away_team: Away team name (e.g., "Boston Celtics")
            
        Returns:
            Dict with odds from multiple bookmakers, or None if not found
        """
        games = self.get_upcoming_games_odds()
        
        if not games:
            print(f"⚠️ No upcoming games found from Odds API")
            return None
        
        print(f"🔍 Searching for: {away_team} @ {home_team}")
        print(f"📊 Found {len(games)} games from Odds API")
        
        # Try to find the game
        for game in games:
            api_home = game.get('home_team', '')
            api_away = game.get('away_team', '')
            
            # Use flexible matching
            home_match = self._teams_match(home_team, api_home)
            away_match = self._teams_match(away_team, api_away)
            
            if home_match and away_match:
                print(f"✅ MATCH FOUND: {api_away} @ {api_home}")
                return self._parse_game_odds(game)
            
            # Debug: Show close matches
            if home_match or away_match:
                print(f"⚠️ Partial match: {api_away} @ {api_home} (home={home_match}, away={away_match})")
        
        # If no match found, show available games for debugging
        print(f"❌ No match found. Available games:")
        for game in games[:5]:  # Show first 5
            print(f"   - {game.get('away_team')} @ {game.get('home_team')}")
                
        return None
    
    def _normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team names for matching.
        
        Examples:
            "Los Angeles Lakers" -> "lakers"
            "LA Lakers" -> "lakers"
            "Golden State Warriors" -> "warriors"
        """
        normalized = team_name.lower()
        
        # Remove city names (more comprehensive)
        cities_to_remove = [
            "los angeles", "new york", "golden state", "oklahoma city",
            "san antonio", "new orleans", "portland", "phoenix",
            "miami", "boston", "chicago", "cleveland", "dallas",
            "denver", "detroit", "houston", "indiana", "milwaukee",
            "minnesota", "memphis", "orlando", "philadelphia",
            "sacramento", "toronto", "utah", "washington", "atlanta",
            "brooklyn", "charlotte", "la ", "ny "
        ]
        
        for city in cities_to_remove:
            normalized = normalized.replace(city, "")
        
        # Get the team name (usually the last word)
        words = normalized.strip().split()
        if words:
            return words[-1]  # "lakers", "celtics", "nuggets", etc.
        return ""
    
    def _teams_match(self, team1: str, team2: str) -> bool:
        """Check if two team names match using multiple strategies"""
        # Strategy 1: Exact match
        if team1.lower() == team2.lower():
            return True
        
        # Strategy 2: Normalized match
        norm1 = self._normalize_team_name(team1)
        norm2 = self._normalize_team_name(team2)
        if norm1 and norm2 and norm1 == norm2:
            return True
        
        # Strategy 3: One contains the other
        if team1.lower() in team2.lower() or team2.lower() in team1.lower():
            return True
        
        return False
    
    def _parse_game_odds(self, game: Dict) -> Dict:
        """
        Parse raw API response into a clean format.
        
        Returns:
            {
                'home_team': str,
                'away_team': str,
                'commence_time': str,
                'bookmakers': {
                    'pinnacle': {'home': 1.85, 'away': 2.10},
                    'bet365': {'home': 1.83, 'away': 2.15},
                    ...
                },
                'best_home_odds': {'bookmaker': 'bet365', 'odds': 1.85},
                'best_away_odds': {'bookmaker': 'pinnacle', 'odds': 2.15}
            }
        """
        result = {
            'home_team': game.get('home_team'),
            'away_team': game.get('away_team'),
            'commence_time': game.get('commence_time'),
            'bookmakers': {}
        }
        
        best_home = {'bookmaker': None, 'odds': 0}
        best_away = {'bookmaker': None, 'odds': 0}
        
        for bookmaker in game.get('bookmakers', []):
            bookie_name = bookmaker.get('key')
            
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = market.get('outcomes', [])
                    
                    home_odds = None
                    away_odds = None
                    
                    for outcome in outcomes:
                        if outcome.get('name') == result['home_team']:
                            home_odds = outcome.get('price')
                        elif outcome.get('name') == result['away_team']:
                            away_odds = outcome.get('price')
                    
                    if home_odds and away_odds:
                        result['bookmakers'][bookie_name] = {
                            'home': home_odds,
                            'away': away_odds
                        }
                        
                        # Track best odds
                        if home_odds > best_home['odds']:
                            best_home = {'bookmaker': bookie_name, 'odds': home_odds}
                        if away_odds > best_away['odds']:
                            best_away = {'bookmaker': bookie_name, 'odds': away_odds}
        
        result['best_home_odds'] = best_home
        result['best_away_odds'] = best_away
        
        return result
    
    def get_quota_info(self) -> Dict:
        """Check remaining API quota"""
        if not self.api_key:
            return {'error': 'No API key provided'}
            
        url = f"{self.base_url}/sports/{self.sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us"
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            return {
                'remaining': response.headers.get('x-requests-remaining', 'Unknown'),
                'used': response.headers.get('x-requests-used', 'Unknown')
            }
        except:
            return {'error': 'Could not fetch quota'}


# Example usage
if __name__ == "__main__":
    # Get your API key from https://the-odds-api.com/
    API_KEY = "YOUR_API_KEY_HERE"
    
    client = OddsAPIClient(api_key=API_KEY)
    
    # Get all upcoming games
    games = client.get_upcoming_games_odds()
    print(f"Found {len(games)} upcoming games")
    
    # Find specific game
    odds = client.find_game_odds("Denver Nuggets", "Indiana Pacers")
    if odds:
        print(f"\n{odds['away_team']} @ {odds['home_team']}")
        print(f"Best odds: Home {odds['best_home_odds']['odds']} ({odds['best_home_odds']['bookmaker']})")
        print(f"           Away {odds['best_away_odds']['odds']} ({odds['best_away_odds']['bookmaker']})")
