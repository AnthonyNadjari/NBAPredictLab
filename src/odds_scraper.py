"""
NBA Betting Odds Generator
Generates realistic odds from Elo ratings and model probabilities.
Since real-time odds APIs require paid subscriptions, this module provides
accurate odds calculations based on the same math bookmakers use.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
from datetime import datetime
import time
import numpy as np


def generate_bookmaker_odds(home_prob: float, away_prob: float = None,
                           home_elo: float = 1500, away_elo: float = 1500) -> Dict:
    """
    Generate realistic bookmaker odds from model probability or Elo ratings.

    This uses the same mathematical principles as real bookmakers:
    1. Calculate fair probability
    2. Apply margin (overround) - typically 3-6% for NBA
    3. Round to standard odds increments

    Args:
        home_prob: Model's home win probability (0-1)
        away_prob: Model's away win probability (optional, calculated if not provided)
        home_elo: Home team Elo rating (used if home_prob not decisive)
        away_elo: Away team Elo rating

    Returns:
        Dict with bookmaker odds
    """
    if away_prob is None:
        away_prob = 1 - home_prob

    # Ensure probabilities are valid
    home_prob = max(0.05, min(0.95, home_prob))
    away_prob = max(0.05, min(0.95, away_prob))

    # Normalize
    total = home_prob + away_prob
    home_prob /= total
    away_prob /= total

    # Different bookmaker profiles with different margins
    bookmaker_profiles = {
        'Pinnacle': {'margin': 0.025, 'type': 'sharp'},      # Sharpest book
        'Bet365': {'margin': 0.045, 'type': 'recreational'},  # Most popular
        'DraftKings': {'margin': 0.05, 'type': 'us'},        # US market
        'FanDuel': {'margin': 0.048, 'type': 'us'},          # US market
        'BetMGM': {'margin': 0.052, 'type': 'us'},           # US market
    }

    bookmakers = {}

    for bookie_name, profile in bookmaker_profiles.items():
        margin = profile['margin']

        # Apply margin using the multiplicative method
        home_implied = home_prob * (1 + margin / 2)
        away_implied = away_prob * (1 + margin / 2)

        # Convert to decimal odds
        home_odds = round(1 / home_implied, 2)
        away_odds = round(1 / away_implied, 2)

        # Round to standard increments (books use specific increments)
        home_odds = _round_to_odds_increment(home_odds)
        away_odds = _round_to_odds_increment(away_odds)

        bookmakers[bookie_name] = {
            'home': home_odds,
            'away': away_odds,
            'margin': round(margin * 100, 1)
        }

    # Find best odds
    best_home = max(bookmakers.items(), key=lambda x: x[1]['home'])
    best_away = max(bookmakers.items(), key=lambda x: x[1]['away'])

    return {
        'bookmakers': bookmakers,
        'best_home': {'bookmaker': best_home[0], 'odds': best_home[1]['home']},
        'best_away': {'bookmaker': best_away[0], 'odds': best_away[1]['away']},
        'fair_home_odds': round(1 / home_prob, 2),
        'fair_away_odds': round(1 / away_prob, 2),
        'home_probability': round(home_prob * 100, 1),
        'away_probability': round(away_prob * 100, 1),
        'source': 'model_calculation'
    }


def _round_to_odds_increment(odds: float) -> float:
    """
    Round odds to standard bookmaker increments.
    Books don't offer odds like 1.873 - they use specific increments.
    """
    if odds < 1.5:
        # Very short odds: 0.01 increments
        return round(odds, 2)
    elif odds < 2.0:
        # Short odds: 0.02 increments
        return round(odds / 0.02) * 0.02
    elif odds < 3.0:
        # Medium odds: 0.05 increments
        return round(odds / 0.05) * 0.05
    elif odds < 5.0:
        # Longer odds: 0.1 increments
        return round(odds / 0.1) * 0.1
    else:
        # Long odds: 0.25 increments
        return round(odds / 0.25) * 0.25


def odds_to_american(decimal_odds: float) -> str:
    """Convert decimal odds to American format"""
    if decimal_odds >= 2.0:
        american = int((decimal_odds - 1) * 100)
        return f"+{american}"
    else:
        american = int(-100 / (decimal_odds - 1))
        return str(american)


def get_elo_based_odds(home_elo: float, away_elo: float,
                       home_court_advantage: float = 100) -> Dict:
    """
    Calculate odds purely from Elo ratings.

    Elo-based probabilities are what bookmakers primarily use
    as their starting point before making adjustments.

    Args:
        home_elo: Home team's Elo rating
        away_elo: Away team's Elo rating
        home_court_advantage: Elo points added for home court (default 100 = ~3.5 points)

    Returns:
        Dict with odds and probabilities
    """
    # Elo formula with home advantage
    elo_diff = (home_elo + home_court_advantage) - away_elo
    home_prob = 1 / (1 + 10 ** (-elo_diff / 400))

    return generate_bookmaker_odds(home_prob, 1 - home_prob, home_elo, away_elo)


class BettingOddsScraper:
    """Scrape real betting odds from popular betting websites"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_oddsportal_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Scrape odds from OddsPortal.com
        
        Returns:
            {
                'home_odds': float,
                'away_odds': float,
                'bookmakers': {
                    'pinnacle': {'home': 1.85, 'away': 2.10},
                    'bet365': {'home': 1.83, 'away': 2.15},
                    ...
                }
            }
        """
        try:
            # OddsPortal URL format: /basketball/usa/nba/
            # Search for the specific game
            search_url = "https://www.oddsportal.com/basketball/usa/nba/"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the game matching our teams
            # OddsPortal uses team abbreviations, so we need to normalize
            home_normalized = self._normalize_team_name(home_team)
            away_normalized = self._normalize_team_name(away_team)
            
            # Look for game links
            game_links = soup.find_all('a', href=re.compile(r'/basketball/usa/nba/'))
            
            for link in game_links:
                text = link.get_text().lower()
                if home_normalized in text and away_normalized in text:
                    # Found the game, get detailed odds
                    game_url = "https://www.oddsportal.com" + link['href']
                    return self._scrape_oddsportal_game(game_url)
            
            return None
            
        except Exception as e:
            print(f"Error scraping OddsPortal: {e}")
            return None
    
    def _scrape_oddsportal_game(self, game_url: str) -> Optional[Dict]:
        """Scrape detailed odds from a specific game page"""
        try:
            response = self.session.get(game_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract odds from the table
            # OddsPortal shows odds in a table format with bookmaker names
            odds_data = {
                'bookmakers': {},
                'home_odds': 0,
                'away_odds': 0
            }
            
            # Find odds table
            odds_table = soup.find('div', class_='table-container')
            if not odds_table:
                return None
            
            # Parse bookmaker odds
            rows = odds_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 3:
                    bookmaker = cells[0].get_text().strip().lower()
                    try:
                        home_odd = float(cells[1].get_text().strip())
                        away_odd = float(cells[2].get_text().strip())
                        
                        odds_data['bookmakers'][bookmaker] = {
                            'home': home_odd,
                            'away': away_odd
                        }
                    except (ValueError, IndexError):
                        continue
            
            # Calculate average odds
            if odds_data['bookmakers']:
                home_odds_list = [b['home'] for b in odds_data['bookmakers'].values()]
                away_odds_list = [b['away'] for b in odds_data['bookmakers'].values()]
                
                odds_data['home_odds'] = sum(home_odds_list) / len(home_odds_list)
                odds_data['away_odds'] = sum(away_odds_list) / len(away_odds_list)
            
            return odds_data if odds_data['bookmakers'] else None
            
        except Exception as e:
            print(f"Error scraping game page: {e}")
            return None
    
    def get_flashscore_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Scrape odds from FlashScore.com
        Similar structure to OddsPortal
        """
        try:
            url = "https://www.flashscore.com/basketball/usa/nba/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # FlashScore uses JavaScript heavily, so this might need Selenium
            # For now, return None and use OddsPortal as primary source
            return None
            
        except Exception as e:
            print(f"Error scraping FlashScore: {e}")
            return None
    
    def get_best_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Get the best odds from all available sources.
        
        Returns:
            {
                'home_team': str,
                'away_team': str,
                'home_odds': float,  # Average across bookmakers
                'away_odds': float,
                'bookmakers': {
                    'pinnacle': {'home': 1.85, 'away': 2.10},
                    'bet365': {'home': 1.83, 'away': 2.15},
                    ...
                },
                'best_home': {'bookmaker': 'bet365', 'odds': 1.85},
                'best_away': {'bookmaker': 'pinnacle', 'odds': 2.15},
                'source': 'oddsportal'
            }
        """
        # Try OddsPortal first (most reliable)
        odds = self.get_oddsportal_odds(home_team, away_team)
        
        if odds:
            # Find best odds for each outcome
            best_home = {'bookmaker': None, 'odds': 0}
            best_away = {'bookmaker': None, 'odds': 0}
            
            for bookmaker, values in odds['bookmakers'].items():
                if values['home'] > best_home['odds']:
                    best_home = {'bookmaker': bookmaker, 'odds': values['home']}
                if values['away'] > best_away['odds']:
                    best_away = {'bookmaker': bookmaker, 'odds': values['away']}
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_odds': odds['home_odds'],
                'away_odds': odds['away_odds'],
                'bookmakers': odds['bookmakers'],
                'best_home': best_home,
                'best_away': best_away,
                'source': 'oddsportal'
            }
        
        # If scraping fails, return None (will fall back to Elo simulation)
        return None
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names for matching"""
        # Remove common words and convert to lowercase
        normalized = team_name.lower()
        normalized = normalized.replace('los angeles', 'la')
        normalized = normalized.replace('new york', 'ny')
        normalized = normalized.replace('golden state', 'gs')
        
        # Extract key words (usually the last word is the team name)
        words = normalized.split()
        if len(words) > 1:
            return words[-1]  # e.g., "lakers", "celtics", "nuggets"
        return normalized


# Simple fallback using a free odds API (no key required for basic access)
class FreeOddsAPI:
    """Fallback using odds-api.com free tier (no key required for some endpoints)"""
    
    def __init__(self):
        self.base_url = "https://odds.p.rapidapi.com/v4"
        # Note: This is a placeholder - RapidAPI requires a key
        # But we can try the direct API without key for limited access
    
    def get_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """Try to get odds without API key (limited)"""
        # This won't work without a key, but kept for reference
        return None


if __name__ == "__main__":
    scraper = BettingOddsScraper()
    
    # Test
    odds = scraper.get_best_odds("Denver Nuggets", "Indiana Pacers")
    
    if odds:
        print(f"\n{odds['away_team']} @ {odds['home_team']}")
        print(f"Average Odds: Home {odds['home_odds']:.2f} | Away {odds['away_odds']:.2f}")
        print(f"Best: Home {odds['best_home']['odds']:.2f} ({odds['best_home']['bookmaker']})")
        print(f"      Away {odds['best_away']['odds']:.2f} ({odds['best_away']['bookmaker']})")
    else:
        print("Could not fetch odds")
