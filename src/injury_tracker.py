"""
NBA Injury Tracker
Fetches and caches injury information for NBA teams
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

class InjuryTracker:
    """
    Tracks NBA player injuries using ESPN's injury report.
    Caches results to minimize API calls.
    """
    
    def __init__(self, db_path: str = "data/nba_predictor.db"):
        self.db_path = Path(db_path)
        self.cache_duration_hours = 6  # Refresh every 6 hours
        self._init_injury_table()
        
        # NBA team abbreviations mapping
        self.team_abbrev_map = {
            1610612738: 'BOS', 1610612751: 'BKN', 1610612752: 'NYK', 1610612755: 'PHI',
            1610612761: 'TOR', 1610612741: 'CHI', 1610612739: 'CLE', 1610612765: 'DET',
            1610612754: 'IND', 1610612749: 'MIL', 1610612737: 'ATL', 1610612766: 'CHA',
            1610612748: 'MIA', 1610612753: 'ORL', 1610612764: 'WAS', 1610612743: 'DEN',
            1610612750: 'MIN', 1610612760: 'OKC', 1610612757: 'POR', 1610612762: 'UTA',
            1610612744: 'GSW', 1610612746: 'LAC', 1610612747: 'LAL', 1610612756: 'PHX',
            1610612758: 'SAC', 1610612742: 'DAL', 1610612745: 'HOU', 1610612763: 'MEM',
            1610612740: 'SAS', 1610612759: 'NOP',
        }
        
        # All-Star players (manually curated - update seasonally)
        self.all_stars = {
            'Giannis Antetokounmpo', 'Luka Doncic', 'Joel Embiid', 'Nikola Jokic',
            'Stephen Curry', 'Kevin Durant', 'LeBron James', 'Jayson Tatum',
            'Damian Lillard', 'Anthony Davis', 'Kawhi Leonard', 'Jimmy Butler',
            'Donovan Mitchell', 'Ja Morant', 'Trae Young', 'Devin Booker',
            'Jaylen Brown', 'Anthony Edwards', 'Tyrese Haliburton', 'Shai Gilgeous-Alexander',
            'Paolo Banchero', 'Lauri Markkanen', 'De\'Aaron Fox', 'Domantas Sabonis',
        }
    
    def _init_injury_table(self):
        """Initialize injury tracking table in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injuries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER,
                team_abbrev TEXT,
                player_name TEXT,
                position TEXT,
                status TEXT,
                injury_type TEXT,
                is_starter BOOLEAN,
                is_star BOOLEAN,
                fetched_at TIMESTAMP,
                UNIQUE(team_id, player_name, fetched_at)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_team_injuries(self, team_id: int, force_refresh: bool = False) -> Dict:
        """
        Get injury information for a team.
        
        Args:
            team_id: NBA team ID
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Dictionary with injury stats:
            - injured_starters: Number of starters out
            - star_injured: Boolean, True if All-Star is out
            - total_injured: Total number of injured players
            - injuries: List of injury details
        """
        # Check cache first
        if not force_refresh:
            cached = self._get_cached_injuries(team_id)
            if cached is not None:
                return cached
        
        # Fetch fresh data
        team_abbrev = self.team_abbrev_map.get(team_id)
        if not team_abbrev:
            return self._empty_injury_dict()
        
        try:
            injuries = self._scrape_espn_injuries(team_abbrev)
            
            # Save to cache
            self._cache_injuries(team_id, team_abbrev, injuries)
            
            # Calculate stats
            return self._calculate_injury_stats(injuries)
            
        except Exception as e:
            print(f"Error fetching injuries for team {team_id}: {e}")
            return self._empty_injury_dict()
    
    def _get_cached_injuries(self, team_id: int) -> Optional[Dict]:
        """Get injuries from cache if recent enough."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=self.cache_duration_hours)
        
        cursor.execute("""
            SELECT player_name, position, status, injury_type, is_starter, is_star
            FROM injuries
            WHERE team_id = ? AND fetched_at > ?
        """, (team_id, cutoff_time.isoformat()))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        injuries = [
            {
                'player': row[0],
                'position': row[1],
                'status': row[2],
                'injury': row[3],
                'is_starter': bool(row[4]),
                'is_star': bool(row[5])
            }
            for row in rows
        ]
        
        return self._calculate_injury_stats(injuries)
    
    def _scrape_espn_injuries(self, team_abbrev: str) -> List[Dict]:
        """
        Scrape injury data from CBS Sports (more reliable than ESPN JS-rendered pages).
        Falls back to Rotowire lineups if CBS fails.
        """
        injuries = []

        # Try CBS Sports first (server-rendered HTML)
        try:
            injuries = self._scrape_cbs_injuries(team_abbrev)
            if injuries:
                return injuries
        except Exception as e:
            print(f"CBS scrape failed for {team_abbrev}: {e}")

        # Fallback to Rotowire lineups page
        try:
            injuries = self._scrape_rotowire_injuries(team_abbrev)
        except Exception as e:
            print(f"Rotowire scrape failed for {team_abbrev}: {e}")

        return injuries

    def _scrape_cbs_injuries(self, team_abbrev: str) -> List[Dict]:
        """Scrape injuries from Hashtag Basketball (reliable, regularly updated)."""
        url = "https://hashtagbasketball.com/nba-injury-report"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        injuries = []

        # Team name mapping for matching
        team_name_map = {
            'ATL': 'Atlanta', 'BOS': 'Boston', 'BKN': 'Brooklyn', 'CHA': 'Charlotte',
            'CHI': 'Chicago', 'CLE': 'Cleveland', 'DAL': 'Dallas', 'DEN': 'Denver',
            'DET': 'Detroit', 'GSW': 'Golden State', 'HOU': 'Houston', 'IND': 'Indiana',
            'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Memphis', 'MIA': 'Miami',
            'MIL': 'Milwaukee', 'MIN': 'Minnesota', 'NOP': 'New Orleans', 'NYK': 'Knicks',
            'OKC': 'Oklahoma', 'ORL': 'Orlando', 'PHI': 'Philadelphia', 'PHX': 'Phoenix',
            'POR': 'Portland', 'SAC': 'Sacramento', 'SAS': 'San Antonio', 'TOR': 'Toronto',
            'UTA': 'Utah', 'WAS': 'Washington'
        }
        team_search = team_name_map.get(team_abbrev, team_abbrev)

        # Find injury table rows
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    # Check if this row is for our team
                    row_text = row.get_text().lower()
                    if team_search.lower() in row_text:
                        player_name = cells[0].get_text().strip()
                        team = cells[1].get_text().strip() if len(cells) > 1 else ''
                        injury_type = cells[2].get_text().strip() if len(cells) > 2 else ''
                        status = cells[3].get_text().strip() if len(cells) > 3 else 'Out'

                        is_star = any(star.lower() in player_name.lower() for star in self.all_stars)

                        injuries.append({
                            'player': player_name,
                            'position': '',
                            'injury': injury_type,
                            'status': status,
                            'is_starter': False,
                            'is_star': is_star
                        })

        return injuries

    def _scrape_rotowire_injuries(self, team_abbrev: str) -> List[Dict]:
        """Scrape injuries from Rotowire NBA lineups page."""
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        injuries = []

        # Map team abbreviations to Rotowire URL patterns
        team_url_map = {
            'ATL': 'hawks', 'BOS': 'celtics', 'BKN': 'nets', 'CHA': 'hornets',
            'CHI': 'bulls', 'CLE': 'cavaliers', 'DAL': 'mavericks', 'DEN': 'nuggets',
            'DET': 'pistons', 'GSW': 'warriors', 'HOU': 'rockets', 'IND': 'pacers',
            'LAC': 'clippers', 'LAL': 'lakers', 'MEM': 'grizzlies', 'MIA': 'heat',
            'MIL': 'bucks', 'MIN': 'timberwolves', 'NOP': 'pelicans', 'NYK': 'knicks',
            'OKC': 'thunder', 'ORL': 'magic', 'PHI': '76ers', 'PHX': 'suns',
            'POR': 'blazers', 'SAC': 'kings', 'SAS': 'spurs', 'TOR': 'raptors',
            'UTA': 'jazz', 'WAS': 'wizards'
        }

        team_pattern = team_url_map.get(team_abbrev, team_abbrev.lower())

        # Find the team's lineup card
        team_cards = soup.find_all('div', class_='lineup__box')

        for card in team_cards:
            # Check if this is the right team
            team_header = card.find('a', class_='lineup__team')
            if not team_header:
                continue

            team_link = team_header.get('href', '').lower()
            # Match team pattern in the link
            if team_pattern not in team_link and team_abbrev.lower() not in team_link:
                continue

            # Find all players and check for injury indicators
            all_players = card.find_all('li', class_='lineup__player')

            for player_li in all_players:
                # Check for any injury-related classes or elements
                player_classes = ' '.join(player_li.get('class', []))
                injury_span = player_li.find('span', class_='lineup__inj')

                # Also look for injury status in title attributes
                title = player_li.get('title', '')

                player_link = player_li.find('a')
                if not player_link:
                    continue

                player_name = player_link.get_text().strip()

                # Check if player has injury indicator
                if injury_span or 'injured' in player_classes.lower() or 'out' in title.lower():
                    injury_text = ''
                    status = 'Day-To-Day'

                    if injury_span:
                        injury_text = injury_span.get('title', '') or injury_span.get_text().strip()
                        injury_classes = ' '.join(injury_span.get('class', []))

                        # Determine status
                        if 'is-out' in injury_classes or injury_text.upper().startswith('O'):
                            status = 'Out'
                        elif 'is-gtd' in injury_classes or injury_text.upper().startswith('GTD'):
                            status = 'Game-Time-Decision'
                        elif 'is-questionable' in injury_classes or injury_text.upper().startswith('Q'):
                            status = 'Questionable'
                        elif 'is-doubtful' in injury_classes or injury_text.upper().startswith('D'):
                            status = 'Doubtful'

                    is_star = any(star.lower() in player_name.lower() for star in self.all_stars)

                    injuries.append({
                        'player': player_name,
                        'position': '',
                        'injury': injury_text,
                        'status': status,
                        'is_starter': True,
                        'is_star': is_star
                    })

            # We found the team, no need to check other cards
            break

        return injuries
    
    def _cache_injuries(self, team_id: int, team_abbrev: str, injuries: List[Dict]):
        """Save injuries to database cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        for injury in injuries:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO injuries 
                    (team_id, team_abbrev, player_name, position, status, injury_type, 
                     is_starter, is_star, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    team_id, team_abbrev, injury['player'], injury['position'],
                    injury['status'], injury['injury'], injury['is_starter'],
                    injury['is_star'], now
                ))
            except Exception as e:
                print(f"Error caching injury: {e}")
                continue
        
        conn.commit()
        conn.close()
    
    def _calculate_injury_stats(self, injuries: List[Dict]) -> Dict:
        """Calculate injury statistics from injury list."""
        if not injuries:
            return self._empty_injury_dict()
        
        # Filter to only "Out" players (not day-to-day)
        out_injuries = [
            inj for inj in injuries 
            if 'out' in inj['status'].lower() or 'dnp' in inj['status'].lower()
        ]
        
        injured_starters = sum(1 for inj in out_injuries if inj['is_starter'])
        star_injured = any(inj['is_star'] for inj in out_injuries)
        
        return {
            'injured_starters': injured_starters,
            'star_injured': int(star_injured),  # Convert to 0/1
            'total_injured': len(out_injuries),
            'injuries': out_injuries
        }
    
    def _empty_injury_dict(self) -> Dict:
        """Return empty injury dictionary."""
        return {
            'injured_starters': 0,
            'star_injured': 0,
            'total_injured': 0,
            'injuries': []
        }
    
    def get_injury_features(self, home_team_id: int, away_team_id: int) -> Dict[str, float]:
        """
        Get injury features for both teams.
        
        Returns:
            Dictionary with 4 features:
            - home_injured_starters
            - away_injured_starters
            - home_star_injured
            - away_star_injured
        """
        home_injuries = self.get_team_injuries(home_team_id)
        away_injuries = self.get_team_injuries(away_team_id)
        
        return {
            'home_injured_starters': home_injuries['injured_starters'],
            'away_injured_starters': away_injuries['injured_starters'],
            'home_star_injured': home_injuries['star_injured'],
            'away_star_injured': away_injuries['star_injured'],
        }


# Example usage
if __name__ == "__main__":
    tracker = InjuryTracker()
    
    # Example: Lakers
    lakers_id = 1610612747
    injuries = tracker.get_team_injuries(lakers_id, force_refresh=True)
    
    print(f"Lakers Injury Report:")
    print(f"  Injured Starters: {injuries['injured_starters']}")
    print(f"  Star Injured: {'Yes' if injuries['star_injured'] else 'No'}")
    print(f"  Total Injured: {injuries['total_injured']}")
    
    if injuries['injuries']:
        print("\n  Details:")
        for inj in injuries['injuries']:
            print(f"    - {inj['player']} ({inj['position']}): {inj['injury']} - {inj['status']}")
