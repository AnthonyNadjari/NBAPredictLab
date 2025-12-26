"""
Public Sentiment Analyzer for NBA Games
Uses Reddit's public JSON API (no authentication needed) + NLP to analyze real public opinion.
Falls back to betting odds if Reddit data is insufficient.
"""

import re
import time
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Try to import VADER for sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available. Install with: pip install vaderSentiment")

from nba_api.stats.static import teams


class SentimentAnalyzer:
    """
    Analyzes public sentiment for NBA games from Reddit.
    Uses VADER sentiment analysis with betting-specific keyword extraction.
    """
    
    def __init__(self, db_path: str = 'data/nba_predictor.db', cache_hours: int = 6):
        """
        Initialize sentiment analyzer.
        
        Args:
            db_path: Path to SQLite database for caching
            cache_hours: Hours to cache sentiment results
        """
        self.db_path = db_path
        self.cache_hours = cache_hours
        self._init_cache_db()
        
        # Initialize VADER sentiment analyzer
        if VADER_AVAILABLE:
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            self.analyzer = None
            logger.warning("VADER sentiment analyzer not available")
        
        # Reddit public JSON API - no authentication needed!
        self.reddit = True  # Flag that Reddit is available
        self._init_reddit_client()
        
        # Team name mappings
        self._init_team_mappings()
        
        # Betting-specific keywords that indicate picks (expanded)
        self.pick_keywords = [
            'taking', 'picking', 'betting on', 'bet on', 'hammer', 'hammering',
            'leaning', 'leaning towards', 'going with', 'rolling with',
            'like', 'love', 'fade', 'fading', 'avoid', 'avoiding',
            'lock', 'locks', 'play', 'plays', 'parlay', 'parlays',
            'confident', 'confidently', 'sure', 'definitely', 'absolutely',
            'backing', 'back', 'on', 'win', 'wins', 'winning', 'lose', 'losing',
            'cover', 'covers', 'over', 'under', 'pick', 'picks', 'bet', 'bets',
            'moneyline', 'ml', 'spread', 'total', 'o/u', 'over/under',
            'take', 'takes', 'vs', 'versus', 'beat', 'beats', 'crush', 'destroy',
            'smash', 'blowout', 'easy', 'lock', 'guarantee', 'guaranteed'
        ]
        
        # Negation words (expanded)
        self.negation_words = [
            'not', 'no', 'never', 'avoid', 'fade', 'fading', 'skip', 'skipping',
            'pass', 'passing', 'stay away', 'stay away from', "don't", "won't",
            "can't", "shouldn't", "wouldn't", "isn't", "aren't"
        ]
        
        # Additional subreddits to check
        self.subreddits = [
            'sportsbook',      # Primary betting subreddit
            'nba',            # Main NBA subreddit
            'fantasybball',   # Fantasy basketball (often has picks)
            'dfsports',       # Daily fantasy sports
            'sportsbetting',  # Alternative betting subreddit
        ]
    
    def _init_cache_db(self):
        """Initialize cache database table."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    home_team TEXT,
                    away_team TEXT,
                    game_date TEXT,
                    home_pct REAL,
                    away_pct REAL,
                    total_mentions INTEGER,
                    confidence TEXT,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(home_team, away_team, game_date)
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing cache DB: {e}")
    
    def _init_reddit_client(self):
        """
        No longer needed! We use Reddit's public JSON API which requires no authentication.
        This method is kept for backwards compatibility but does nothing.
        """
        # Reddit public JSON API doesn't need authentication
        # We'll use requests directly to fetch JSON endpoints
        self.reddit = True  # Just a flag that we can use Reddit
        logger.info("Using Reddit public JSON API (no authentication required)")
    
    def _init_team_mappings(self):
        """Initialize team name mappings for flexible matching."""
        all_teams = teams.get_teams()
        
        # Full name to abbreviation
        self.team_full_to_abbrev = {t['full_name']: t['abbreviation'] for t in all_teams}
        
        # Abbreviation to full name
        self.team_abbrev_to_full = {t['abbreviation']: t['full_name'] for t in all_teams}
        
        # Common nicknames/city names (expanded)
        self.team_nicknames = {
            'lakers': 'LAL', 'laker': 'LAL', 'la lakers': 'LAL',
            'celtics': 'BOS', 'celtic': 'BOS', 'boston': 'BOS',
            'warriors': 'GSW', 'warrior': 'GSW', 'dubs': 'GSW', 'golden state': 'GSW',
            'heat': 'MIA', 'miami': 'MIA',
            'bulls': 'CHI', 'bull': 'CHI', 'chicago': 'CHI',
            'knicks': 'NYK', 'knick': 'NYK', 'new york': 'NYK',
            'nets': 'BKN', 'net': 'BKN', 'brooklyn': 'BKN',
            'sixers': 'PHI', 'sixer': 'PHI', '76ers': 'PHI', 'philly': 'PHI', 'philadelphia': 'PHI',
            'bucks': 'MIL', 'buck': 'MIL', 'milwaukee': 'MIL',
            'suns': 'PHX', 'sun': 'PHX', 'phoenix': 'PHX',
            'mavs': 'DAL', 'mav': 'DAL', 'mavericks': 'DAL', 'dallas': 'DAL',
            'nuggets': 'DEN', 'nugget': 'DEN', 'denver': 'DEN',
            'clippers': 'LAC', 'clipper': 'LAC', 'clips': 'LAC', 'la clippers': 'LAC',
            'jazz': 'UTA', 'utah': 'UTA',
            'blazers': 'POR', 'blazer': 'POR', 'trail blazers': 'POR', 'portland': 'POR',
            'grizzlies': 'MEM', 'grizzly': 'MEM', 'grizz': 'MEM', 'memphis': 'MEM',
            'spurs': 'SAS', 'spur': 'SAS', 'san antonio': 'SAS',
            'kings': 'SAC', 'king': 'SAC', 'sacramento': 'SAC',
            'pelicans': 'NOP', 'pelican': 'NOP', 'pels': 'NOP', 'new orleans': 'NOP',
            'thunder': 'OKC', 'okc': 'OKC', 'oklahoma city': 'OKC',
            'rockets': 'HOU', 'rocket': 'HOU', 'houston': 'HOU',
            'wolves': 'MIN', 'wolf': 'MIN', 'timberwolves': 'MIN', 'minnesota': 'MIN',
            'pistons': 'DET', 'piston': 'DET', 'detroit': 'DET',
            'cavs': 'CLE', 'cav': 'CLE', 'cavaliers': 'CLE', 'cleveland': 'CLE',
            'pacers': 'IND', 'pacer': 'IND', 'indiana': 'IND',
            'hawks': 'ATL', 'hawk': 'ATL', 'atlanta': 'ATL',
            'hornets': 'CHA', 'hornet': 'CHA', 'charlotte': 'CHA',
            'wizards': 'WAS', 'wizard': 'WAS', 'washington': 'WAS',
            'magic': 'ORL', 'orlando': 'ORL',
            'raptors': 'TOR', 'raptor': 'TOR', 'toronto': 'TOR'
        }
        
        # Create search patterns for each team
        self.team_patterns = {}
        for team in all_teams:
            abbrev = team['abbreviation']
            full_name = team['full_name']
            city = full_name.split()[-1]  # Last word is usually the team name
            
            # Create regex patterns
            patterns = [
                rf'\b{re.escape(abbrev)}\b',
                rf'\b{re.escape(full_name)}\b',
                rf'\b{re.escape(city)}\b',
            ]
            
            # Add nickname if exists
            for nick, abbr in self.team_nicknames.items():
                if abbr == abbrev:
                    patterns.append(rf'\b{re.escape(nick)}\b')
            
            self.team_patterns[abbrev] = re.compile('|'.join(patterns), re.IGNORECASE)
    
    def _get_cached_sentiment(self, home_team: str, away_team: str, game_date: str) -> Optional[Dict]:
        """Check cache for existing sentiment data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if cached result exists and is still valid
            cursor.execute("""
                SELECT home_pct, away_pct, total_mentions, confidence, cached_at
                FROM sentiment_cache
                WHERE home_team = ? AND away_team = ? AND game_date = ?
                AND datetime(cached_at) > datetime('now', '-' || ? || ' hours')
            """, (home_team, away_team, game_date, self.cache_hours))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'home_pct': result[0],
                    'away_pct': result[1],
                    'total_mentions': result[2],
                    'confidence': result[3],
                    'cached': True
                }
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
        
        return None
    
    def _cache_sentiment(self, home_team: str, away_team: str, game_date: str, 
                        home_pct: float, away_pct: float, total_mentions: int, confidence: str):
        """Cache sentiment result."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sentiment_cache
                (home_team, away_team, game_date, home_pct, away_pct, total_mentions, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (home_team, away_team, game_date, home_pct, away_pct, total_mentions, confidence))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error caching sentiment: {e}")
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team name to abbreviation."""
        # Try direct match
        if team_name in self.team_abbrev_to_full:
            return team_name
        
        # Try full name
        if team_name in self.team_full_to_abbrev:
            return self.team_full_to_abbrev[team_name]
        
        # Try nickname
        team_lower = team_name.lower()
        if team_lower in self.team_nicknames:
            return self.team_nicknames[team_lower]
        
        # Try partial match in full names
        for full_name, abbrev in self.team_full_to_abbrev.items():
            if team_name.lower() in full_name.lower() or full_name.lower() in team_name.lower():
                return abbrev
        
        return team_name  # Return as-is if no match
    
    def _fetch_reddit_json(self, subreddit_name: str, sort: str = 'hot', limit: int = 100) -> List[Dict]:
        """
        Fetch Reddit posts using public JSON API (no authentication needed!).
        
        Args:
            subreddit_name: Subreddit name (e.g., 'sportsbook', 'nba')
            sort: Sort method ('hot', 'new', 'top', 'rising')
            limit: Number of posts to fetch (max 100 per request)
        
        Returns:
            List of post dictionaries
        """
        posts = []
        try:
            # Reddit's public JSON endpoint
            url = f"https://www.reddit.com/r/{subreddit_name}/{sort}.json"
            params = {
                'limit': min(limit, 100),
                'raw_json': 1
            }
            
            headers = {
                'User-Agent': 'NBA_Predictor_Sentiment/1.0 (by /u/nbapredictor)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract posts from JSON structure
            if 'data' in data and 'children' in data['data']:
                for child in data['data']['children']:
                    if 'data' in child:
                        post = child['data']
                        posts.append({
                            'title': post.get('title', ''),
                            'selftext': post.get('selftext', ''),
                            'score': post.get('score', 0),
                            'created_utc': post.get('created_utc', 0),
                            'num_comments': post.get('num_comments', 0),
                            'permalink': post.get('permalink', ''),
                            'id': post.get('id', '')
                        })
            
            logger.info(f"Fetched {len(posts)} posts from r/{subreddit_name}")
            time.sleep(1)  # Be nice to Reddit's servers
            
        except Exception as e:
            logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
        
        return posts
    
    def _fetch_reddit_comments(self, subreddit_name: str, team_names: List[str], 
                               time_window_hours: int = 72, limit: int = 200) -> List[Dict]:
        """
        Fetch comments from Reddit using public JSON API (no auth needed!).
        
        Args:
            subreddit_name: Name of subreddit (e.g., 'sportsbook', 'nba')
            team_names: List of team names/abbreviations to search for
            time_window_hours: Hours to look back (default 72 for more data)
            limit: Maximum number of posts to fetch
        
        Returns:
            List of comment dictionaries with 'body', 'created_utc', 'score'
        """
        if not self.reddit:
            return []
        
        comments = []
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        try:
            # Strategy 1: Fetch hot posts
            hot_posts = self._fetch_reddit_json(subreddit_name, sort='hot', limit=100)
            for post in hot_posts:
                if post['created_utc'] >= cutoff_time:
                    # Add post title and text as comment
                    text = f"{post['title']} {post['selftext']}".strip()
                    if text:
                        comments.append({
                            'body': text,
                            'created_utc': post['created_utc'],
                            'score': post['score']
                        })
                    # Fetch comments from this post
                    self._fetch_post_comments(post['permalink'], comments, cutoff_time)
            
            # Strategy 2: Fetch new posts
            new_posts = self._fetch_reddit_json(subreddit_name, sort='new', limit=100)
            for post in new_posts:
                if post['created_utc'] >= cutoff_time:
                    text = f"{post['title']} {post['selftext']}".strip()
                    if text:
                        comments.append({
                            'body': text,
                            'created_utc': post['created_utc'],
                            'score': post['score']
                        })
                    self._fetch_post_comments(post['permalink'], comments, cutoff_time)
            
            # Strategy 3: Search for team-specific posts (using Reddit search JSON)
            for team in team_names[:2]:  # Limit searches
                try:
                    search_url = f"https://www.reddit.com/r/{subreddit_name}/search.json"
                    params = {
                        'q': team,
                        'sort': 'new',
                        'limit': 25,
                        't': 'week',
                        'raw_json': 1
                    }
                    headers = {'User-Agent': 'NBA_Predictor_Sentiment/1.0'}
                    response = requests.get(search_url, params=params, headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data and 'children' in data['data']:
                            for child in data['data']['children']:
                                if 'data' in child:
                                    post = child['data']
                                    if post.get('created_utc', 0) >= cutoff_time:
                                        text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
                                        if text:
                                            comments.append({
                                                'body': text,
                                                'created_utc': post.get('created_utc', 0),
                                                'score': post.get('score', 0)
                                            })
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Error searching for {team}: {e}")
        
        except Exception as e:
            logger.error(f"Error fetching Reddit data from r/{subreddit_name}: {e}")
        
        return comments
    
    def _fetch_post_comments(self, permalink: str, comments_list: List, cutoff_time: float):
        """Fetch comments from a specific Reddit post using JSON API."""
        try:
            # Reddit comment JSON endpoint
            url = f"https://www.reddit.com{permalink}.json"
            headers = {'User-Agent': 'NBA_Predictor_Sentiment/1.0'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Reddit returns array: [post_data, comment_data]
                if len(data) > 1 and 'data' in data[1]:
                    self._extract_comments_from_json(data[1]['data'], comments_list, cutoff_time)
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            logger.debug(f"Error fetching comments from {permalink}: {e}")
    
    def _extract_comments_from_json(self, comment_data: Dict, comments_list: List, cutoff_time: float, depth: int = 0):
        """Recursively extract comments from Reddit JSON structure."""
        if depth > 5:  # Limit recursion
            return
        
        if 'children' in comment_data:
            for child in comment_data['children']:
                if 'data' in child:
                    comment = child['data']
                    # Skip "more" objects
                    if comment.get('kind') == 'more':
                        continue
                    
                    body = comment.get('body', '')
                    created_utc = comment.get('created_utc', 0)
                    
                    if body and created_utc >= cutoff_time:
                        comments_list.append({
                            'body': body,
                            'created_utc': created_utc,
                            'score': comment.get('score', 0)
                        })
                    
                    # Recursively get replies
                    if 'replies' in comment and comment['replies']:
                        if isinstance(comment['replies'], dict) and 'data' in comment['replies']:
                            self._extract_comments_from_json(comment['replies']['data'], comments_list, cutoff_time, depth + 1)
    
    
    def _extract_team_mentions(self, text: str, home_abbrev: str, away_abbrev: str) -> List[Dict]:
        """
        Extract team mentions from text with sentiment context.
        More lenient matching to catch more mentions.
        
        Returns:
            List of dicts with 'team', 'sentiment', 'confidence', 'is_negated'
        """
        mentions = []
        text_lower = text.lower()
        
        # Find all team mentions (both teams)
        home_matches = list(self.team_patterns[home_abbrev].finditer(text))
        away_matches = list(self.team_patterns[away_abbrev].finditer(text))
        
        # If we have both teams mentioned, this is likely game-relevant
        has_both_teams = len(home_matches) > 0 and len(away_matches) > 0
        
        # Check context around each mention
        for match in home_matches + away_matches:
            team = home_abbrev if match in home_matches else away_abbrev
            start = max(0, match.start() - 100)  # Larger context window
            end = min(len(text), match.end() + 100)
            context = text[start:end].lower()
            
            # More lenient: if both teams mentioned, or has betting keyword, or has vs/at
            has_betting_keyword = any(keyword in context for keyword in self.pick_keywords)
            has_game_context = any(word in context for word in ['vs', 'versus', '@', 'at', 'game', 'matchup', 'play'])
            
            # Accept if: betting keyword OR (both teams mentioned AND game context)
            if not (has_betting_keyword or (has_both_teams and has_game_context)):
                continue  # Skip non-relevant mentions
            
            # Check for negation
            is_negated = any(neg_word in context for neg_word in self.negation_words)
            
            # Get VADER sentiment score
            sentiment_score = 0.0
            if self.analyzer:
                vs = self.analyzer.polarity_scores(context)
                sentiment_score = vs['compound']
            
            # Determine pick direction (positive = picking, negative = fading)
            if is_negated:
                pick_direction = -1  # Fading this team
            else:
                pick_direction = 1 if sentiment_score > 0 else -1
            
            # Weight by sentiment strength
            confidence_weight = abs(sentiment_score) if self.analyzer else 0.5
            
            mentions.append({
                'team': team,
                'pick_direction': pick_direction,  # 1 = picking, -1 = fading
                'confidence': confidence_weight,
                'is_negated': is_negated,
                'context': context
            })
        
        return mentions
    
    def get_public_sentiment(self, away_team: str, home_team: str, game_date: Optional[str] = None,
                           home_win_probability: Optional[float] = None) -> Dict:
        """
        Get public sentiment for a game using betting odds (primary method).
        Falls back to Reddit if odds not available and Reddit is configured.
        
        Args:
            away_team: Away team name (can be full name or abbreviation)
            home_team: Home team name (can be full name or abbreviation)
            game_date: Game date (YYYY-MM-DD format). If None, uses today.
            home_win_probability: Model's home win probability (0-1). Used to generate odds.
        
        Returns:
            Dict with:
                - home_pct: Percentage favoring home team (0-100)
                - away_pct: Percentage favoring away team (0-100)
                - total_mentions: Number of relevant mentions found (or 100 for odds-based)
                - confidence: 'high', 'medium', 'low', or 'insufficient'
                - source: 'betting_odds' or 'reddit'
        """
        # Normalize team names
        home_team_abbrev = self._normalize_team_name(home_team)
        away_team_abbrev = self._normalize_team_name(away_team)
        
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check cache first
        cached = self._get_cached_sentiment(home_team_abbrev, away_team_abbrev, game_date)
        if cached:
            return cached
        
        # PRIMARY METHOD: Real NLP analysis from Reddit (no API setup needed!)
        # Uses Reddit's public JSON API to get actual people's opinions
        if self.reddit:
            reddit_result = self._get_sentiment_from_reddit(home_team_abbrev, away_team_abbrev, game_date)
            # Only use Reddit if we got sufficient data
            if reddit_result.get('confidence') != 'insufficient' or reddit_result.get('total_mentions', 0) >= 5:
                return reddit_result
        
        # FALLBACK: Use betting odds if Reddit data is insufficient
        if home_win_probability is not None:
            logger.info("Reddit data insufficient, falling back to odds-based sentiment")
            return self._get_sentiment_from_odds(home_win_probability, home_team_abbrev, away_team_abbrev, game_date)
        
        # Last resort: default
        logger.info("Using default 50-50 sentiment (no data available)")
        return {
            'home_pct': 50.0,
            'away_pct': 50.0,
            'total_mentions': 0,
            'confidence': 'insufficient',
            'source': 'default',
            'error': 'No Reddit or odds data available'
        }
    
    def _get_sentiment_from_odds(self, home_win_prob: float, home_team: str, away_team: str, 
                                 game_date: str) -> Dict:
        """
        Derive public sentiment from betting odds.
        Lower odds = more public money = higher sentiment.
        """
        try:
            # Import odds generator
            from src.odds_scraper import generate_bookmaker_odds
            
            # Generate odds from model probability
            odds_data = generate_bookmaker_odds(home_win_prob)
            
            # Use Pinnacle odds (sharpest book, best reflects public sentiment)
            pinnacle = odds_data.get('bookmakers', {}).get('Pinnacle', {})
            home_odds = pinnacle.get('home', 2.0)
            away_odds = pinnacle.get('away', 2.0)
            
            # Convert odds to implied probabilities (what public thinks)
            # Lower odds = more public money = higher sentiment
            home_implied_prob = 1 / home_odds
            away_implied_prob = 1 / away_odds
            
            # Normalize to percentages
            total = home_implied_prob + away_implied_prob
            home_pct = (home_implied_prob / total) * 100
            away_pct = (away_implied_prob / total) * 100
            
            # Confidence based on how far from 50-50 (more lopsided = higher confidence)
            deviation = abs(home_pct - 50)
            if deviation > 20:
                confidence = 'high'
            elif deviation > 10:
                confidence = 'medium'
            elif deviation > 5:
                confidence = 'low'
            else:
                confidence = 'low'  # Even odds = low confidence in sentiment
            
            result = {
                'home_pct': round(home_pct, 1),
                'away_pct': round(away_pct, 1),
                'total_mentions': 100,  # Odds represent aggregated public opinion
                'confidence': confidence,
                'source': 'betting_odds',
                'home_odds': home_odds,
                'away_odds': away_odds
            }
            
            # Cache result
            self._cache_sentiment(home_team, away_team, game_date, 
                                home_pct, away_pct, 100, confidence)
            
            logger.info(f"Sentiment from odds: {home_team} {home_pct:.1f}% vs {away_team} {away_pct:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error getting sentiment from odds: {e}")
            # Fallback to Reddit if available
            if self.reddit:
                return self._get_sentiment_from_reddit(home_team, away_team, game_date)
            return {
                'home_pct': 50.0,
                'away_pct': 50.0,
                'total_mentions': 0,
                'confidence': 'insufficient',
                'source': 'error',
                'error': str(e)
            }
    
    def _get_sentiment_from_reddit(self, home_team: str, away_team: str, game_date: str) -> Dict:
        """
        Get sentiment from Reddit (fallback method).
        """
        # Normalize team names
        home_team_abbrev = self._normalize_team_name(home_team)
        away_team_abbrev = self._normalize_team_name(away_team)
        
        # Fetch comments from multiple subreddits with team-specific searches
        all_comments = []
        team_names_for_search = [home_team_abbrev, away_team_abbrev, 
                                 self.team_abbrev_to_full.get(home_team_abbrev, ''),
                                 self.team_abbrev_to_full.get(away_team_abbrev, '')]
        team_names_for_search = [t for t in team_names_for_search if t]  # Remove empty
        
        # Fetch from all configured subreddits
        for subreddit_name in self.subreddits:
            try:
                comments = self._fetch_reddit_comments(
                    subreddit_name, 
                    team_names_for_search,
                    time_window_hours=72,  # 3 days for more data
                    limit=150
                )
                all_comments.extend(comments)
                logger.info(f"Fetched {len(comments)} comments from r/{subreddit_name}")
                time.sleep(1)  # Rate limiting between subreddits
            except Exception as e:
                logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                continue
        
        # Remove duplicates based on body text
        seen_bodies = set()
        unique_comments = []
        for comment in all_comments:
            body_hash = hash(comment['body'][:100])  # Hash first 100 chars
            if body_hash not in seen_bodies:
                seen_bodies.add(body_hash)
                unique_comments.append(comment)
        
        all_comments = unique_comments
        logger.info(f"Total unique comments collected: {len(all_comments)}")
        
        # If we got very few comments, log a warning
        if len(all_comments) < 50:
            logger.warning(f"Low comment count ({len(all_comments)}). May indicate Reddit rate limiting or lack of discussion.")
        
        # Extract team mentions with sentiment
        home_mentions = []
        away_mentions = []
        
        for comment in all_comments:
            text = comment.get('body', '')
            mentions = self._extract_team_mentions(text, home_team_abbrev, away_team_abbrev)
            
            for mention in mentions:
                # Weight by comment score (upvotes)
                score_weight = min(comment.get('score', 0) / 10.0, 2.0)  # Cap at 2x
                mention['weight'] = mention['confidence'] * (1 + score_weight * 0.1)
                
                if mention['team'] == home_team_abbrev:
                    home_mentions.append(mention)
                elif mention['team'] == away_team_abbrev:
                    away_mentions.append(mention)
        
        # Calculate sentiment scores
        home_score = sum(m['pick_direction'] * m['weight'] for m in home_mentions)
        away_score = sum(m['pick_direction'] * m['weight'] for m in away_mentions)
        
        total_mentions = len(home_mentions) + len(away_mentions)
        
        # Log detailed stats
        logger.info(f"Sentiment analysis for {away_team_abbrev} @ {home_team_abbrev}:")
        logger.info(f"  - Total comments analyzed: {len(all_comments)}")
        logger.info(f"  - Home team mentions: {len(home_mentions)}")
        logger.info(f"  - Away team mentions: {len(away_mentions)}")
        logger.info(f"  - Total relevant mentions: {total_mentions}")
        
        # Determine confidence level (lowered thresholds for more data)
        if total_mentions < 10:
            confidence = 'insufficient'
        elif total_mentions < 25:
            confidence = 'low'
        elif total_mentions < 50:
            confidence = 'medium'
        else:
            confidence = 'high'
        
        # Calculate percentages
        total_score = abs(home_score) + abs(away_score)
        if total_score == 0:
            # No clear sentiment, default to 50-50
            home_pct = 50.0
            away_pct = 50.0
        else:
            # Normalize scores to percentages
            home_pct = 50 + (home_score / total_score) * 50
            away_pct = 50 + (away_score / total_score) * 50
            
            # Ensure percentages are in valid range
            home_pct = max(0, min(100, home_pct))
            away_pct = max(0, min(100, away_pct))
        
        result = {
            'home_pct': round(home_pct, 1),
            'away_pct': round(away_pct, 1),
            'total_mentions': total_mentions,
            'confidence': confidence,
            'source': 'reddit',  # Real NLP from actual Reddit comments!
            'comments_analyzed': len(all_comments)
        }
        
        # Cache result
        self._cache_sentiment(home_team_abbrev, away_team_abbrev, game_date, 
                            home_pct, away_pct, total_mentions, confidence)
        
        return result


# Convenience function for easy import
def get_public_sentiment(away_team: str, home_team: str, game_date: Optional[str] = None) -> Dict:
    """
    Convenience function to get public sentiment.
    
    Args:
        away_team: Away team name
        home_team: Home team name  
        game_date: Game date (YYYY-MM-DD) or None for today
    
    Returns:
        Dict with sentiment scores and confidence
    """
    analyzer = SentimentAnalyzer()
    return analyzer.get_public_sentiment(away_team, home_team, game_date)

