"""
Improved NBA Sentiment Analyzer

Solves the key issues:
1. More flexible mention detection (not just explicit betting keywords)
2. Better team name matching (nicknames, cities, players)
3. Smarter context detection (implied preferences, not just explicit picks)
4. Sentence-level analysis for more granular sentiment
"""

import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
import logging

logger = logging.getLogger(__name__)

# Try to import VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available. Install with: pip install vaderSentiment")

@dataclass
class TeamMention:
    """Represents a single mention of a team in a comment."""
    team: str  # 'home' or 'away'
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    context: str  # The matching text
    weight: float  # Comment score weight

class ImprovedSentimentAnalyzer:
    """
    Improved sentiment analyzer that finds more relevant mentions
    by using relaxed matching and smarter context detection.
    """
    
    # Comprehensive team mappings: abbreviation -> [all possible references]
    TEAM_REFERENCES = {
        'ATL': ['hawks', 'atlanta', 'atl', 'trae', 'young'],
        'BOS': ['celtics', 'boston', 'bos', 'celts', 'cs', 'tatum', 'jaylen', 'jays'],
        'BKN': ['nets', 'brooklyn', 'bkn', 'bk'],
        'CHA': ['hornets', 'charlotte', 'cha', 'lamelo', 'ball'],
        'CHI': ['bulls', 'chicago', 'chi', 'lavine'],
        'CLE': ['cavs', 'cavaliers', 'cleveland', 'cle', 'garland', 'mitchell'],
        'DAL': ['mavs', 'mavericks', 'dallas', 'dal', 'luka', 'doncic', 'kyrie'],
        'DEN': ['nuggets', 'denver', 'den', 'nugs', 'jokic', 'joker', 'murray'],
        'DET': ['pistons', 'detroit', 'det'],
        'GSW': ['warriors', 'golden state', 'gsw', 'gs', 'dubs', 'curry', 'steph', 'klay'],
        'HOU': ['rockets', 'houston', 'hou'],
        'IND': ['pacers', 'indiana', 'ind', 'haliburton', 'tyrese'],
        'LAC': ['clippers', 'lac', 'clips', 'kawhi', 'leonard'],
        'LAL': ['lakers', 'los angeles lakers', 'lal', 'la lakers', 'lebron', 'james', 'ad', 'davis'],
        'MEM': ['grizzlies', 'memphis', 'mem', 'grizz', 'ja', 'morant'],
        'MIA': ['heat', 'miami', 'mia', 'butler', 'jimmy', 'bam'],
        'MIL': ['bucks', 'milwaukee', 'mil', 'giannis', 'antetokounmpo'],
        'MIN': ['timberwolves', 'minnesota', 'min', 'wolves', 'twolves', 'ant', 'edwards', 'kat'],
        'NOP': ['pelicans', 'new orleans', 'nop', 'pels', 'zion', 'ingram'],
        'NYK': ['knicks', 'new york', 'nyk', 'ny', 'brunson', 'jalen'],
        'OKC': ['thunder', 'oklahoma', 'okc', 'shai', 'sga', 'gilgeous'],
        'ORL': ['magic', 'orlando', 'orl', 'paolo', 'banchero'],
        'PHI': ['sixers', 'philadelphia', 'phi', 'philly', '76ers', 'embiid', 'joel', 'maxey'],
        'PHX': ['suns', 'phoenix', 'phx', 'booker', 'book', 'durant', 'kd', 'beal'],
        'POR': ['blazers', 'portland', 'por', 'trail blazers'],
        'SAC': ['kings', 'sacramento', 'sac', 'fox', 'sabonis'],
        'SAS': ['spurs', 'san antonio', 'sas', 'wemby', 'wembanyama', 'victor'],
        'TOR': ['raptors', 'toronto', 'tor', 'raps'],
        'UTA': ['jazz', 'utah', 'uta'],
        'WAS': ['wizards', 'washington', 'was', 'wiz'],
    }
    
    # Positive betting indicators (explicit picks)
    POSITIVE_BETTING_KEYWORDS = [
        r'\btaking\b', r'\bpicking\b', r'\bbetting\s+on\b', r'\bhammer\b', r'\bslam\b',
        r'\block\b', r'\block\s+of\s+the\s+(day|week|night)\b', r'\bpotd\b',
        r'\bmax\s+bet\b', r'\bheavy\s+on\b', r'\bgoing\s+with\b', r'\blike\b',
        r'\blove\b', r'\bfancy\b', r'\bparlay\b', r'\bconfident\b', r'\bsure\s+thing\b',
        r'\bpound\b', r'\bsmash\b', r'\briding\b', r'\broll\s+with\b',
        r'\bml\b', r'\bmoneyline\b', r'\bspread\b', r'\bats\b', r'\bover\b', r'\bunder\b'
    ]
    
    # Negative betting indicators (fading/avoiding)
    NEGATIVE_BETTING_KEYWORDS = [
        r'\bfade\b', r'\bfading\b', r'\bagainst\b', r'\bavoid\b', r'\bstay\s+away\b',
        r'\bnot\s+touching\b', r'\btrash\b', r'\bgarbage\b', r'\bterrible\b',
        r'\btrap\b', r'\bpublic\s+trap\b', r'\bsquare\b', r'\boverrated\b',
        r'\bbust\b', r'\bwont\s+cover\b', r'\bcant\s+cover\b'
    ]
    
    # General positive sentiment (not explicit picks but favorable)
    POSITIVE_SENTIMENT_PHRASES = [
        r'\bwill\s+win\b', r'\bgoing\s+to\s+win\b', r'\bshould\s+win\b',
        r'\beasy\s+win\b', r'\bblowout\b', r'\bdominate\b', r'\bcruise\b',
        r'\bno\s+chance\s+they\s+lose\b', r'\btoo\s+good\b', r'\bon\s+fire\b',
        r'\bhot\b', r'\brolling\b', r'\bstreaking\b', r'\bunstoppable\b',
        r'\bcant\s+lose\b', r'\bwont\s+lose\b'
    ]
    
    # General negative sentiment
    NEGATIVE_SENTIMENT_PHRASES = [
        r'\bwill\s+lose\b', r'\bgoing\s+to\s+lose\b', r'\bno\s+chance\b',
        r'\boverrated\b', r'\bstruggling\b', r'\btanking\b', r'\bbad\b',
        r'\bterrible\b', r'\bawful\b', r'\binjured\b', r'\bresting\b',
        r'\bmissing\b', r'\bout\b', r'\bdoubtful\b', r'\bquestionable\b'
    ]
    
    # Game context indicators
    GAME_CONTEXT_PATTERNS = [
        r'\btonight\b', r'\btonights?\s+game\b', r'\bgame\s+tonight\b',
        r'\btoday\b', r'\btodays?\s+game\b', r'\bgame\s+today\b',
        r'\btomorrow\b', r'\btomorrows?\s+game\b', 
        r'\bvs\.?\b', r'\b@\b', r'\bat\b', r'\bagainst\b',
        r'\bmatchup\b', r'\bplayoff\b', r'\bseries\b',
        r'\bspread\b', r'\bline\b', r'\bodds\b', r'\bou\b', r'\bo/u\b',
        r'\bpick\b', r'\bbet\b', r'\bwager\b', r'\bplay\b'
    ]
    
    # Subreddits to search (ordered by relevance)
    SUBREDDITS = [
        ('sportsbook', 1.5),      # Explicit betting picks - highest weight
        ('sportsbetting', 1.3),   # More betting content
        ('nba', 1.0),             # High volume general NBA discussion
        ('fantasybball', 0.8),    # Fantasy relevant but less betting-focused
        ('dfsports', 0.9),        # DFS picks often align with game outcomes
    ]
    
    def __init__(self, cache_hours: int = 6):
        """Initialize the analyzer with caching."""
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
            logger.warning("VADER not available, sentiment analysis will be limited")
        
        self.cache = {}
        self.cache_duration = timedelta(hours=cache_hours)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _get_cache_key(self, away_team: str, home_team: str, game_date: str) -> str:
        """Generate a unique cache key for a game."""
        key = f"{away_team}_{home_team}_{game_date}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _normalize_team(self, team_name: str) -> Optional[str]:
        """Convert any team reference to standard abbreviation."""
        team_lower = team_name.lower().strip()
        
        # Direct abbreviation match
        for abbrev in self.TEAM_REFERENCES:
            if team_lower == abbrev.lower():
                return abbrev
        
        # Check all references
        for abbrev, refs in self.TEAM_REFERENCES.items():
            if team_lower in refs:
                return abbrev
            # Partial match for longer names
            for ref in refs:
                if ref in team_lower or team_lower in ref:
                    return abbrev
        
        return None
    
    def _build_team_pattern(self, team_abbrev: str) -> re.Pattern:
        """Build regex pattern to match all team references."""
        refs = self.TEAM_REFERENCES.get(team_abbrev, [team_abbrev.lower()])
        # Escape special characters and join with OR
        patterns = [re.escape(ref) for ref in refs]
        patterns.append(re.escape(team_abbrev.lower()))
        return re.compile(r'\b(' + '|'.join(patterns) + r')\b', re.IGNORECASE)
    
    def _fetch_reddit_posts(self, subreddit: str, sort: str = 'hot', 
                           limit: int = 100) -> List[dict]:
        """Fetch posts from a subreddit using public JSON API."""
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params = {'limit': min(limit, 100)}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {}).get('children', [])
            elif response.status_code == 429:
                time.sleep(2)  # Rate limited, wait and return empty
            return []
        except Exception as e:
            logger.debug(f"Error fetching r/{subreddit}: {e}")
            return []
    
    def _fetch_post_comments(self, permalink: str, limit: int = 200) -> List[Tuple[str, float]]:
        """Fetch comments from a specific post."""
        url = f"https://www.reddit.com{permalink}.json"
        params = {'limit': limit, 'depth': 3}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                comments = []
                self._extract_comments(data, comments)
                return comments
            return []
        except Exception as e:
            logger.debug(f"Error fetching comments: {e}")
            return []
    
    def _extract_comments(self, data: any, comments: List[Tuple[str, float]]):
        """Recursively extract comment text from Reddit JSON."""
        if isinstance(data, list):
            for item in data:
                self._extract_comments(item, comments)
        elif isinstance(data, dict):
            if data.get('kind') == 't1':  # Comment
                comment_data = data.get('data', {})
                body = comment_data.get('body', '')
                score = comment_data.get('score', 1)
                if body and body != '[deleted]' and body != '[removed]':
                    comments.append((body, max(1, score)))
                # Get nested replies
                replies = comment_data.get('replies', {})
                if replies:
                    self._extract_comments(replies, comments)
            elif 'data' in data:
                self._extract_comments(data['data'], comments)
            elif 'children' in data:
                self._extract_comments(data['children'], comments)
    
    def _search_subreddit(self, subreddit: str, query: str, 
                          time_filter: str = 'week') -> List[dict]:
        """Search a subreddit for specific terms."""
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            'q': query,
            'restrict_sr': 'true',
            't': time_filter,
            'limit': 50,
            'sort': 'relevance'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('data', {}).get('children', [])
            return []
        except Exception as e:
            logger.debug(f"Error searching r/{subreddit}: {e}")
            return []
    
    def _collect_comments(self, home_abbrev: str, away_abbrev: str) -> List[Tuple[str, float, str]]:
        """
        Collect relevant comments from multiple sources.
        Returns list of (comment_text, score, subreddit).
        """
        all_comments = []
        
        # Get team references for search
        home_refs = self.TEAM_REFERENCES.get(home_abbrev, [home_abbrev.lower()])
        away_refs = self.TEAM_REFERENCES.get(away_abbrev, [away_abbrev.lower()])
        
        # Search queries
        matchup_queries = [
            f"{home_refs[0]} {away_refs[0]}",  # e.g., "lakers celtics"
            f"{home_abbrev} vs {away_abbrev}",
        ]
        
        for subreddit, weight in self.SUBREDDITS:
            try:
                # 1. Get hot posts (active game day threads)
                hot_posts = self._fetch_reddit_posts(subreddit, 'hot', 50)
                
                # 2. Get new posts (recent discussions)
                new_posts = self._fetch_reddit_posts(subreddit, 'new', 50)
                
                # 3. Search for matchup-specific threads
                search_posts = []
                for query in matchup_queries[:2]:  # Limit queries to avoid rate limits
                    search_posts.extend(self._search_subreddit(subreddit, query, 'week'))
                    time.sleep(0.5)  # Gentle rate limiting
                
                # Combine and dedupe posts
                all_posts = {}
                for post_list in [hot_posts, new_posts, search_posts]:
                    for post in post_list:
                        post_data = post.get('data', {})
                        post_id = post_data.get('id')
                        if post_id and post_id not in all_posts:
                            all_posts[post_id] = post_data
                
                # Filter for relevant posts and fetch comments
                for post_id, post_data in list(all_posts.items())[:30]:  # Limit to top 30
                    title = post_data.get('title', '').lower()
                    selftext = post_data.get('selftext', '').lower()
                    permalink = post_data.get('permalink', '')
                    
                    # Check if post is relevant to our game
                    is_relevant = self._is_game_relevant(
                        title + ' ' + selftext, home_abbrev, away_abbrev
                    )
                    
                    # Also include general betting threads
                    is_betting_thread = any(kw in title.lower() for kw in [
                        'daily', 'picks', 'thread', 'discussion', 'nba', 'basketball',
                        'potd', 'parlay', 'best bet', 'lock'
                    ])
                    
                    if is_relevant or is_betting_thread:
                        # Include post title/text as a "comment"
                        if selftext and len(selftext) > 20:
                            all_comments.append((selftext, post_data.get('score', 1) * 2, subreddit))
                        
                        # Fetch comments from this post
                        if permalink:
                            comments = self._fetch_post_comments(permalink, 100)
                            for comment_text, comment_score in comments:
                                all_comments.append((comment_text, comment_score * weight, subreddit))
                            
                            time.sleep(0.3)  # Rate limiting between posts
                
                time.sleep(0.5)  # Rate limiting between subreddits
            except Exception as e:
                logger.debug(f"Error processing subreddit {subreddit}: {e}")
                continue
        
        return all_comments
    
    def _is_game_relevant(self, text: str, home_abbrev: str, away_abbrev: str) -> bool:
        """Check if text is relevant to the specific game."""
        text_lower = text.lower()
        
        home_pattern = self._build_team_pattern(home_abbrev)
        away_pattern = self._build_team_pattern(away_abbrev)
        
        has_home = bool(home_pattern.search(text_lower))
        has_away = bool(away_pattern.search(text_lower))
        
        # Either team mentioned + game context, or both teams mentioned
        if has_home and has_away:
            return True
        
        if has_home or has_away:
            # Check for game context
            for pattern in self.GAME_CONTEXT_PATTERNS:
                if re.search(pattern, text_lower):
                    return True
        
        return False
    
    def _analyze_comment(self, comment: str, home_abbrev: str, away_abbrev: str) -> List[TeamMention]:
        """
        Analyze a single comment for team sentiment.
        Uses a multi-layered approach:
        1. Explicit betting keywords (highest confidence)
        2. Positive/negative sentiment phrases
        3. General VADER sentiment with team context
        """
        mentions = []
        comment_lower = comment.lower()
        
        home_pattern = self._build_team_pattern(home_abbrev)
        away_pattern = self._build_team_pattern(away_abbrev)
        
        has_home = bool(home_pattern.search(comment_lower))
        has_away = bool(away_pattern.search(comment_lower))
        
        if not has_home and not has_away:
            return mentions
        
        # Split comment into sentences for more granular analysis
        sentences = re.split(r'[.!?]+', comment_lower)
        
        for sentence in sentences:
            if len(sentence.strip()) < 5:
                continue
                
            home_in_sentence = bool(home_pattern.search(sentence))
            away_in_sentence = bool(away_pattern.search(sentence))
            
            if not home_in_sentence and not away_in_sentence:
                continue
            
            # Determine which team the sentence is about
            target_team = None
            if home_in_sentence and not away_in_sentence:
                target_team = 'home'
            elif away_in_sentence and not home_in_sentence:
                target_team = 'away'
            else:
                # Both teams mentioned - need to determine context
                home_match = home_pattern.search(sentence)
                away_match = away_pattern.search(sentence)
                
                # Check for "X vs Y" pattern and look at surrounding sentiment
                vs_match = re.search(r'vs\.?|@|against|at', sentence)
                if vs_match and home_match and away_match:
                    # Team mentioned before "vs" is often the pick
                    if home_match.start() < vs_match.start():
                        target_team = 'home'
                    else:
                        target_team = 'away'
            
            if not target_team:
                continue
            
            # Calculate sentiment score and confidence
            sentiment_score = 0.0
            confidence = 0.0
            
            # Check for explicit betting keywords (highest confidence)
            for pattern in self.POSITIVE_BETTING_KEYWORDS:
                if re.search(pattern, sentence):
                    sentiment_score = 0.8
                    confidence = 0.9
                    break
            
            if sentiment_score == 0:
                for pattern in self.NEGATIVE_BETTING_KEYWORDS:
                    if re.search(pattern, sentence):
                        sentiment_score = -0.8
                        confidence = 0.9
                        # Negate target team
                        target_team = 'away' if target_team == 'home' else 'home'
                        break
            
            # Check for sentiment phrases (medium confidence)
            if sentiment_score == 0:
                for pattern in self.POSITIVE_SENTIMENT_PHRASES:
                    if re.search(pattern, sentence):
                        sentiment_score = 0.6
                        confidence = 0.7
                        break
            
            if sentiment_score == 0:
                for pattern in self.NEGATIVE_SENTIMENT_PHRASES:
                    if re.search(pattern, sentence):
                        sentiment_score = -0.6
                        confidence = 0.7
                        target_team = 'away' if target_team == 'home' else 'home'
                        break
            
            # Fall back to VADER sentiment (lower confidence)
            if sentiment_score == 0 and self.vader:
                vader_scores = self.vader.polarity_scores(sentence)
                compound = vader_scores['compound']
                if abs(compound) > 0.2:  # Only use if sentiment is clear enough
                    sentiment_score = compound * 0.5
                    confidence = 0.5
                    if compound < 0:
                        target_team = 'away' if target_team == 'home' else 'home'
            
            if sentiment_score != 0 and confidence > 0:
                mentions.append(TeamMention(
                    team=target_team,
                    sentiment_score=abs(sentiment_score),
                    confidence=confidence,
                    context=sentence.strip()[:100],
                    weight=1.0
                ))
        
        return mentions
    
    def get_public_sentiment(self, away_team: str, home_team: str, 
                            game_date: str,
                            home_win_probability: Optional[float] = None) -> Dict:
        """
        Get public sentiment for a game.
        
        Args:
            away_team: Away team name or abbreviation
            home_team: Home team name or abbreviation  
            game_date: Game date (for caching)
            home_win_probability: Optional model probability (for fallback)
            
        Returns:
            Dict with home_pct, away_pct, total_mentions, confidence, source
        """
        # Normalize team names
        home_abbrev = self._normalize_team(home_team)
        away_abbrev = self._normalize_team(away_team)
        
        if not home_abbrev or not away_abbrev:
            return self._fallback_result(home_win_probability, "Unknown team")
        
        # Check cache
        cache_key = self._get_cache_key(away_abbrev, home_abbrev, game_date)
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_result
        
        logger.info(f"Collecting sentiment for {away_abbrev} @ {home_abbrev}...")
        
        # Collect comments
        comments = self._collect_comments(home_abbrev, away_abbrev)
        logger.info(f"Collected {len(comments)} comments")
        
        # Analyze comments
        all_mentions = []
        for comment_text, comment_score, subreddit in comments:
            mentions = self._analyze_comment(comment_text, home_abbrev, away_abbrev)
            # Weight by comment score
            weight = max(1.0, min(5.0, 1.0 + comment_score / 10))
            for mention in mentions:
                mention.weight = weight * mention.confidence
            all_mentions.extend(mentions)
        
        logger.info(f"Found {len(all_mentions)} team mentions")
        
        # Calculate scores
        if len(all_mentions) < 5:
            result = self._fallback_result(home_win_probability, 
                                          f"Only {len(all_mentions)} mentions found")
        else:
            home_score = sum(m.sentiment_score * m.weight 
                           for m in all_mentions if m.team == 'home')
            away_score = sum(m.sentiment_score * m.weight 
                           for m in all_mentions if m.team == 'away')
            
            total = home_score + away_score
            if total > 0:
                home_pct = (home_score / total) * 100
                away_pct = (away_score / total) * 100
            else:
                home_pct = away_pct = 50.0
            
            # Determine confidence level
            mention_count = len(all_mentions)
            if mention_count >= 50:
                confidence = 'high'
            elif mention_count >= 25:
                confidence = 'medium'
            elif mention_count >= 15:
                confidence = 'low'
            else:
                confidence = 'insufficient'
            
            result = {
                'home_pct': round(home_pct, 1),
                'away_pct': round(away_pct, 1),
                'total_mentions': mention_count,
                'confidence': confidence,
                'source': 'reddit_nlp',
                'home_mentions': len([m for m in all_mentions if m.team == 'home']),
                'away_mentions': len([m for m in all_mentions if m.team == 'away']),
            }
        
        # Cache result
        self.cache[cache_key] = (datetime.now(), result)
        
        return result
    
    def _fallback_result(self, home_win_probability: Optional[float], 
                        reason: str) -> Dict:
        """Generate fallback result when Reddit data is insufficient."""
        if home_win_probability is not None:
            # Use betting odds as proxy
            home_pct = home_win_probability * 100
            return {
                'home_pct': round(home_pct, 1),
                'away_pct': round(100 - home_pct, 1),
                'total_mentions': 0,
                'confidence': 'insufficient',
                'source': 'betting_odds_fallback',
                'fallback_reason': reason
            }
        else:
            return {
                'home_pct': 50.0,
                'away_pct': 50.0,
                'total_mentions': 0,
                'confidence': 'insufficient',
                'source': 'no_data',
                'fallback_reason': reason
            }

