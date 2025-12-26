# NLP Sentiment Analysis Implementation - Detailed Prompt for AI

## Context

I have an NBA game prediction system that uses machine learning to predict game outcomes. I want to add a **Public Sentiment Analysis** feature that analyzes what people are actually saying about games on Reddit to provide a complementary view to the model's predictions.

## Current Implementation Attempt

### What We Tried

1. **Initial Approach: Reddit API with PRAW**
   - Attempted to use PRAW (Python Reddit API Wrapper) with OAuth authentication
   - Required Reddit API credentials (client_id, client_secret, user_agent)
   - User encountered issues setting up Reddit API credentials (form submission problems)
   - Abandoned this approach due to setup complexity

2. **Second Approach: Reddit Public JSON API**
   - Switched to Reddit's public JSON endpoints (no authentication required)
   - Used `requests` library to fetch: `https://www.reddit.com/r/{subreddit}/{sort}.json`
   - Fetched from multiple subreddits: r/sportsbook, r/nba, r/fantasybball, r/dfsports, r/sportsbetting
   - Collected 200-1000+ comments per game
   - Used VADER sentiment analysis to analyze comment tone
   - Implemented keyword extraction for betting-specific terms
   - **Problem**: Still getting "Insufficient data" errors - not enough relevant mentions found

3. **Third Approach: Betting Odds as Proxy**
   - Used betting odds as a proxy for public sentiment (no Reddit needed)
   - Lower odds = more public money = higher sentiment
   - This worked but doesn't provide real NLP analysis of what people think

### Current Code Structure

The sentiment analyzer is in `src/sentiment_analyzer.py` with the following components:

1. **Data Collection**:
   - `_fetch_reddit_json()`: Fetches Reddit posts using public JSON API
   - `_fetch_reddit_comments()`: Fetches comments from posts
   - `_fetch_post_comments()`: Fetches nested comments from specific posts
   - Searches multiple subreddits, hot/new posts, team-specific searches
   - 72-hour time window, up to 150 posts per subreddit

2. **Team Matching**:
   - `_normalize_team_name()`: Converts team names to abbreviations
   - Extensive team nickname mapping (50+ variations)
   - Regex patterns for flexible team name detection

3. **Sentiment Analysis**:
   - `_extract_team_mentions()`: Finds team mentions with betting context
   - Uses VADER sentiment analyzer for emotional tone
   - Keyword detection: "taking", "picking", "betting on", "hammer", "fade", "lock", etc.
   - Negation handling: "not touching", "fading", "avoid"
   - Weighting by comment score (upvotes) and sentiment strength

4. **Scoring**:
   - Calculates weighted scores for home/away teams
   - Converts to percentages (0-100%)
   - Confidence levels: high (50+ mentions), medium (25-49), low (10-24), insufficient (<10)

### Problems Encountered

1. **Insufficient Data**: Even with aggressive collection (5 subreddits, 72-hour window, 200+ posts), we're getting <10 relevant mentions per game
2. **Keyword Matching Too Strict**: Comments mentioning teams but without explicit betting keywords are ignored
3. **Reddit Rate Limiting**: Multiple subreddit searches can hit rate limits
4. **Comment Quality**: Many comments don't contain clear betting picks ("Lakers are good" vs "Taking Lakers -5")

## What We Want to Achieve

### Goal

Create a robust sentiment analysis system that:
- Analyzes **actual human opinions** from Reddit (not just betting odds)
- Provides sentiment scores like "68% public favors Lakers" or "Public Split: 52-48"
- Works reliably for 80%+ of games
- Processes in <30 seconds per game
- Requires **zero setup** (no API keys, no authentication)

### Requirements

1. **Data Sources**:
   - Primary: Reddit r/sportsbook (explicit betting picks)
   - Secondary: r/nba, r/fantasybball (game discussions)
   - Time window: Last 24-72 hours before game time

2. **Analysis Method**:
   - Hybrid approach: Keyword extraction + VADER sentiment + context analysis
   - Must handle betting-specific language ("lock", "fade", "hammer", "taking")
   - Must handle negation ("not touching", "fading", "avoid")
   - Must weight by confidence indicators ("confidently taking" > "maybe Lakers")

3. **Output Format**:
   ```python
   {
       'home_pct': 65.3,      # Percentage favoring home team
       'away_pct': 34.7,      # Percentage favoring away team
       'total_mentions': 42,  # Number of relevant mentions
       'confidence': 'medium' # 'high', 'medium', 'low', 'insufficient'
   }
   ```

4. **Thresholds**:
   - Minimum 15-20 relevant mentions for reliable score
   - Display "Insufficient data" if <15 mentions
   - Confidence: High (50+), Medium (25-49), Low (15-24), Insufficient (<15)

## Technical Constraints

- **Free APIs only**: Reddit public JSON API (no PRAW/OAuth)
- **Python libraries**: requests, vaderSentiment, re (regex)
- **No external paid services**
- **Must work without user setup** (no API keys)
- **Performance**: <30 seconds per game
- **Caching**: 6-hour cache to avoid repeated API calls

## Current Issues to Solve

1. **Not enough mentions found**: Even with 200-1000 comments collected, we're finding <10 relevant mentions
   - Possible causes:
     - Keyword matching too strict
     - Team name matching not catching all variations
     - Comments don't use explicit betting language
     - Need better context detection

2. **Improve mention detection**:
   - Currently requires betting keywords OR (both teams mentioned + game context)
   - Maybe too lenient or too strict?
   - Need better game context detection

3. **Better data collection**:
   - Maybe need to search for specific game threads?
   - Look for "Game Thread" or "Pre-Game Thread" posts?
   - Search for team matchups explicitly?

## What We Need

A solution that:
1. **Finds more relevant mentions** (15-20+ per game minimum)
2. **Better understands betting language** in context
3. **Handles edge cases** gracefully (no data, API failures, ambiguous comments)
4. **Works reliably** for most games (80%+ success rate)
5. **Is maintainable** and doesn't break with Reddit changes

## Code Location

- Main module: `src/sentiment_analyzer.py`
- Class: `SentimentAnalyzer`
- Main method: `get_public_sentiment(away_team, home_team, game_date, home_win_probability)`
- Currently falls back to betting odds if Reddit data insufficient

## Success Criteria

The implementation is successful if:
1. ✅ Sentiment scores displayed for 80%+ of games
2. ✅ Scores feel intuitive (heavily favored teams show 65%+, toss-ups around 50-50)
3. ✅ System handles edge cases gracefully
4. ✅ Processing time <30 seconds per game
5. ✅ No user setup required (works out of the box)

## Questions to Consider

1. Should we prioritize r/sportsbook (explicit picks) or r/nba (larger volume)?
2. How should we handle comments mentioning multiple games?
3. Should sentiment from game day vs day before be weighted differently?
4. Should we filter out bot/spam comments?
5. How to handle comments that are ambiguous ("Lakers might win" vs "Lakers will win")?

## Next Steps Needed

1. Improve mention detection to find more relevant comments
2. Better keyword/context matching for betting language
3. Possibly add game thread detection
4. Improve team name matching
5. Better handling of ambiguous sentiment
6. Add spam/bot filtering if needed

