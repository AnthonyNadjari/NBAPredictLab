"""
Local Tweet Counter - Tracks tweets posted in last 24 hours

Since Twitter Free tier doesn't allow reading your own tweets via API,
we track them locally by recording each tweet we post.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# File to store tweet history
TWEET_LOG_FILE = Path("data/tweet_history.json")
RATE_LIMIT_CACHE_FILE = Path("data/latest_rate_limit.json")

# Free tier limit
FREE_TIER_24H_LIMIT = 17


def log_tweet_posted(tweet_id: str, text_preview: str = "", rate_limit_info: Optional[Dict] = None) -> None:
    """
    Record that a tweet was posted.

    Args:
        tweet_id: The ID of the posted tweet
        text_preview: Optional preview of tweet text (first 50 chars)
        rate_limit_info: Optional dict with 'remaining', 'limit', 'reset' from API headers
    """
    try:
        # Load existing log
        tweet_log = _load_tweet_log()

        # Add new entry
        entry = {
            'tweet_id': tweet_id,
            'posted_at': datetime.now(timezone.utc).isoformat(),
            'text_preview': text_preview[:50] if text_preview else ""
        }

        # Add rate limit info if provided
        if rate_limit_info:
            entry['rate_limit_info'] = rate_limit_info

        tweet_log.append(entry)

        # Save updated log
        _save_tweet_log(tweet_log)

        logger.info(f"📝 Logged tweet {tweet_id} to local history")

        # If we have rate limit info from headers, cache it separately for quick access
        if rate_limit_info and rate_limit_info.get('remaining') is not None:
            _save_latest_rate_limit(rate_limit_info)

    except Exception as e:
        logger.error(f"Failed to log tweet: {e}")


def get_tweet_count_24h() -> Dict:
    """
    Count tweets posted in the last 24 hours from local log.

    Returns:
        Dict with 'count', 'limit', 'remaining', 'tweets', and 'oldest_tweet_age_hours'
    """
    try:
        # Load tweet log
        tweet_log = _load_tweet_log()

        # Filter to last 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        tweets_24h = []

        for tweet in tweet_log:
            posted_at = datetime.fromisoformat(tweet['posted_at'])
            if posted_at >= cutoff_time:
                tweets_24h.append({
                    'id': tweet['tweet_id'],
                    'posted_at': posted_at.isoformat(),
                    'text': tweet.get('text_preview', ''),
                    'age_hours': (datetime.now(timezone.utc) - posted_at).total_seconds() / 3600
                })

        # Calculate stats
        count = len(tweets_24h)
        remaining = max(0, FREE_TIER_24H_LIMIT - count)

        # Find oldest tweet age (when we can post again if at limit)
        oldest_tweet_age_hours = None
        if tweets_24h:
            oldest_tweet = min(tweets_24h, key=lambda t: t['posted_at'])
            oldest_tweet_age_hours = oldest_tweet['age_hours']

        # Clean up old tweets (older than 48 hours)
        _cleanup_old_tweets()

        # Try to get the latest rate limit info from API headers
        api_rate_limit = get_latest_rate_limit()

        result = {
            'count': count,
            'limit': FREE_TIER_24H_LIMIT,
            'remaining': remaining,
            'tweets': sorted(tweets_24h, key=lambda t: t['posted_at'], reverse=True),  # Most recent first
            'oldest_tweet_age_hours': oldest_tweet_age_hours,
            'source': 'local_log',
            'api_rate_limit': api_rate_limit  # From response headers if available
        }

        logger.info(f"📊 Tweet count: {count}/{FREE_TIER_24H_LIMIT} (local log)")
        if api_rate_limit:
            logger.info(f"   API headers say: {api_rate_limit.get('remaining')}/{api_rate_limit.get('limit')} remaining")

        return result

    except Exception as e:
        logger.error(f"Error getting tweet count: {e}")
        return {
            'count': 0,
            'limit': FREE_TIER_24H_LIMIT,
            'remaining': FREE_TIER_24H_LIMIT,
            'tweets': [],
            'error': str(e),
            'source': 'local_log'
        }


def reset_tweet_count() -> None:
    """Clear all logged tweets (use with caution)."""
    try:
        _save_tweet_log([])
        logger.info("🗑️ Cleared tweet history log")
    except Exception as e:
        logger.error(f"Failed to reset tweet log: {e}")


def _load_tweet_log() -> List[Dict]:
    """Load tweet history from file."""
    try:
        if TWEET_LOG_FILE.exists():
            with open(TWEET_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load tweet log: {e}")

    return []


def _save_tweet_log(tweet_log: List[Dict]) -> None:
    """Save tweet history to file."""
    try:
        TWEET_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TWEET_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(tweet_log, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save tweet log: {e}")


def _cleanup_old_tweets() -> None:
    """Remove tweets older than 48 hours from the log."""
    try:
        tweet_log = _load_tweet_log()
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=48)

        # Keep only tweets from last 48 hours
        filtered_log = [
            tweet for tweet in tweet_log
            if datetime.fromisoformat(tweet['posted_at']) >= cutoff_time
        ]

        if len(filtered_log) < len(tweet_log):
            removed_count = len(tweet_log) - len(filtered_log)
            _save_tweet_log(filtered_log)
            logger.debug(f"Cleaned up {removed_count} old tweets from log")

    except Exception as e:
        logger.warning(f"Failed to cleanup old tweets: {e}")


def _save_latest_rate_limit(rate_limit_info: Dict) -> None:
    """Save the latest rate limit info from API headers."""
    try:
        cache_data = {
            'remaining': rate_limit_info.get('remaining'),
            'limit': rate_limit_info.get('limit'),
            'reset': rate_limit_info.get('reset'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        RATE_LIMIT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RATE_LIMIT_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        logger.debug(f"Saved rate limit cache: {cache_data['remaining']}/{cache_data['limit']}")
    except Exception as e:
        logger.warning(f"Failed to save rate limit cache: {e}")


def get_latest_rate_limit() -> Optional[Dict]:
    """
    Get the most recent rate limit info from API headers.

    Returns cached rate limit info from the last tweet posted, if available.
    """
    try:
        if RATE_LIMIT_CACHE_FILE.exists():
            with open(RATE_LIMIT_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # If rate limit is exhausted (remaining = 0), cache is valid until reset time
            if data.get('remaining') == 0 and data.get('reset'):
                reset_time = datetime.fromtimestamp(int(data['reset']), tz=timezone.utc)
                now = datetime.now(timezone.utc)
                if now < reset_time:
                    # Still exhausted, return cached data
                    return data
                else:
                    # Reset time has passed, cache is stale
                    return None

            # For non-zero remaining, check if cache is recent (within last hour)
            cached_time = datetime.fromisoformat(data['timestamp'])
            age_seconds = (datetime.now(timezone.utc) - cached_time).total_seconds()

            if age_seconds < 3600:  # 1 hour
                return data

    except Exception as e:
        logger.debug(f"Could not load rate limit cache: {e}")

    return None
