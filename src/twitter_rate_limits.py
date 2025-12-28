#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter Rate Limit Checker

Properly checks the 24-hour tweet limits that matter on Free tier.
"""

import tweepy
import logging
from datetime import datetime
from typing import Dict, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache file for last known rate limit info
CACHE_FILE = Path("data/twitter_rate_limits_cache.json")


def get_24h_rate_limits_from_api(client_v2: tweepy.Client, api_v1: Optional[tweepy.API] = None) -> Optional[Dict]:
    """
    Get 24-hour rate limits using API v1.1 rate_limit_status().

    Args:
        client_v2: Tweepy Client v2 instance (for compatibility)
        api_v1: Tweepy API v1.1 instance (for rate_limit_status)

    Returns:
        Dict with 'app_24h', 'user_24h', and 'window_15min' limit info, or None
    """
    # Try API v1.1 rate_limit_status() first - most reliable
    if api_v1:
        try:
            logger.info("Fetching rate limits via API v1.1 rate_limit_status()...")
            rate_limits = api_v1.rate_limit_status()

            # Extract tweet posting limits
            tweets_limits = rate_limits.get('resources', {}).get('tweets', {})

            # For tweet creation endpoint
            create_tweet = tweets_limits.get('/2/tweets', {})

            # Also check statuses/update for v1.1 compatibility
            statuses_update = rate_limits.get('resources', {}).get('statuses', {}).get('/statuses/update', {})

            # Use v2 endpoint if available, fallback to v1.1
            tweet_endpoint = create_tweet if create_tweet else statuses_update

            if tweet_endpoint:
                limit = tweet_endpoint.get('limit', 17)
                remaining = tweet_endpoint.get('remaining', 0)
                reset_ts = tweet_endpoint.get('reset', 0)

                result = {
                    'app_24h': {
                        'limit': limit,
                        'remaining': remaining,
                        'reset': reset_ts,
                    },
                    'user_24h': {
                        'limit': limit,
                        'remaining': remaining,
                        'reset': reset_ts,
                    },
                    'window_15min': {
                        'limit': limit,
                        'remaining': remaining,
                        'reset': reset_ts,
                    },
                    'timestamp': datetime.now().isoformat(),
                    'source': 'rate_limit_status'
                }

                logger.info(f"Rate limits fetched: {remaining}/{limit} tweets remaining")
                _save_cache(result)
                return result
            else:
                logger.warning("Tweet posting endpoint not found in rate_limit_status response")

        except tweepy.TooManyRequests as e:
            logger.warning("Hit rate limit while checking rate limits (ironic!)")
            # Try to extract from error headers
            if hasattr(e.response, 'headers'):
                headers = e.response.headers
                reset_ts = int(headers.get('x-rate-limit-reset', 0))

                result = {
                    'app_24h': {
                        'limit': 17,
                        'remaining': 0,
                        'reset': reset_ts,
                    },
                    'user_24h': {
                        'limit': 17,
                        'remaining': 0,
                        'reset': reset_ts,
                    },
                    'window_15min': {
                        'limit': 17,
                        'remaining': 0,
                        'reset': reset_ts,
                    },
                    'timestamp': datetime.now().isoformat(),
                    'source': 'error_headers'
                }
                _save_cache(result)
                return result

        except Exception as e:
            logger.error(f"Error getting rate limits from API v1.1: {e}")

    # Fallback: Try to extract from any API call error
    try:
        # Try get_me() - lightweight call
        response = client_v2.get_me()
        logger.info("API call succeeded - not rate limited")
        return None  # Can't get exact numbers without an error

    except tweepy.TooManyRequests as e:
        # Extract from error headers
        logger.debug("Rate limit hit - extracting limits from error headers")

        if hasattr(e.response, 'headers'):
            headers = e.response.headers

            # Extract 24-hour limits
            app_limit = headers.get('x-app-limit-24hour-limit')
            app_remaining = headers.get('x-app-limit-24hour-remaining')
            app_reset = headers.get('x-app-limit-24hour-reset')

            user_limit = headers.get('x-user-limit-24hour-limit')
            user_remaining = headers.get('x-user-limit-24hour-remaining')
            user_reset = headers.get('x-user-limit-24hour-reset')

            # Extract 15-minute window limits
            window_limit = headers.get('x-rate-limit-limit')
            window_remaining = headers.get('x-rate-limit-remaining')
            window_reset = headers.get('x-rate-limit-reset')

            result = {
                'app_24h': {
                    'limit': int(app_limit) if app_limit else 17,
                    'remaining': int(app_remaining) if app_remaining else 0,
                    'reset': int(app_reset) if app_reset else 0,
                },
                'user_24h': {
                    'limit': int(user_limit) if user_limit else 17,
                    'remaining': int(user_remaining) if user_remaining else 0,
                    'reset': int(user_reset) if user_reset else 0,
                },
                'window_15min': {
                    'limit': int(window_limit) if window_limit else 1080000,
                    'remaining': int(window_remaining) if window_remaining else 1080000,
                    'reset': int(window_reset) if window_reset else 0,
                },
                'timestamp': datetime.now().isoformat(),
                'source': 'error_headers'
            }

            _save_cache(result)
            return result

    except Exception as e:
        logger.error(f"Error getting rate limits: {e}")
        return None


def get_cached_rate_limits() -> Optional[Dict]:
    """Load cached rate limit info from last check"""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)

            # Check if cache is recent (within last hour)
            cached_time = datetime.fromisoformat(data['timestamp'])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600

            if age_hours < 1:
                logger.debug(f"Using cached rate limits from {age_hours:.1f} hours ago")
                return data
            else:
                logger.debug(f"Cache is {age_hours:.1f} hours old, will need refresh on next check")

    except Exception as e:
        logger.debug(f"Could not load cache: {e}")

    return None


def _save_cache(data: Dict):
    """Save rate limit info to cache"""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug("Rate limit info cached")
    except Exception as e:
        logger.warning(f"Could not save cache: {e}")


def format_rate_limit_display(limits: Dict) -> Dict:
    """
    Format rate limits for display in Streamlit

    Returns dict with formatted strings for display
    """
    if not limits:
        return {
            'status': 'unknown',
            'message': 'Rate limit information not available',
        }

    app = limits.get('app_24h', {})
    user = limits.get('user_24h', {})
    window = limits.get('window_15min', {})

    # Get local timezone
    now = datetime.now()

    # Format APP 24h
    app_status = "🔴 EXHAUSTED" if app.get('remaining', 0) == 0 else "🟢 OK"
    app_reset_ts = app.get('reset')
    if app_reset_ts and app_reset_ts > 0:
        app_reset_dt = datetime.fromtimestamp(app_reset_ts)
        app_reset_str = app_reset_dt.strftime('%Y-%m-%d %H:%M:%S')
        app_hours_until = (app_reset_dt - now).total_seconds() / 3600
        if app_hours_until <= 0:
            app_hours_until = 0
            app_reset_str = "Resetting now (within the hour)"
    else:
        app_reset_dt = None
        app_reset_str = "Unknown"
        app_hours_until = 0

    # Format USER 24h
    user_status = "🔴 EXHAUSTED" if user.get('remaining', 0) == 0 else "🟢 OK"
    user_reset_ts = user.get('reset')
    if user_reset_ts and user_reset_ts > 0:
        user_reset_dt = datetime.fromtimestamp(user_reset_ts)
        user_reset_str = user_reset_dt.strftime('%Y-%m-%d %H:%M:%S')
        user_hours_until = (user_reset_dt - now).total_seconds() / 3600
        if user_hours_until <= 0:
            user_hours_until = 0
            user_reset_str = "Resetting now (within the hour)"
    else:
        user_reset_dt = None
        user_reset_str = "Unknown"
        user_hours_until = 0

    # Overall status
    can_post = app.get('remaining', 0) > 0 and user.get('remaining', 0) > 0

    # Use whichever reset time is valid (prefer user reset if app reset is missing)
    primary_reset_str = user_reset_str if (user_reset_ts and user_reset_ts > 0) else app_reset_str
    primary_hours_until = user_hours_until if (user_reset_ts and user_reset_ts > 0) else app_hours_until

    result = {
        'can_post': can_post,
        'app_24h': {
            'status': app_status,
            'limit': app.get('limit', 17),
            'remaining': app.get('remaining', 0),
            'used': app.get('limit', 17) - app.get('remaining', 0),
            'reset_time': primary_reset_str,  # Use primary reset time
            'hours_until_reset': primary_hours_until,  # Use primary hours
        },
        'user_24h': {
            'status': user_status,
            'limit': user.get('limit', 25),
            'remaining': user.get('remaining', 0),
            'used': user.get('limit', 25) - user.get('remaining', 0),
            'reset_time': user_reset_str,
            'hours_until_reset': user_hours_until,
        },
        'window_15min': {
            'limit': window.get('limit', 1080000),
            'remaining': window.get('remaining', 1080000),
        }
    }

    return result


def check_can_post_tweet(client_v2: tweepy.Client) -> tuple[bool, str]:
    """
    Check if we can post a tweet right now

    Returns:
        (can_post: bool, message: str)
    """
    # Try to get live limits
    limits = get_24h_rate_limits_from_api(client_v2)

    # Fallback to cache if live check failed
    if not limits:
        limits = get_cached_rate_limits()

    if not limits:
        # No data available - assume we can try
        return True, "Rate limit status unknown - proceed with caution"

    app = limits.get('app_24h', {})
    user = limits.get('user_24h', {})

    app_remaining = app.get('remaining', 0)
    user_remaining = user.get('remaining', 0)

    if app_remaining == 0:
        reset_dt = datetime.fromtimestamp(app['reset'])
        hours = (reset_dt - datetime.now()).total_seconds() / 3600
        return False, f"APP 24-hour limit exhausted. Resets in {hours:.1f} hours at {reset_dt.strftime('%H:%M:%S')}"

    if user_remaining == 0:
        reset_dt = datetime.fromtimestamp(user['reset'])
        hours = (reset_dt - datetime.now()).total_seconds() / 3600
        return False, f"USER 24-hour limit exhausted. Resets in {hours:.1f} hours at {reset_dt.strftime('%H:%M:%S')}"

    return True, f"Can post - {min(app_remaining, user_remaining)} tweets remaining in 24h window"
