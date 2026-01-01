#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter Rate Limit Checker

Properly checks the 24-hour tweet limits that matter on Free tier.
"""

import tweepy
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache file for last known rate limit info
CACHE_FILE = Path("data/twitter_rate_limits_cache.json")

# Free tier daily tweet limit
FREE_TIER_24H_LIMIT = 17


def get_24h_rate_limits_from_api(client_v2: tweepy.Client, api_v1: Optional[tweepy.API] = None) -> Optional[Dict]:
    """
    Get 24-hour rate limits using API v1.1 rate_limit_status().

    Args:
        client_v2: Tweepy Client v2 instance (for compatibility)
        api_v1: Tweepy API v1.1 instance (for rate_limit_status)

    Returns:
        Dict with 'app_24h', 'user_24h', and 'window_15min' limit info, or None
    """
    # CRITICAL: Test confirmed that rate_limit_status() does NOT show 24-hour limits
    # - No endpoint with limit=17 (the 24-hour limit for free tier)
    # - All endpoints show 15-minute window limits (75, 900, 180, etc.)
    # - The 24-hour limit (17 tweets) is ONLY visible in 429 error response headers
    
    # Try to get cached 24-hour limit data from previous 429 errors
    cached_data = get_cached_rate_limits()
    if cached_data and cached_data.get('source') in ['error_headers', 'error_response']:
        # We have valid 24h limit data from a previous 429 error
        logger.info("✅ Using cached 24-hour limit data from previous 429 error")
        logger.info(f"   24h limit: {cached_data.get('app_24h', {}).get('remaining', 0)}/{cached_data.get('app_24h', {}).get('limit', 17)} tweets")
        return cached_data
    
    # No cached 24h data available - rate_limit_status() won't help us
    logger.warning("⚠️ Cannot get 24-hour limit. rate_limit_status() does NOT show 24-hour limits.")
    logger.warning("   The 24-hour limit (17 tweets) is only visible in 429 error response headers.")
    logger.warning("   Need to hit rate limit to see 24h limit data.")
    return None
    
    # OLD CODE BELOW - rate_limit_status() doesn't show tweet posting endpoints or 24h limits
    # Keeping for reference but it's not useful for 24-hour limits
    if False and api_v1:
        try:
            logger.info("Fetching rate limits via API v1.1 rate_limit_status()...")
            rate_limits = api_v1.rate_limit_status()
            
            # Get all resources - this shows ALL rate limits
            resources = rate_limits.get('resources', {})
            
            # Check for tweet posting endpoints
            tweets_resources = resources.get('tweets', {})
            statuses_resources = resources.get('statuses', {})
            
            # Try API v2 endpoint: /2/tweets
            tweet_v2_endpoint = tweets_resources.get('/2/tweets', {})
            
            # Try v1.1 endpoint: /statuses/update
            statuses_update = statuses_resources.get('/statuses/update', {})
            
            # Also check /statuses/update_with_media
            statuses_update_media = statuses_resources.get('/statuses/update_with_media', {})
            
            # Use v2 endpoint if available, otherwise v1.1
            tweet_endpoint = tweet_v2_endpoint if tweet_v2_endpoint else (statuses_update if statuses_update else statuses_update_media)

            if tweet_endpoint:
                # Extract the limit data from the endpoint
                endpoint_limit = tweet_endpoint.get('limit', 0)
                endpoint_remaining = tweet_endpoint.get('remaining', 0)
                endpoint_reset_ts = tweet_endpoint.get('reset', 0)
                
                logger.info(f"Endpoint data: {endpoint_remaining}/{endpoint_limit} remaining")
                
                # CRITICAL: rate_limit_status() ONLY returns 15-minute window limits, NOT 24-hour limits
                # The 24-hour limit (17 tweets) is ONLY visible in 429 error response headers
                # So we should NOT use this data for app_24h/user_24h - it's 15-minute window data
                window_limit = endpoint_limit
                window_remaining = endpoint_remaining
                window_reset_ts = endpoint_reset_ts
                
                # For 24-hour limits, we MUST use cached data from previous 429 errors
                # or return None if we don't have it (don't show wrong 15-min data as 24h limit)
                cached_data = get_cached_rate_limits()
                if cached_data and cached_data.get('source') in ['error_headers', 'error_response']:
                    # This is real 24h limit data from a 429 error
                    app_24h_limit = cached_data.get('app_24h', {}).get('limit', 17)
                    app_24h_remaining = cached_data.get('app_24h', {}).get('remaining', 0)
                    app_24h_reset = cached_data.get('app_24h', {}).get('reset', 0)
                    logger.info(f"✅ Using cached 24h limit from error headers: {app_24h_remaining}/{app_24h_limit} tweets")
                else:
                    # No valid 24h data - return None so we don't show wrong 15-min data as 24h limit
                    logger.warning("⚠️ rate_limit_status() only shows 15-minute window limits, not 24-hour limits")
                    logger.warning("   The 24-hour limit (17 tweets) is only visible in 429 error response headers")
                    logger.warning("   Returning None to avoid showing incorrect 15-min data as 24h limit")
                    return None  # Don't return wrong data

                result = {
                    'app_24h': {
                        'limit': app_24h_limit,
                        'remaining': app_24h_remaining,
                        'reset': app_24h_reset,
                    },
                    'user_24h': {
                        'limit': app_24h_limit,  # Same for free tier
                        'remaining': app_24h_remaining,
                        'reset': app_24h_reset,
                    },
                    'window_15min': {
                        'limit': window_limit,
                        'remaining': window_remaining,
                        'reset': window_reset_ts,
                    },
                    'timestamp': datetime.now().isoformat(),
                    'source': 'rate_limit_status_with_cached_24h'
                }

                _save_cache(result)
                logger.info(f"✅ Final rate limits: 24h: {app_24h_remaining}/{app_24h_limit}, 15-min: {window_remaining}/{window_limit}")
                return result
            else:
                # Log what we actually got for debugging
                logger.warning(f"⚠️ Tweet posting endpoint not found. Available statuses endpoints: {list(statuses.keys())}")
                logger.warning(f"   Full resources structure: {list(resources.keys())}")
                # Try to return something useful anyway - check if there's any status endpoint
                if statuses:
                    # Use first available status endpoint as fallback
                    first_endpoint = list(statuses.values())[0]
                    if isinstance(first_endpoint, dict) and 'limit' in first_endpoint:
                        limit = first_endpoint.get('limit', 17)
                        remaining = first_endpoint.get('remaining', 0)
                        reset_ts = first_endpoint.get('reset', 0)
                        logger.info(f"Using fallback endpoint: {remaining}/{limit} remaining")
                        result = {
                            'app_24h': {'limit': limit, 'remaining': remaining, 'reset': reset_ts},
                            'user_24h': {'limit': limit, 'remaining': remaining, 'reset': reset_ts},
                            'window_15min': {'limit': limit, 'remaining': remaining, 'reset': reset_ts},
                            'timestamp': datetime.now().isoformat(),
                            'source': 'rate_limit_status_fallback'
                        }
                        _save_cache(result)
                        return result

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
            logger.error(f"   This might be because API v1.1 is not available or credentials don't have v1.1 access")
            # Don't return None here - continue to fallback methods

    # Fallback: Try to extract from any API call error
    # Note: API v2 doesn't expose 24h limits in successful responses
    # We can only see them when we hit a rate limit (429 error)
    try:
        # Try get_me() - lightweight call to see if we're rate limited
        response = client_v2.get_me()
        logger.info("API v2 call succeeded - not currently rate limited")
        logger.info("⚠️ Twitter API v2 doesn't expose 24h rate limits in successful responses")
        logger.info("   Limits are only visible when you hit the limit (429 error) or via API v1.1 rate_limit_status()")
        # Return None - we can't get exact numbers without hitting a limit or using v1.1
        return None

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

            # CRITICAL: Extract 24-hour limits from error headers
            # Test confirmed these headers exist: x-app-limit-24hour-*, x-user-limit-24hour-*
            # These are the REAL limits (17 tweets for Free tier)
            app_24h_limit = int(app_limit) if app_limit else FREE_TIER_24H_LIMIT
            app_24h_remaining = int(app_remaining) if app_remaining else 0
            app_24h_reset = int(app_reset) if app_reset else 0
            
            user_24h_limit = int(user_limit) if user_limit else FREE_TIER_24H_LIMIT
            user_24h_remaining = int(user_remaining) if user_remaining else 0
            user_24h_reset = int(user_reset) if user_reset else 0
            
            # 15-minute window limits (different from 24h limits - not what we display)
            window_15min_limit = int(window_limit) if window_limit else 300
            window_15min_remaining = int(window_remaining) if window_remaining else 0
            window_15min_reset = int(window_reset) if window_reset else 0

            result = {
                'app_24h': {
                    'limit': app_24h_limit,
                    'remaining': app_24h_remaining,
                    'reset': app_24h_reset,
                },
                'user_24h': {
                    'limit': user_24h_limit,
                    'remaining': user_24h_remaining,
                    'reset': user_24h_reset,
                },
                'window_15min': {
                    'limit': window_15min_limit,
                    'remaining': window_15min_remaining,
                    'reset': window_15min_reset,
                },
                'timestamp': datetime.now().isoformat(),
                'source': 'error_headers'
            }

            reset_time = datetime.fromtimestamp(app_24h_reset, tz=timezone.utc) if app_24h_reset > 0 else None
            logger.info(f"✅ Extracted 24h limits from 429 error headers:")
            logger.info(f"   APP 24h: {app_24h_remaining}/{app_24h_limit} tweets remaining")
            logger.info(f"   USER 24h: {user_24h_remaining}/{user_24h_limit} tweets remaining")
            if reset_time:
                hours_until = (reset_time - datetime.now(timezone.utc)).total_seconds() / 3600
                logger.info(f"   24h reset time: {reset_time} UTC ({hours_until:.1f} hours)")
            _save_cache(result)
            return result

    except Exception as e:
        logger.error(f"Error getting rate limits: {e}")
        return None


def count_tweets_last_24h(client_v2: tweepy.Client, api_v1: Optional[tweepy.API] = None) -> Dict:
    """
    Count how many tweets the authenticated user posted in the last 24 hours.

    Uses API v2 get_users_tweets() or API v1.1 user_timeline() to fetch recent tweets
    and count those posted within the last 24 hours.

    Args:
        client_v2: Tweepy Client v2 instance
        api_v1: Optional Tweepy API v1.1 instance (fallback)

    Returns:
        Dict with 'count', 'limit', 'remaining', and 'tweets' (list of recent tweets with timestamps)
    """
    try:
        # Calculate cutoff time (24 hours ago)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        logger.info(f"📊 Counting tweets posted since: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        tweets_in_24h = []

        # Try to get authenticated user's ID first
        try:
            # Get authenticated user info (uses /users/me endpoint)
            me = client_v2.get_me()
            if me and me.data:
                user_id = me.data.id
                username = me.data.username
                logger.info(f"✅ Authenticated as: @{username} (ID: {user_id})")
            else:
                logger.warning("⚠️ Could not get authenticated user info")
                return {
                    'count': 0,
                    'limit': FREE_TIER_24H_LIMIT,
                    'remaining': FREE_TIER_24H_LIMIT,
                    'tweets': [],
                    'error': 'Could not get user info'
                }
        except tweepy.TooManyRequests:
            logger.warning("⚠️ Rate limited on /users/me endpoint (25 calls per 24h limit)")
            # Try to use cached user_id if available
            cached_limits = get_cached_rate_limits()
            if cached_limits and 'user_id' in cached_limits:
                user_id = cached_limits['user_id']
                logger.info(f"Using cached user_id: {user_id}")
            else:
                return {
                    'count': 0,
                    'limit': FREE_TIER_24H_LIMIT,
                    'remaining': FREE_TIER_24H_LIMIT,
                    'tweets': [],
                    'error': 'Rate limited on /users/me'
                }
        except Exception as e:
            logger.error(f"Error getting authenticated user: {e}")
            return {
                'count': 0,
                'limit': FREE_TIER_24H_LIMIT,
                'remaining': FREE_TIER_24H_LIMIT,
                'tweets': [],
                'error': str(e)
            }

        # Fetch user's recent tweets using API v2
        try:
            logger.info(f"📥 Fetching recent tweets for user {user_id}...")

            # Get tweets from last 24 hours
            # max_results: 5-100 (default 10), we'll use 100 to get as many as possible
            response = client_v2.get_users_tweets(
                id=user_id,
                max_results=100,  # Max allowed per request
                tweet_fields=['created_at', 'id', 'text'],
                exclude=['retweets', 'replies']  # Only count original tweets
            )

            if response and response.data:
                logger.info(f"✅ Retrieved {len(response.data)} recent tweets")

                # Filter tweets from last 24 hours
                for tweet in response.data:
                    # tweet.created_at is timezone-aware datetime
                    if tweet.created_at >= cutoff_time:
                        tweets_in_24h.append({
                            'id': tweet.id,
                            'text': tweet.text[:50] + '...' if len(tweet.text) > 50 else tweet.text,
                            'created_at': tweet.created_at.isoformat()
                        })

                logger.info(f"📊 Found {len(tweets_in_24h)} tweets in last 24 hours")
            else:
                logger.info("No recent tweets found")

        except tweepy.TooManyRequests:
            logger.warning("⚠️ Rate limited on get_users_tweets endpoint")
            # Fallback: use cached count if available
            cached_limits = get_cached_rate_limits()
            if cached_limits and 'tweet_count_24h' in cached_limits:
                cached_count = cached_limits['tweet_count_24h']
                logger.info(f"Using cached tweet count: {cached_count}")
                return {
                    'count': cached_count,
                    'limit': FREE_TIER_24H_LIMIT,
                    'remaining': FREE_TIER_24H_LIMIT - cached_count,
                    'tweets': [],
                    'cached': True
                }
            # If no cache, return error
            return {
                'count': 0,
                'limit': FREE_TIER_24H_LIMIT,
                'remaining': FREE_TIER_24H_LIMIT,
                'tweets': [],
                'error': 'Rate limited on tweet fetch'
            }
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
            # Try API v1.1 fallback
            if api_v1:
                try:
                    logger.info("Trying API v1.1 user_timeline fallback...")
                    timeline = api_v1.user_timeline(count=100, exclude_replies=True, include_rts=False)

                    for status in timeline:
                        # status.created_at is timezone-aware datetime
                        if status.created_at.replace(tzinfo=timezone.utc) >= cutoff_time:
                            tweets_in_24h.append({
                                'id': status.id_str,
                                'text': status.text[:50] + '...' if len(status.text) > 50 else status.text,
                                'created_at': status.created_at.isoformat()
                            })

                    logger.info(f"📊 Found {len(tweets_in_24h)} tweets via v1.1 API")
                except Exception as v1_error:
                    logger.error(f"API v1.1 fallback also failed: {v1_error}")
                    return {
                        'count': 0,
                        'limit': FREE_TIER_24H_LIMIT,
                        'remaining': FREE_TIER_24H_LIMIT,
                        'tweets': [],
                        'error': f'Both v2 and v1.1 failed: {str(e)}, {str(v1_error)}'
                    }

        # Calculate remaining tweets
        count = len(tweets_in_24h)
        remaining = max(0, FREE_TIER_24H_LIMIT - count)

        # Cache the count (with user_id for future requests)
        _save_cache({
            'tweet_count_24h': count,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'source': 'tweet_count'
        })

        logger.info(f"✅ Tweet count: {count}/{FREE_TIER_24H_LIMIT} used, {remaining} remaining")

        return {
            'count': count,
            'limit': FREE_TIER_24H_LIMIT,
            'remaining': remaining,
            'tweets': tweets_in_24h
        }

    except Exception as e:
        logger.error(f"Unexpected error in count_tweets_last_24h: {e}")
        return {
            'count': 0,
            'limit': FREE_TIER_24H_LIMIT,
            'remaining': FREE_TIER_24H_LIMIT,
            'tweets': [],
            'error': str(e)
        }


def get_cached_rate_limits() -> Optional[Dict]:
    """Load cached rate limit info from last check"""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)

            # For 24-hour limits, cache can be up to 24 hours old (since that's the reset period)
            # The 24h limit data is only available when we hit a rate limit (429 error)
            cached_time = datetime.fromisoformat(data['timestamp'])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600

            # Accept cache if it's less than 24 hours old (since 24h limits reset every 24h)
            if age_hours < 24:
                logger.debug(f"Using cached 24h rate limits from {age_hours:.1f} hours ago")
                return data
            else:
                logger.debug(f"Cache is {age_hours:.1f} hours old (older than 24h reset window)")

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
