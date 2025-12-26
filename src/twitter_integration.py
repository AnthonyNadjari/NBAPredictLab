"""
Twitter/X Integration Module
Handles authentication, posting, media upload, and thread creation
"""

import os
import io
import time
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import plotly.io as pio
import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Logging setup
logger = logging.getLogger("twitter_integration")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# Orange color scheme constants (matching visualization theme)
ORANGE = {
    "primary": "#f97316",
    "primary_2": "#fb923c",
    "accent": "#fdba74",
    "dark": "#ea580c",
}

# Team colors mapping (fill with your team colors)
TEAM_COLORS = {
    "BOS": "#007A33", "MIL": "#00471B", "GSW": "#1D428A", "LAL": "#552583",
    "MIA": "#98002E", "PHI": "#006BB6", "BKN": "#000000", "NY": "#006BB6",
    "CHI": "#CE1141", "CLE": "#860038", "DET": "#C8102E", "IND": "#002D62",
    "ATL": "#E03A3E", "CHA": "#1D1160", "ORL": "#0077C0", "WAS": "#002B5C",
    "DEN": "#0E2240", "MIN": "#0C2340", "OKC": "#007AC1", "POR": "#E03A3E",
    "UTA": "#002B5C", "DAL": "#00538C", "HOU": "#CE1141", "MEM": "#5D76A9",
    "NO": "#0C2340", "SA": "#C4CED4", "PHX": "#1D1160", "SAC": "#5A2D81",
    "LAC": "#C8102E", "TOR": "#CE1141",
}


def load_credentials_from_env() -> Dict:
    """Load Twitter credentials from environment variables or Streamlit secrets."""
    # Load .env FIRST (before checking secrets) to ensure fresh values
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)  # Force reload .env file
    except:
        pass
    
    # Try Streamlit secrets (only if .env doesn't have values)
    try:
        import streamlit as st
        from pathlib import Path
        secrets_path = Path(".streamlit/secrets.toml")
        
        # Only use secrets if .env values are missing
        api_key_from_env = os.getenv("TW_API_KEY")
        if not api_key_from_env and secrets_path.exists() and hasattr(st, "secrets"):
            try:
                if "twitter" in st.secrets:
                    creds = st.secrets["twitter"]
                    return {
                        "api_key": creds.get("api_key"),
                        "api_key_secret": creds.get("api_key_secret"),
                        "access_token": creds.get("access_token"),
                        "access_token_secret": creds.get("access_token_secret"),
                        "bearer_token": creds.get("bearer_token"),
                        "client_id": creds.get("client_id"),
                        "client_secret": creds.get("client_secret"),
                        "redirect_uri": creds.get("redirect_uri"),
                        "use_oauth2": creds.get("use_oauth2", False),
                        "dry_run": creds.get("dry_run", True),
                    }
            except:
                pass
    except:
        pass
    return {
        "api_key": os.getenv("TW_API_KEY"),
        "api_key_secret": os.getenv("TW_API_SECRET"),
        "access_token": os.getenv("TW_ACCESS_TOKEN"),
        "access_token_secret": os.getenv("TW_ACCESS_SECRET"),
        "bearer_token": os.getenv("TW_BEARER_TOKEN"),
        "client_id": os.getenv("TW_CLIENT_ID"),
        "client_secret": os.getenv("TW_CLIENT_SECRET"),
        "redirect_uri": os.getenv("TW_REDIRECT_URI"),
        "use_oauth2": os.getenv("TW_USE_OAUTH2", "false").lower() in ("1", "true", "yes"),
        "dry_run": os.getenv("TW_DRY_RUN", "true").lower() in ("1", "true", "yes"),
    }


def create_fresh_twitter_client() -> Dict:
    """
    Create a completely fresh Twitter client, bypassing all caching.
    This ensures we're using the latest credentials from .env file or Streamlit secrets.
    
    This is the recommended method to use in Streamlit to avoid caching issues
    that can cause 403 Forbidden errors.
    
    Returns:
        Dict with 'client_v2' and 'api_v1' keys
    """
    import os
    from dotenv import load_dotenv
    
    # Force fresh load of environment variables from .env
    load_dotenv(override=True)
    
    # Try to load from Streamlit secrets as fallback
    # Suppress Streamlit secrets file warnings when not in Streamlit context
    secrets = None
    
    # Only try to access Streamlit secrets if we're actually in a Streamlit runtime
    # Check for Streamlit runtime before importing to avoid warnings
    try:
        import sys
        # Check if Streamlit is running by looking for the runtime module
        if 'streamlit.runtime' in sys.modules:
            import warnings
            # Suppress the secrets file warning
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*secrets.*')
                warnings.filterwarnings('ignore', category=UserWarning)
                try:
                    import streamlit as st
                    if hasattr(st, 'secrets') and hasattr(st, 'runtime'):
                        try:
                            # Only try to access secrets if runtime is actually initialized
                            if st.runtime.exists():
                                secrets = st.secrets.get("twitter", {})
                        except (AttributeError, RuntimeError, FileNotFoundError):
                            # Not in Streamlit runtime or secrets file doesn't exist
                            secrets = None
                except (ImportError, AttributeError):
                    secrets = None
    except Exception:
        # Any error - just use .env
        secrets = None
    
    # Load credentials from secrets (if available) or .env
    if secrets:
        api_key = secrets.get("TW_API_KEY") or os.getenv("TW_API_KEY")
        api_key_secret = secrets.get("TW_API_SECRET") or os.getenv("TW_API_SECRET")
        access_token = secrets.get("TW_ACCESS_TOKEN") or os.getenv("TW_ACCESS_TOKEN")
        # Use same naming as load_credentials_from_env() - try both variants for compatibility
        access_token_secret = (
            secrets.get("TW_ACCESS_SECRET") or 
            secrets.get("TW_ACCESS_TOKEN_SECRET") or 
            os.getenv("TW_ACCESS_SECRET") or 
            os.getenv("TW_ACCESS_TOKEN_SECRET")
        )
    else:
        # No Streamlit secrets, use .env only
        api_key = os.getenv("TW_API_KEY")
        api_key_secret = os.getenv("TW_API_SECRET")
        access_token = os.getenv("TW_ACCESS_TOKEN")
        # Use same naming as load_credentials_from_env() - try both variants
        access_token_secret = os.getenv("TW_ACCESS_SECRET") or os.getenv("TW_ACCESS_TOKEN_SECRET")
    
    # Validate all credentials are present
    missing = []
    if not api_key:
        missing.append("TW_API_KEY")
    if not api_key_secret:
        missing.append("TW_API_SECRET")
    if not access_token:
        missing.append("TW_ACCESS_TOKEN")
    if not access_token_secret:
        missing.append("TW_ACCESS_TOKEN_SECRET")
    
    if missing:
        error_msg = f"Missing required Twitter credentials: {', '.join(missing)}"
        error_msg += "\n\nPlease check:"
        error_msg += "\n1. Your .env file contains all Twitter credentials"
        error_msg += "\n2. Or configure Streamlit secrets (if using Streamlit Cloud)"
        raise ValueError(error_msg)
    
    logger.debug(f"Creating fresh Twitter client with API Key: {api_key[:10]}...")
    logger.debug(f"Creating fresh Twitter client with Access Token: {access_token[:10]}...")
    
    # Create OAuth 1.0a handler first (for media upload)
    auth = tweepy.OAuth1UserHandler(
        api_key,
        api_key_secret,
        access_token,
        access_token_secret
    )
    api_v1 = tweepy.API(auth, wait_on_rate_limit=True)
    
    # Create v2 client WITHOUT bearer token (OAuth 1.0a only)
    # This matches exactly what works in our test scripts
    # Note: wait_on_rate_limit=False makes it fail fast instead of waiting
    client_v2 = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_key_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        wait_on_rate_limit=False  # Fail fast instead of auto-waiting
    )
    
    # IMPORTANT: Do NOT call get_me() here!
    # Free tier only allows 25 requests per 24 hours for /users/me
    # Every app start, refresh, or test consumes this quota
    # Instead, we trust the credentials are valid if no error during client creation
    auth_status = {
        "verified": True,  # Assume valid - credentials were accepted during client creation
        "username": None,  # Not fetched to save rate limit
        "user_id": None,   # Not fetched to save rate limit
        "rate_limited": False,
        "error": None,
        "skip_verification": True  # Flag to indicate we skipped API verification
    }

    logger.info(f"✅ Twitter client created (verification skipped to preserve rate limits)")
    logger.info(f"💡 Free tier: only 25 /users/me calls per 24h, 17 tweets per 24h")

    return {"client_v2": client_v2, "api_v1": api_v1, "auth_status": auth_status}


def setup_twitter_api(credentials: Dict) -> Dict:
    """
    Initialize tweepy client for Twitter API.
    Returns dict with client_v2, api_v1 (for media upload), and auth_status.
    """
    api_key = credentials.get("api_key")
    api_key_secret = credentials.get("api_key_secret")
    bearer = credentials.get("bearer_token")
    access_token = credentials.get("access_token")
    access_token_secret = credentials.get("access_token_secret")
    use_oauth2 = credentials.get("use_oauth2", False)

    client_v2 = None
    api_v1 = None
    auth_status = {
        "verified": False,
        "username": None,
        "user_id": None,
        "rate_limited": False,
        "error": None
    }

    if use_oauth2 and bearer:
        # OAuth 2.0 with bearer token
        if access_token:
            client_v2 = tweepy.Client(
                bearer_token=bearer,
                consumer_key=api_key,
                consumer_secret=api_key_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                wait_on_rate_limit=True
            )
        else:
            # App-only client (limited - cannot post as user)
            client_v2 = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
    else:
        # OAuth 1.0a (more straightforward for posting)
        # This is the method that works in our tests
        logger.debug(f"Setting up OAuth1 client with: api_key={api_key[:10]}..., access_token={access_token[:10]}...")

        # IMPORTANT: Create API v1.1 client FIRST (for media upload)
        auth = tweepy.OAuth1UserHandler(
            api_key, api_key_secret, access_token, access_token_secret
        )
        api_v1 = tweepy.API(auth, wait_on_rate_limit=True)

        # Create v2 client WITHOUT bearer token (OAuth 1.0a only)
        # This matches exactly what works in our test scripts
        client_v2 = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_key_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=True
        )

        logger.debug(f"Client v2 created: {type(client_v2)}")

        # IMPORTANT: Do NOT call get_me() here!
        # Free tier only allows 25 requests per 24 hours for /users/me
        # Trust credentials are valid if client creation succeeded
        auth_status["verified"] = True
        auth_status["skip_verification"] = True
        logger.debug(f"✅ Client created (verification skipped to preserve rate limits)")

    return {"client_v2": client_v2, "api_v1": api_v1, "auth_status": auth_status}


def create_chart_image(
    fig: go.Figure, 
    filename: str, 
    width: int = 1200, 
    height: int = 675, 
    scale: int = 1
) -> str:
    """Convert Plotly figure to PNG image using Kaleido."""
    try:
        fig.write_image(filename, format="png", width=width, height=height, scale=scale)
        # Check file size (must be < 5MB for Twitter)
        file_size = Path(filename).stat().st_size / (1024 * 1024)  # MB
        if file_size > 4.5:  # Warn if close to limit
            logger.warning(f"Image {filename} is {file_size:.2f}MB (close to 5MB limit)")
        logger.info(f"Created chart image: {filename} ({file_size:.2f}MB)")
        return filename
    except Exception as e:
        logger.error(f"Failed to create chart image: {e}")
        raise


def create_composite_image(
    image_paths: List[str], 
    output_path: str, 
    rows: int = 1, 
    cols: int = None,
    tile_size: Tuple[int, int] = (1200, 675)
) -> str:
    """Compose multiple images into a single grid image."""
    images = [Image.open(p) for p in image_paths]
    n = len(images)
    
    if cols is None:
        cols = (n + rows - 1) // rows
    
    tile_w, tile_h = tile_size
    out_w = cols * tile_w
    out_h = rows * tile_h
    
    composite = Image.new("RGB", (out_w, out_h), color=(255, 255, 255))
    
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        
        # Resize while preserving aspect ratio
        img_thumb = img.copy()
        img_thumb.thumbnail((tile_w, tile_h), Image.LANCZOS)
        
        # Center in tile
        x = c * tile_w + (tile_w - img_thumb.width) // 2
        y = r * tile_h + (tile_h - img_thumb.height) // 2
        
        composite.paste(img_thumb, (x, y))
    
    composite.save(output_path, optimize=True, quality=90)
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Created composite image: {output_path} ({file_size:.2f}MB)")
    return output_path


def format_prediction_tweet(prediction: Dict, features: Dict, max_len: int = 280) -> str:
    """
    Format prediction data into Twitter-friendly text.

    Args:
        prediction: Dict with 'home_team', 'away_team', 'prediction', 'confidence', etc.
        features: Dict with feature values
        max_len: Maximum tweet length (Twitter limit is 280 characters)

    Returns:
        Formatted tweet text (guaranteed to be under max_len)
    """
    home_team = prediction.get('home_team', 'Home')
    away_team = prediction.get('away_team', 'Away')

    # Get full team names
    from nba_api.stats.static import teams
    all_teams = teams.get_teams()
    team_dict = {team['abbreviation']: team['full_name'] for team in all_teams}

    home_full = team_dict.get(home_team, home_team)
    away_full = team_dict.get(away_team, away_team)

    # Calculate predicted winner if not present
    if 'predicted_winner' in prediction:
        predicted_winner = prediction.get('predicted_winner', home_team)
    elif 'prediction' in prediction:
        # prediction['prediction'] is 'home' or 'away'
        predicted_winner = home_team if prediction.get('prediction') == 'home' else away_team
    else:
        predicted_winner = home_team  # Default fallback
    confidence = prediction.get('confidence', 0) * 100

    # Calculate odds for both teams
    home_win_prob = prediction.get('home_win_probability', 0.5)
    away_win_prob = prediction.get('away_win_probability', 0.5)

    # Calculate decimal odds (1 / probability)
    home_odds = 1 / home_win_prob if home_win_prob > 0 else 99.0
    away_odds = 1 / away_win_prob if away_win_prob > 0 else 99.0

    # Extract key metrics
    home_ortg = features.get('home_last10_offensive_rating', 0)
    away_ortg = features.get('away_last10_offensive_rating', 0)
    home_drtg = features.get('home_last10_defensive_rating', 0)
    away_drtg = features.get('away_last10_defensive_rating', 0)
    home_3pt = features.get('home_last10_fg3_pct', 0) * 100
    away_3pt = features.get('away_last10_fg3_pct', 0) * 100
    home_net = home_ortg - home_drtg
    away_net = away_ortg - away_drtg

    # Start with compact format with full names, odds, and orange-themed emojis
    text = (
        f"🏀 {away_full} ({away_team}) @ {home_full} ({home_team})\n"
        f"💰 Odds: {away_team} {away_odds:.2f} | {home_team} {home_odds:.2f}\n\n"
        f"🔥 Prediction: {predicted_winner} ({confidence:.0f}%)\n\n"
        f"📊 Last 10:\n"
        f"🟠 Off: {home_team} {home_ortg:.1f} | {away_team} {away_ortg:.1f}\n"
        f"🛡️ Def: {home_team} {home_drtg:.1f} | {away_team} {away_drtg:.1f}\n"
        f"📈 Net: {home_team} {home_net:+.1f} | {away_team} {away_net:+.1f}\n"
        f"🎯 3PT: {home_team} {home_3pt:.1f}% | {away_team} {away_3pt:.1f}%\n\n"
        f"🧡 Read thread for full analysis ⬇️"
    )

    # If still too long, use even more compact format
    if len(text) > max_len:
        text = (
            f"🏀 {away_team} @ {home_team}\n"
            f"🔥 {predicted_winner} {confidence:.0f}%\n\n"
            f"L10: Off {home_team} {home_ortg:.1f}/{away_team} {away_ortg:.1f}\n"
            f"Def {home_team} {home_drtg:.1f}/{away_team} {away_drtg:.1f}\n"
            f"Net {home_team} {home_net:+.1f}/{away_team} {away_net:+.1f}\n\n"
            f"🧡 Thread below ⬇️"
        )

    # Final safety check - truncate if still too long
    if len(text) > max_len:
        text = text[:max_len-3] + "..."

    logger.debug(f"Generated tweet text ({len(text)} chars): {text[:100]}...")
    return text


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((tweepy.TooManyRequests, tweepy.TwitterServerError))
)
def post_tweet_with_image(
    api_clients: Dict,
    text: str,
    image_path: str,
    alt_text: Optional[str] = None,
    dry_run: bool = False
) -> Dict:
    """
    Post a single tweet with an image.

    Args:
        api_clients: Dict with 'client_v2' and optionally 'api_v1'
        text: Tweet text
        image_path: Path to image file
        alt_text: Alt text for accessibility
        dry_run: If True, don't post, just log

    Returns:
        Response dict or dry_run info
    """
    if dry_run:
        logger.info(f"[DRY RUN] Would post tweet:\n{text}\nImage: {image_path}")
        return {"dry_run": True, "text": text, "image": image_path}

    # Debug: Log client object IDs to verify we're using fresh clients
    logger.info(f"🔍 Posting with client_v2 ID: {id(api_clients.get('client_v2'))}, api_v1 ID: {id(api_clients.get('api_v1'))}")

    client_v2 = api_clients.get("client_v2")
    api_v1 = api_clients.get("api_v1")
    
    if not client_v2:
        raise ValueError("Twitter API client not initialized")
    
    # Upload media
    media_id = None
    try:
        logger.info(f"📤 Attempting media upload for: {image_path}")
        if api_v1:
            # Use v1.1 media upload (more reliable)
            logger.info("Using API v1.1 for media upload...")
            media = api_v1.media_upload(filename=image_path)
            media_id = media.media_id_string
            logger.info(f"✅ Media uploaded successfully. Media ID: {media_id}")

            # Add alt text if provided
            if alt_text:
                try:
                    api_v1.create_media_metadata(media_id, alt_text)
                    logger.info(f"✅ Alt text added successfully")
                except Exception as e:
                    logger.warning(f"Could not set alt text: {e}")
        else:
            # Use v2 media upload (requires different approach)
            # For now, fallback: upload via multipart
            logger.warning("v1 API not available, media upload may fail")
            with open(image_path, "rb") as f:
                media_data = f.read()
            # Note: v2 media upload may require different method depending on tweepy version
            # This is a simplified version
            media_id = None  # Would need proper v2 media upload implementation

    except Exception as e:
        logger.error(f"❌ Media upload failed: {e}")
        raise
    
    if not media_id:
        raise RuntimeError("Media upload failed - no media_id returned")
    
    # Post tweet
    try:
        logger.info(f"📝 Attempting to post tweet (text length: {len(text)}, has_media: {bool(media_id)})...")
        if media_id:
            logger.info(f"Posting tweet with media_id: {media_id}")
            response = client_v2.create_tweet(text=text, media_ids=[media_id])
        else:
            logger.info("Posting tweet without media")
            response = client_v2.create_tweet(text=text)

        tweet_id = response.data.get('id') if hasattr(response, 'data') else response.get('data', {}).get('id')
        logger.info(f"✅ Posted tweet successfully. ID: {tweet_id}")
        return {"success": True, "tweet_id": tweet_id, "response": response}
    
    except tweepy.Forbidden as e:
        error_msg = (
            f"403 Forbidden: Your Twitter App doesn't have 'Read and write' permissions.\n"
            f"Fix: Go to https://developer.twitter.com/ → Your App → Settings → User authentication settings\n"
            f"Set 'App permissions' to 'Read and write', then regenerate Access Tokens in Keys and tokens."
        )
        logger.error(error_msg)
        # Create a more informative error by modifying the message
        original_msg = str(e)
        enhanced_msg = f"{original_msg}\n\n{error_msg}"
        raise Exception(enhanced_msg) from e
    except tweepy.TooManyRequests:
        logger.error("Rate limit exceeded")
        raise
    except Exception as e:
        logger.error(f"Failed to post tweet: {e}")
        raise


def create_twitter_thread(
    api_clients: Dict,
    texts: List[str],
    image_paths: Optional[List[str]] = None,
    dry_run: bool = False
) -> List[Dict]:
    """
    Create a Twitter thread with multiple tweets.

    Args:
        api_clients: Dict with Twitter API clients
        texts: List of tweet texts in order
        image_paths: Optional list of image paths (one per tweet or single for first)
        dry_run: If True, don't post

    Returns:
        List of response dicts
    """
    logger.info(f"🧵 Creating Twitter thread with {len(texts)} tweets (dry_run={dry_run})")

    if dry_run:
        logger.info(f"[DRY RUN] Would create thread with {len(texts)} tweets")
        for i, text in enumerate(texts):
            logger.info(f"Tweet {i+1}: {text[:100]}...")
        return [{"dry_run": True, "text": t} for t in texts]

    # Debug: Log client object IDs
    logger.info(f"🔍 Thread posting with client_v2 ID: {id(api_clients.get('client_v2'))}, api_v1 ID: {id(api_clients.get('api_v1'))}")

    client_v2 = api_clients.get("client_v2")
    api_v1 = api_clients.get("api_v1")

    if not client_v2:
        raise ValueError("Twitter API client not initialized")
    
    prev_tweet_id = None
    responses = []
    
    for i, text in enumerate(texts):
        media_ids = None
        
        # Handle images
        if image_paths:
            if isinstance(image_paths, list) and i < len(image_paths) and image_paths[i]:
                image_path = image_paths[i]
            elif isinstance(image_paths, str) and i == 0:
                image_path = image_paths
            else:
                image_path = None
            
            if image_path and api_v1:
                try:
                    media = api_v1.media_upload(filename=image_path)
                    media_ids = [media.media_id_string]
                except Exception as e:
                    logger.warning(f"Failed to upload image for tweet {i+1}: {e}")
        
        # Create tweet
        kwargs = {"text": text}
        if media_ids:
            kwargs["media_ids"] = media_ids
        if prev_tweet_id:
            kwargs["in_reply_to_tweet_id"] = prev_tweet_id
        
        try:
            # Log first tweet attempt for debugging
            if i == 0:
                logger.info(f"Attempting to post first tweet in thread (text length: {len(text)})")
            response = client_v2.create_tweet(**kwargs)
            tweet_id = response.data.get('id') if hasattr(response, 'data') else response.get('data', {}).get('id')
            prev_tweet_id = tweet_id
            responses.append({"success": True, "tweet_id": tweet_id, "response": response})
            logger.info(f"Successfully posted tweet {i+1}/{len(texts)}. ID: {tweet_id}")
            time.sleep(1)  # Small delay between thread tweets
        except tweepy.TooManyRequests as e:
            # Try to extract rate limit reset time from response headers
            reset_time_str = "in 15-30 minutes"
            try:
                if hasattr(e.response, 'headers'):
                    reset_timestamp = e.response.headers.get('x-rate-limit-reset')
                    if reset_timestamp:
                        from datetime import datetime, timezone
                        reset_time = datetime.fromtimestamp(int(reset_timestamp), tz=timezone.utc)
                        reset_local = reset_time.astimezone()
                        reset_time_str = f"at {reset_local.strftime('%H:%M:%S')}"

                        # Calculate minutes until reset
                        time_until = reset_time - datetime.now(timezone.utc)
                        minutes_until = int(time_until.total_seconds() / 60)
                        if minutes_until > 0:
                            reset_time_str = f"at {reset_local.strftime('%H:%M:%S')} ({minutes_until} minutes)"
            except:
                pass

            error_msg = f"⚠️ Twitter rate limit hit on tweet {i+1}/{len(texts)}. Rate limit resets {reset_time_str}."
            logger.error(error_msg)
            logger.error(f"Already posted {len(responses)} tweets successfully before rate limit.")
            raise Exception(error_msg) from e
        except tweepy.Forbidden as e:
            error_msg = (
                f"403 Forbidden: Your Twitter App doesn't have 'Read and write' permissions.\n"
                f"Fix: Go to https://developer.twitter.com/ → Your App → Settings → User authentication settings\n"
                f"Set 'App permissions' to 'Read and write', then regenerate Access Tokens."
            )
            logger.error(error_msg)
            # Create a more informative error by modifying the message
            original_msg = str(e)
            enhanced_msg = f"{original_msg}\n\n{error_msg}"
            raise Exception(enhanced_msg) from e
        except Exception as e:
            logger.error(f"Failed to post thread tweet {i+1}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise
    
    return responses

