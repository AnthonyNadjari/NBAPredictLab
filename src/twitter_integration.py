"""
Twitter/X Integration Module
Handles authentication, posting, media upload, and thread creation
"""

import os
import io
import time
import logging
from datetime import datetime, timezone
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
    from pathlib import Path
    
    # Read .env file DIRECTLY to avoid Python environment variable caching
    # This is critical - os.getenv() may return cached values even after load_dotenv()
    env_file = Path(".env")
    env_vars = {}
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    env_vars[key.strip()] = value
    
    # Force fresh load of environment variables from .env (as backup)
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
    
    # Load credentials from secrets (if available) or .env file directly (to avoid caching)
    if secrets:
        # Prefer secrets, but fall back to .env file (not os.getenv to avoid caching)
        api_key = secrets.get("TW_API_KEY") or env_vars.get("TW_API_KEY") or os.getenv("TW_API_KEY")
        api_key_secret = secrets.get("TW_API_SECRET") or env_vars.get("TW_API_SECRET") or os.getenv("TW_API_SECRET")
        access_token = secrets.get("TW_ACCESS_TOKEN") or env_vars.get("TW_ACCESS_TOKEN") or os.getenv("TW_ACCESS_TOKEN")
        # Use same naming as load_credentials_from_env() - try both variants for compatibility
        access_token_secret = (
            secrets.get("TW_ACCESS_SECRET") or 
            secrets.get("TW_ACCESS_TOKEN_SECRET") or
            env_vars.get("TW_ACCESS_SECRET") or 
            env_vars.get("TW_ACCESS_TOKEN_SECRET") or
            os.getenv("TW_ACCESS_SECRET") or 
            os.getenv("TW_ACCESS_TOKEN_SECRET")
        )
    else:
        # No Streamlit secrets, use .env file directly (not os.getenv to avoid caching)
        api_key = env_vars.get("TW_API_KEY") or os.getenv("TW_API_KEY")
        api_key_secret = env_vars.get("TW_API_SECRET") or os.getenv("TW_API_SECRET")
        access_token = env_vars.get("TW_ACCESS_TOKEN") or os.getenv("TW_ACCESS_TOKEN")
        # Use same naming as load_credentials_from_env() - try both variants
        access_token_secret = (
            env_vars.get("TW_ACCESS_SECRET") or 
            env_vars.get("TW_ACCESS_TOKEN_SECRET") or
            os.getenv("TW_ACCESS_SECRET") or 
            os.getenv("TW_ACCESS_TOKEN_SECRET")
        )
    
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
    
    # Log credentials being used (first/last few chars for debugging)
    logger.info(f"🔑 Creating Twitter client with:")
    logger.info(f"   API Key: {api_key[:15]}...{api_key[-5:] if len(api_key) > 20 else ''} (full length: {len(api_key)})")
    logger.info(f"   Access Token: {access_token[:30]}...{access_token[-10:] if len(access_token) > 40 else ''} (full length: {len(access_token)})")
    logger.info(f"   Access Token Secret: {access_token_secret[:15]}...{access_token_secret[-5:] if len(access_token_secret) > 20 else ''} (full length: {len(access_token_secret)})")
    
    # Validate credentials are not empty
    if not api_key or not api_key_secret or not access_token or not access_token_secret:
        raise ValueError(f"Missing credentials: api_key={bool(api_key)}, api_key_secret={bool(api_key_secret)}, access_token={bool(access_token)}, access_token_secret={bool(access_token_secret)}")
    
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
    
    # Store credentials in client for later verification (for debugging)
    try:
        client_v2._debug_api_key = api_key
        client_v2._debug_access_token = access_token
    except:
        pass  # If we can't store debug info, that's fine
    
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

    # Get full team names and abbreviations
    from nba_api.stats.static import teams
    all_teams = teams.get_teams()
    team_dict = {team['abbreviation']: team['full_name'] for team in all_teams}
    # Reverse mapping: full_name -> abbreviation
    abbrev_dict = {team['full_name']: team['abbreviation'] for team in all_teams}

    # Get full names (if input is abbreviation, get full name; if already full name, use as-is)
    home_full = team_dict.get(home_team, home_team) if len(home_team) <= 3 else home_team
    away_full = team_dict.get(away_team, away_team) if len(away_team) <= 3 else away_team
    
    # Get abbreviations (if input is full name, get abbreviation; if already abbreviation, use as-is)
    home_abbrev = abbrev_dict.get(home_team, home_team) if len(home_team) > 3 else home_team
    away_abbrev = abbrev_dict.get(away_team, away_team) if len(away_team) > 3 else away_team

    # Calculate predicted winner if not present
    if 'predicted_winner' in prediction:
        predicted_winner_raw = prediction.get('predicted_winner', home_team)
        # Convert to full name if it's an abbreviation
        predicted_winner = team_dict.get(predicted_winner_raw, predicted_winner_raw) if len(predicted_winner_raw) <= 3 else predicted_winner_raw
    elif 'prediction' in prediction:
        # prediction['prediction'] is 'home' or 'away'
        predicted_winner = home_full if prediction.get('prediction') == 'home' else away_full
    else:
        predicted_winner = home_full  # Default fallback
    confidence = prediction.get('confidence', 0) * 100

    # Get prediction quality indicator (from calibrated model)
    prediction_quality = prediction.get('prediction_quality', 'medium')
    should_predict = prediction.get('should_predict', True)

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

    # Add pattern adjustments note if present (make it CLEAR what it means)
    adjustment_note = ""
    if 'pattern_adjustments' in prediction and prediction['pattern_adjustments']:
        # Get first/most important adjustment
        first_adj = prediction['pattern_adjustments'][0]
        if "Hot road" in first_adj:
            adjustment_note = "\n⚡ AI Override: Away team on 4+ win streak - boosted +10%"
        elif "Cold home" in first_adj:
            adjustment_note = "\n⚠️ AI Override: Home team on 3+ loss streak - reduced -8%"
        elif "Heavy travel" in first_adj:
            adjustment_note = "\n🛫 AI Override: Cross-country B2B travel - away team penalized -15%"
        elif "Large ELO" in first_adj:
            adjustment_note = "\n📊 AI Override: Big favorite (model was 50% wrong historically) - reduced -5%"
        elif "B2B" in first_adj:
            adjustment_note = "\n😴 AI Override: Home team on back-to-back - fatigue penalty -6%"

    # Determine which template to use based on game dynamics
    home_rest = features.get('home_rest_days', 1)
    away_rest = features.get('away_rest_days', 1)
    rest_diff = abs(home_rest - away_rest)
    def_diff = abs(home_drtg - away_drtg)

    # Get win streaks if available
    home_streak = features.get('home_last3_win_pct', 0.5) * 100
    away_streak = features.get('away_last3_win_pct', 0.5) * 100

    # Select template using hash-based rotation (6 templates)
    import hashlib
    template_seed = int(hashlib.md5(f"{home_team}{away_team}{datetime.now().strftime('%Y%m%d')}".encode()).hexdigest()[:8], 16)
    template_variant = template_seed % 6

    # Add quality indicator emoji
    quality_indicator = ""
    if prediction_quality == "high":
        quality_indicator = "🎯"  # High confidence signal
    elif prediction_quality == "low" or not should_predict:
        quality_indicator = "⚠️"  # Caution - close game

    # Standard header (always included) - format: "Full Name (ABBREV) @ Full Name (ABBREV)"
    header = (
        f"🏀 {away_full} ({away_abbrev}) @ {home_full} ({home_abbrev})\n"
        f"💰 Odds: {away_abbrev} {away_odds:.2f} | {home_abbrev} {home_odds:.2f}\n\n"
        f"🔥 Prediction: {predicted_winner} ({confidence:.0f}%) {quality_indicator}\n"
    )

    # Determine if prediction is for home or away team
    is_home_pick = prediction.get('prediction') == 'home'
    predicted_team_abbrev = home_abbrev if is_home_pick else away_abbrev

    # Calculate which team has the advantage in each metric
    rested_team = home_team if home_rest > away_rest else away_team
    rested_team_abbrev = home_abbrev if home_rest > away_rest else away_abbrev
    tired_team = away_team if home_rest > away_rest else home_team
    better_def = home_team if home_drtg < away_drtg else away_team
    better_def_abbrev = home_abbrev if home_drtg < away_drtg else away_abbrev
    hotter_team = home_team if home_streak > away_streak else away_team
    hotter_team_abbrev = home_abbrev if home_streak > away_streak else away_abbrev
    hotter_streak = max(home_streak, away_streak)

    # CONSISTENCY CHECK: Only use hooks where the advantage supports the predicted winner
    # This prevents confusing tweets like "Model picks Lakers" but "Heat on fire"
    rest_supports_pick = (rested_team_abbrev == predicted_team_abbrev)
    defense_supports_pick = (better_def_abbrev == predicted_team_abbrev)
    streak_supports_pick = (hotter_team_abbrev == predicted_team_abbrev)

    # Template variations (hook + CTA) - ONLY use when metric supports prediction
    hook = None  # Will use default if no consistent hook found

    if rest_diff >= 2 and template_variant in [0, 1] and rest_supports_pick:
        # Rest advantage templates - ONLY if rested team = predicted winner
        if template_variant == 0:
            hook = f"\n⚡ {rested_team} rested vs {tired_team} on B2B\nRest = 5pt edge"
        else:
            hook = f"\n😴 Fatigue factor: {tired_team} on short rest\nFresh legs win"
    elif def_diff >= 8 and template_variant in [2, 3] and defense_supports_pick:
        # Defense mismatch templates - ONLY if better defense = predicted winner
        if template_variant == 2:
            hook = f"\n🛡️ {better_def}'s elite D ({min(home_drtg, away_drtg):.0f} DRtg)\nDefense travels"
        else:
            hook = f"\n🔒 Defense wins: {better_def} locks down\n{abs(def_diff):.0f}pt edge"
    elif confidence >= 70 and template_variant == 4:
        # High confidence template - always safe (doesn't mention specific advantage)
        hook = f"\n🎯 Model loves this spot\n{confidence:.0f}% = strong signal"
    elif hotter_streak >= 75 and template_variant == 5 and streak_supports_pick:
        # Hot team template - ONLY if hot team = predicted winner
        hook = f"\n🔥 {hotter_team} scorching ({hotter_streak:.0f}% L3)\nMomentum is real"

    # Default fallback templates (always consistent - reference predicted winner)
    if hook is None:
        if template_variant % 2 == 0:
            hook = f"\n📊 Edge: {predicted_winner} checks boxes\nNet rating advantage"
        else:
            hook = f"\n💡 Sharp play: {predicted_winner}\nModel sees value"

    # CTA variations
    cta_options = ["Thread ⬇️", "Full edge ⬇️", "Why this hits ⬇️", "Breakdown ⬇️", "Analysis ⬇️", "Read on ⬇️"]
    cta = f"\n\n{cta_options[template_variant]}"

    text = header + hook + cta

    # Verify it fits (should be ~220-260 chars)
    actual_len = len(text)
    if actual_len > 270:
        # Ultra-compact fallback
        text = (
            f"🏀 {away_full} ({away_abbrev}) @ {home_full} ({home_abbrev})\n"
            f"💰 Odds: {away_abbrev} {away_odds:.2f} | {home_abbrev} {home_odds:.2f}\n\n"
            f"🔥 Prediction: {predicted_winner} ({confidence:.0f}%)\n\n"
            f"Thread ⬇️"
        )
        actual_len = len(text)

    logger.debug(f"Generated tweet text ({actual_len} chars): {text[:100]}...")
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

        # Extract rate limit headers from response
        rate_limit_info = None
        try:
            # Access the underlying response object to get headers
            if hasattr(response, 'headers'):
                headers = response.headers
            elif hasattr(response, 'response') and hasattr(response.response, 'headers'):
                headers = response.response.headers
            else:
                headers = {}

            if headers:
                rate_limit_info = {
                    'remaining': headers.get('x-rate-limit-remaining'),
                    'limit': headers.get('x-rate-limit-limit'),
                    'reset': headers.get('x-rate-limit-reset'),
                }
                logger.info(f"📊 Rate limit from headers: {rate_limit_info['remaining']}/{rate_limit_info['limit']} remaining")
        except Exception as e:
            logger.debug(f"Could not extract rate limit headers: {e}")

        # Log tweet to local counter (with rate limit info if available)
        try:
            from src.tweet_counter import log_tweet_posted
            log_tweet_posted(str(tweet_id), text[:50], rate_limit_info)
        except Exception as e:
            logger.warning(f"Could not log tweet to local counter: {e}")

        return {"success": True, "tweet_id": tweet_id, "response": response, "rate_limit_info": rate_limit_info}
    
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

    # CRITICAL: Create a FRESH client right before posting to ensure we use latest credentials
    # This bypasses any potential caching issues
    logger.info("🔄 Creating fresh Twitter client for posting (to ensure latest credentials)...")
    try:
        fresh_clients = create_fresh_twitter_client()
        client_v2 = fresh_clients.get("client_v2")
        api_v1 = fresh_clients.get("api_v1")
        logger.info("✅ Fresh client created successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create fresh client, using provided client: {e}")
        # Fallback to provided client if fresh creation fails
        client_v2 = api_clients.get("client_v2")
        api_v1 = api_clients.get("api_v1")

    if not client_v2:
        raise ValueError("Twitter API client not initialized")
    
    # Debug: Try to get client credentials to verify what's being used
    try:
        # Check if client has credentials attribute (for debugging)
        if hasattr(client_v2, '_client') and hasattr(client_v2._client, 'auth'):
            auth = client_v2._client.auth
            if hasattr(auth, 'access_token'):
                logger.info(f"🔑 Client access token (first 30 chars): {auth.access_token[:30]}...")
            if hasattr(auth, 'consumer_key'):
                logger.info(f"🔑 Client API key (first 15 chars): {auth.consumer_key[:15]}...")
    except Exception as debug_e:
        logger.debug(f"Could not extract client credentials for debugging: {debug_e}")
    
    prev_tweet_id = None
    responses = []
    
    for i, text in enumerate(texts):
        media_ids = None
        
        # Validate text length (format should already be designed to fit)
        text = text.strip()
        text_len = len(text)
        
        # Log character count for monitoring
        logger.info(f"📝 Tweet {i+1}/{len(texts)}: {text_len} characters")
        
        # Warn if somehow over limit (shouldn't happen with new format)
        if text_len > 280:
            logger.error(f"❌ Tweet {i+1} is {text_len} chars (over 280 limit)! Format needs adjustment.")
            raise ValueError(f"Tweet {i+1} exceeds 280 character limit ({text_len} chars). Format must be redesigned to fit.")
        
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
                # Debug: Verify credentials in client match what we expect
                try:
                    if hasattr(client_v2, '_debug_api_key'):
                        logger.info(f"🔍 Debug: Client API Key: {client_v2._debug_api_key[:15]}...")
                        logger.info(f"🔍 Debug: Client Access Token: {client_v2._debug_access_token[:30]}...")
                except:
                    pass
            response = client_v2.create_tweet(**kwargs)
            tweet_id = response.data.get('id') if hasattr(response, 'data') else response.get('data', {}).get('id')
            prev_tweet_id = tweet_id
            responses.append({"success": True, "tweet_id": tweet_id, "response": response})
            logger.info(f"Successfully posted tweet {i+1}/{len(texts)}. ID: {tweet_id}")

            # Extract rate limit headers from response
            rate_limit_info = None
            try:
                # Access the underlying response object to get headers
                if hasattr(response, 'headers'):
                    headers = response.headers
                elif hasattr(response, 'response') and hasattr(response.response, 'headers'):
                    headers = response.response.headers
                else:
                    headers = {}

                if headers:
                    rate_limit_info = {
                        'remaining': headers.get('x-rate-limit-remaining'),
                        'limit': headers.get('x-rate-limit-limit'),
                        'reset': headers.get('x-rate-limit-reset'),
                    }
                    logger.info(f"📊 Rate limit from headers: {rate_limit_info['remaining']}/{rate_limit_info['limit']} remaining")
            except Exception as e:
                logger.debug(f"Could not extract rate limit headers: {e}")

            # Log tweet to local counter (with rate limit info if available)
            try:
                from src.tweet_counter import log_tweet_posted
                log_tweet_posted(str(tweet_id), text[:50], rate_limit_info)
            except Exception as e:
                logger.warning(f"Could not log tweet to local counter: {e}")
            
            # Try to extract rate limit headers from response if available
            # Note: Twitter API v2 only shows 24h limits in 429 errors, not successful responses
            # But we can try to access response metadata if Tweepy exposes it
            try:
                if hasattr(response, 'response') and hasattr(response.response, 'headers'):
                    headers = response.response.headers
                    # Log if we find any rate limit info (unlikely for 24h limits in success)
                    if 'x-rate-limit-remaining' in headers:
                        logger.debug(f"15-min window remaining: {headers.get('x-rate-limit-remaining')}")
            except:
                pass  # Headers not accessible, which is expected
            
            time.sleep(1)  # Small delay between thread tweets
        except tweepy.TooManyRequests as e:
            # Extract and cache rate limit info from headers
            reset_time_str = "soon"
            try:
                if hasattr(e.response, 'headers'):
                    headers = e.response.headers

                    # Extract 24-hour limits (the real constraint)
                    app_limit = headers.get('x-app-limit-24hour-limit')
                    app_remaining = headers.get('x-app-limit-24hour-remaining')
                    app_reset = headers.get('x-app-limit-24hour-reset')

                    user_limit = headers.get('x-user-limit-24hour-limit')
                    user_remaining = headers.get('x-user-limit-24hour-remaining')
                    user_reset = headers.get('x-user-limit-24hour-reset')

                    window_limit = headers.get('x-rate-limit-limit')
                    window_remaining = headers.get('x-rate-limit-remaining')
                    window_reset = headers.get('x-rate-limit-reset')

                    # Cache this data for the status tab
                    import json
                    from datetime import datetime, timezone
                    from pathlib import Path

                    cache_data = {
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
                        'source': 'error_response',
                    }

                    cache_file = Path("data/twitter_rate_limits_cache.json")
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f, indent=2)

                    # Format better error message based on 24h limits
                    if app_remaining == '0' or user_remaining == '0':
                        reset_ts = int(app_reset) if app_reset else int(user_reset)
                        reset_time = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
                        reset_local = reset_time.astimezone()
                        hours_until = (reset_time - datetime.now(timezone.utc)).total_seconds() / 3600

                        reset_time_str = f"at {reset_local.strftime('%Y-%m-%d %H:%M:%S')} ({hours_until:.1f} hours)"
                        error_msg = f"⚠️ 24-hour tweet limit exhausted (17 tweets/24h on Free tier). Resets {reset_time_str}."
                    else:
                        # Fallback to 15-min window message
                        reset_timestamp = window_reset
                        if reset_timestamp:
                            reset_time = datetime.fromtimestamp(int(reset_timestamp), tz=timezone.utc)
                            reset_local = reset_time.astimezone()
                            minutes_until = (reset_time - datetime.now(timezone.utc)).total_seconds() / 60
                            reset_time_str = f"at {reset_local.strftime('%H:%M:%S')} ({minutes_until:.0f} minutes)"
                        error_msg = f"⚠️ Twitter rate limit hit on tweet {i+1}/{len(texts)}. Rate limit resets {reset_time_str}."
            except Exception as cache_error:
                logger.warning(f"Failed to cache rate limit info: {cache_error}")
                error_msg = f"⚠️ Twitter rate limit hit on tweet {i+1}/{len(texts)}. Rate limit resets {reset_time_str}."

            logger.error(error_msg)
            logger.error(f"Already posted {len(responses)} tweets successfully before rate limit.")

            # Update local counter with actual count from API (if we can determine it)
            try:
                if app_remaining == '0' or user_remaining == '0':
                    # We hit the 17-tweet limit
                    # Update local counter to reflect this
                    from src.tweet_counter import log_tweet_posted
                    actual_limit = int(app_limit) if app_limit else 17
                    tweets_posted_today = actual_limit  # If remaining is 0, we used all

                    # Save rate limit info to local counter cache
                    rate_limit_info = {
                        'remaining': 0,
                        'limit': actual_limit,
                        'reset': int(app_reset) if app_reset else int(user_reset),
                    }

                    # Import and save directly to cache
                    from src.tweet_counter import _save_latest_rate_limit
                    _save_latest_rate_limit(rate_limit_info)

                    logger.info(f"📊 Updated local counter: {tweets_posted_today}/{actual_limit} tweets used (from 429 error)")
            except Exception as update_error:
                logger.warning(f"Could not update local counter from 429 error: {update_error}")

            raise Exception(error_msg) from e
        except tweepy.Forbidden as e:
            # Log detailed error information for debugging
            logger.error(f"❌ 403 Forbidden error details:")
            logger.error(f"   Error: {e}")
            logger.error(f"   Tweet {i+1} text length: {len(text)} characters")
            logger.error(f"   Tweet {i+1} text preview: {text[:150]}...")
            logger.error(f"   Has response attr: {hasattr(e, 'response')}")

            # Try to get the API error details from the response
            api_error_code = None
            api_error_message = None
            if hasattr(e, 'response'):
                logger.error(f"   Response status code: {getattr(e.response, 'status_code', 'N/A')}")
                try:
                    resp_json = e.response.json()
                    logger.error(f"   Response JSON: {resp_json}")
                    # Twitter API v2 error format
                    if 'detail' in resp_json:
                        api_error_message = resp_json.get('detail')
                    if 'errors' in resp_json:
                        for err in resp_json.get('errors', []):
                            logger.error(f"   API Error: {err}")
                    if 'title' in resp_json:
                        api_error_code = resp_json.get('title')
                except Exception as json_err:
                    logger.error(f"   Could not parse response JSON: {json_err}")
                    try:
                        logger.error(f"   Response text: {e.response.text[:500]}")
                    except:
                        pass

            # Extract rate limit headers from 403 response (Twitter may include them)
            reset_time_str = "unknown"
            hours_until_reset = None
            rate_limit_found = False

            if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                headers = e.response.headers
                logger.error(f"   Response headers: {dict(headers)}")

                # Check for 24-hour limit headers
                app_remaining = headers.get('x-app-limit-24hour-remaining')
                app_reset = headers.get('x-app-limit-24hour-reset')
                user_remaining = headers.get('x-user-limit-24hour-remaining')
                user_reset = headers.get('x-user-limit-24hour-reset')

                # Also check standard rate limit headers
                window_remaining = headers.get('x-rate-limit-remaining')
                window_reset = headers.get('x-rate-limit-reset')

                if app_reset or user_reset or window_reset:
                    rate_limit_found = True
                    reset_ts = int(app_reset or user_reset or window_reset or 0)
                    if reset_ts > 0:
                        from datetime import datetime, timezone
                        reset_time = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
                        reset_local = reset_time.astimezone()
                        hours_until_reset = (reset_time - datetime.now(timezone.utc)).total_seconds() / 3600
                        reset_time_str = f"{reset_local.strftime('%Y-%m-%d %H:%M:%S')} local time ({hours_until_reset:.1f} hours from now)"

                        # Cache the rate limit info
                        try:
                            import json
                            from pathlib import Path
                            cache_data = {
                                'app_24h': {
                                    'limit': int(headers.get('x-app-limit-24hour-limit', 17)),
                                    'remaining': int(app_remaining) if app_remaining else 0,
                                    'reset': reset_ts,
                                },
                                'user_24h': {
                                    'limit': int(headers.get('x-user-limit-24hour-limit', 17)),
                                    'remaining': int(user_remaining) if user_remaining else 0,
                                    'reset': reset_ts,
                                },
                                'timestamp': datetime.now().isoformat(),
                                'source': 'forbidden_error',
                            }
                            cache_file = Path("data/twitter_rate_limits_cache.json")
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(cache_file, 'w') as f:
                                json.dump(cache_data, f, indent=2)
                            logger.info(f"Cached rate limit info from 403 error")
                        except Exception as cache_err:
                            logger.warning(f"Could not cache rate limit: {cache_err}")

                logger.error(f"   App 24h remaining: {app_remaining}")
                logger.error(f"   User 24h remaining: {user_remaining}")
                logger.error(f"   Reset time: {reset_time_str}")

            if hasattr(e, 'response'):
                logger.error(f"   Response status: {getattr(e.response, 'status_code', 'N/A')}")
                if hasattr(e.response, 'json'):
                    try:
                        error_json = e.response.json()
                        logger.error(f"   Error response: {error_json}")
                    except:
                        pass
                if hasattr(e.response, 'text'):
                    try:
                        logger.error(f"   Response text: {e.response.text[:500]}")
                    except:
                        pass

            # Try to get more details from the exception
            error_details = str(e)
            if hasattr(e, 'api_codes'):
                logger.error(f"   API codes: {e.api_codes}")
            if hasattr(e, 'api_messages'):
                logger.error(f"   API messages: {e.api_messages}")

            # Build error message with reset time if available
            reset_info = ""
            if rate_limit_found and hours_until_reset is not None:
                reset_info = f"\n\n⏰ RATE LIMIT RESET TIME: {reset_time_str}"

            error_msg = (
                f"403 Forbidden: Cannot post tweets.\n\n"
                f"🔍 Debug Information:\n"
                f"   Error occurred on tweet {i+1}/{len(texts)}\n"
                f"   Error details: {error_details}"
                f"{reset_info}\n\n"
                f"⚠️ MOST LIKELY CAUSE: Twitter Free Tier Rate Limit\n"
                f"   • Twitter Free tier allows only 17 tweets per 24 hours\n"
                f"   • This limit is NOT shown in API rate limit responses\n"
                f"   • Each tweet in a thread counts toward this limit\n"
                f"   • Media uploads also count toward limits\n\n"
                f"🔧 SOLUTIONS:\n"
                f"1. ⏰ Wait until rate limit resets: {reset_time_str}\n"
                f"2. 📊 Track your daily tweet count manually\n"
                f"3. 💰 Upgrade to Twitter Basic tier ($100/month) for 3,000 tweets/month\n\n"
                f"⚠️ OTHER POSSIBLE CAUSES (less likely if credentials worked before):\n"
                f"1. ❌ API Key and Access Token are from DIFFERENT Twitter Apps\n"
                f"2. ❌ App permissions are 'Read' only (not 'Read and write')\n"
                f"3. ❌ Access tokens were generated BEFORE permissions were changed\n\n"
                f"🔧 To verify credentials are correct:\n"
                f"1. Go to: https://developer.twitter.com/en/portal/projects-and-apps\n"
                f"2. Find your app (API Key: {api_key[:25] if 'api_key' in dir() else 'N/A'}...)\n"
                f"3. Check 'App permissions' is 'Read and write'\n"
                f"4. Verify all 4 credentials are from the SAME app\n\n"
                f"💡 TIP: Run 'python test_twitter_post.py' to verify credentials work.\n"
                f"        If that works but Streamlit doesn't, it's the rate limit!"
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

