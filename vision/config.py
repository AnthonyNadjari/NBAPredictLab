"""
NBAVision Engine — Configuration centralisée (Spec Section 2, 4, 11).
Credentials: credentials.json (or env vars).
"""
from __future__ import annotations
import json
import os
from datetime import timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")  # never walk up into unrelated .env files
REPO_ROOT = PROJECT_ROOT.parent

# Machine-local state (browser profile, last session): OUTSIDE the git checkout,
# because actions/checkout runs `git clean -ffdx` and wiped state.json every run.
LOCAL_STATE_DIR = Path(os.getenv("NBAVISION_STATE_DIR") or (
    Path(os.getenv("LOCALAPPDATA") or Path.home() / ".local" / "share") / "NBAVision"))
LOCAL_STATE_DIR.mkdir(parents=True, exist_ok=True)

from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Paris")  # schedule and timestamps follow Paris time (DST-aware)


def _credentials_path() -> Path | None:
    for base in (PROJECT_ROOT, Path.cwd()):
        p = base / "credentials.json"
        if p.is_file():
            return p
    return None


def _load_credentials() -> dict | None:
    path = _credentials_path()
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Searches must be unambiguously NBA: bare "Heat", "Magic", "Kings", "Spurs", "Hawks",
# "Thunder", "Jazz", "Nets" also match football, AFL, weather... (dry run 05/10: most
# scraped tweets were not about basketball).
KEYWORDS = [
    "NBA", "NBA trade", "NBA injury", "NBA preseason", "NBA season", "NBA MVP", "NBA rumors",
    "Lakers", "Celtics", "Warriors", "Bucks", "Nuggets", "76ers", "Sixers", "Knicks", "Mavericks",
    "Cavaliers", "Clippers", "Grizzlies", "Timberwolves", "Pelicans", "Pacers", "Raptors", "Hornets",
    "Wizards", "Pistons", "Trail Blazers", "Phoenix Suns", "Houston Rockets", "Atlanta Hawks",
    "Chicago Bulls", "Miami Heat", "Orlando Magic", "Sacramento Kings", "OKC Thunder", "Utah Jazz",
    "Brooklyn Nets", "San Antonio Spurs",
    "LeBron", "Steph Curry", "Jokic", "Giannis", "Wembanyama", "Luka Doncic", "Jayson Tatum",
    "Shai Gilgeous-Alexander", "Anthony Edwards", "Jalen Brunson", "Joel Embiid", "Kevin Durant",
    "Ja Morant", "Cooper Flagg", "Tyrese Haliburton", "Donovan Mitchell", "Devin Booker",
]

# Tweet text must contain at least one of these (lowercase) to pass filter — keeps replies NBA-only
NBA_TEXT_KEYWORDS = frozenset([
    "nba", "lakers", "celtics", "warriors", "bucks", "nuggets", "76ers", "sixers", "knicks",
    "mavericks", "mavs", "cavaliers", "cavs", "clippers", "grizzlies", "timberwolves", "pelicans",
    "pacers", "raptors", "hornets", "wizards", "pistons", "trail blazers", "blazers",
    "phoenix suns", "houston rockets", "atlanta hawks", "chicago bulls", "miami heat", "orlando magic",
    "sacramento kings", "okc", "utah jazz", "brooklyn nets", "san antonio spurs",
    "lebron", "curry", "jokic", "giannis", "wembanyama", "wemby", "luka", "doncic", "tatum", "embiid",
    "durant", "shai", "sga", "brunson", "ja morant", "cooper flagg", "haliburton", "booker",
])

# Search targeting: X operators appended to every keyword search so the "Latest"
# tab returns tweets that already have traction (before: 166k of 225k scraped tweets
# were dropped for too few likes), in English, and not replies themselves.
SEARCH_MIN_FAVES = int(os.getenv("SEARCH_MIN_FAVES", "15"))
SEARCH_SUFFIX = f" min_faves:{SEARCH_MIN_FAVES} lang:en -filter:replies"

# High-reach NBA accounts: replying early under their posts is where impressions are.
WATCHLIST_ACCOUNTS = [
    "NBA", "ShamsCharania", "BleacherReport", "TheHoopCentral", "LegionHoops",
    "ClutchPoints", "statmuse", "espn", "TheNBACentral", "NBAonTNT", "UnderdogNBA",
]
WATCHLIST_QUERIES_PER_CYCLE = 4   # big accounts first: early replies there get the views
WATCHLIST_SCORE_BONUS = 2.0      # added to the ranking score of watchlist tweets
EARLY_REPLY_MINUTES = 20         # extra bonus while a tweet is this fresh

# How many keywords to sample per cycle
KEYWORDS_PER_CYCLE = int(os.getenv("KEYWORDS_PER_CYCLE", "28"))

# Session limits
def _schedule_setting(key: str, default):
    try:
        return json.loads((PROJECT_ROOT.parent / "docs" / "vision" / "schedule.json")
                          .read_text(encoding="utf-8")).get("settings", {}).get(key, default)
    except Exception:
        return default


# Replies per session: workflow input > schedule.json setting (control panel) > 25
MAX_REPLIES = int(os.getenv("MAX_REPLIES") or _schedule_setting("max_replies", 25))
MAX_REPLIES_PER_AUTHOR = 1
CYCLE_INTERVAL_MINUTES = 0.5
MAX_CONSECUTIVE_ERRORS = 5
MAX_POSTING_FAILURES = 5

# Filtering — engagement-based; skip image/video tweets with no real text
MAX_MINUTES_SINCE_POST = int(os.getenv("MAX_MINUTES_SINCE_POST", "120"))  # 360 off-season (fewer fresh tweets)
MIN_LIKES = int(os.getenv("MIN_LIKES", "5"))  # Only reply to tweets with some traction
MIN_TEXT_LENGTH = 20
MAX_HASHTAGS = 5
MIN_TEXT_LENGTH_IF_MEDIA = 40

# Scoring — top N kept per cycle
TOP_N_SCORED = 70   # DeepSeek skips more (35% of tweets answered vs 85%): look at more

# LLM
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))  # DeepSeek reasons before answering: 60
LLM_RETRY_MAX = 2

# Reply validation
MAX_RESPONSES_SAME_FIRST_WORD = 4
MAX_EMOJIS_IN_SESSION = 6
MAX_SENTENCES = 3
TFIDF_SIMILARITY_THRESHOLD = 0.65

# Posting — human-like, faster
WAIT_BEFORE_NEXT_TWEET_SEC_MIN = 20
WAIT_BEFORE_NEXT_TWEET_SEC_MAX = 45

# Scraping delays
SEARCH_WAIT_SEC_MIN = 2.5
SEARCH_WAIT_SEC_MAX = 4
SCROLL_WAIT_SEC_MIN = 1.2
SCROLL_WAIT_SEC_MAX = 2.5
SCROLL_DELTA_MIN = 1200
SCROLL_DELTA_MAX = 2000
SCROLL_COUNT = 5
CYCLE_INTERVAL_JITTER_SEC = 10

# X (Twitter)
TWITTER_HOME_URL = "https://x.com/home"
TWITTER_SEARCH_BASE = "https://x.com/search?q={query}&f=live"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
BROWSER_VIEWPORT = {"width": 1280, "height": 720}

# Cookie / state persistence
STATE_FILE = LOCAL_STATE_DIR / "state.json"
PROFILE_DIR = LOCAL_STATE_DIR / "profile"   # persistent Chromium profile

# Stats tracker (follower count per day, scraped at run start)
BOT_PROFILE_USERNAME = os.getenv("BOT_PROFILE_USERNAME", "").strip()
STATS_FILE = REPO_ROOT / "docs" / "vision" / "stats.json"
RUNS_FILE = REPO_ROOT / "docs" / "vision" / "runs.json"      # compact session summaries for the console
SCHEDULE_FILE = REPO_ROOT / "docs" / "vision" / "schedule.json"

# Dry-run: scrape + LLM but do NOT post replies
DRY_RUN = os.getenv("DRY_RUN", "").strip().lower() in ("1", "true", "yes")

# Discord webhook for failure notifications (optional)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


# --------------- credential helpers ---------------

def get_twitter_cookies_json() -> str:
    raw = os.getenv("TWITTER_COOKIES_JSON", "").strip()
    if raw and raw not in ("", "[]"):
        return raw
    creds = _load_credentials()
    if creds and isinstance(creds.get("twitter_cookies"), list):
        return json.dumps(creds["twitter_cookies"], separators=(",", ":"))
    path = os.getenv("TWITTER_COOKIES_FILE", "cookies.json").strip()
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return "[]"


def get_llm_api_key() -> str:
    s = os.getenv("LLM_API_KEY", "").strip()
    if s:
        return s
    creds = _load_credentials()
    if creds and isinstance(creds.get("llm_api_key"), str):
        return creds["llm_api_key"].strip()
    return ""


def get_llm_model() -> str:
    s = os.getenv("LLM_MODEL", "").strip()
    if s:
        return s
    creds = _load_credentials()
    if creds and isinstance(creds.get("llm_model"), str):
        return creds["llm_model"].strip()
    # Groq model list (Oct 2026): GPT OSS 120B / 20B, Qwen. The Llama models were retired.
    return "openai/gpt-oss-120b"
