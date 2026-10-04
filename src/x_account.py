"""X account health: API login check + daily follower count.

Writes docs/x_status.json (shown in the control panel) and adds today's
follower count to docs/vision/stats.json (the follower trend, shared with the
reply bot which also records it when it runs).

    python -m src.x_account      # needs TW_API_KEY, TW_API_SECRET, TW_ACCESS_TOKEN, TW_ACCESS_SECRET
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "docs" / "x_status.json"
STATS_PATH = ROOT / "docs" / "vision" / "stats.json"
PARIS = ZoneInfo("Europe/Paris")


def check_account() -> dict:
    """Return {'ok', 'handle', 'followers', 'following', 'tweets', 'error'} using the posting credentials."""
    keys = [os.getenv(k, "") for k in ("TW_API_KEY", "TW_API_SECRET", "TW_ACCESS_TOKEN", "TW_ACCESS_SECRET")]
    if not all(keys):
        return {"ok": False, "error": "Twitter API secrets missing in this workflow"}
    try:
        import tweepy
        client = tweepy.Client(consumer_key=keys[0], consumer_secret=keys[1],
                               access_token=keys[2], access_token_secret=keys[3])
        me = client.get_me(user_fields=["public_metrics"], user_auth=True)
        m = (me.data.public_metrics or {}) if me and me.data else {}
        return {"ok": True, "handle": f"@{me.data.username}", "followers": m.get("followers_count"),
                "following": m.get("following_count"), "tweets": m.get("tweet_count"), "error": None}
    except Exception as e:  # report, never crash the daily run
        msg = str(e)
        if "401" in msg:
            msg = "401 Unauthorized: keys or tokens revoked/regenerated; update the TWITTER_* secrets"
        elif "403" in msg:
            msg = "403 Forbidden: app lacks access (check the X developer portal project/app permissions)"
        elif "429" in msg:
            msg = "429 rate limited (users/me allows ~25 calls/day on the free tier)"
        return {"ok": False, "error": msg[:300]}


def record_followers(followers: int, following: int = None, path: Path = STATS_PATH) -> None:
    """Upsert today's (Paris date) follower count into the trend file."""
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        entries = []
    today = datetime.now(PARIS).strftime("%Y-%m-%d")
    by_date = {e.get("date"): e for e in entries if isinstance(e, dict)}
    by_date[today] = {**by_date.get(today, {}), "date": today, "followers": followers,
                      **({"following": following} if following is not None else {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(by_date.values(), key=lambda e: e["date"]), indent=2), encoding="utf-8")


def write_status(result: dict, path: Path = STATUS_PATH) -> None:
    previous = {}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = {**result, "checked_at": now,
              "last_ok_at": now if result.get("ok") else previous.get("last_ok_at")}
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run() -> dict:
    result = check_account()
    write_status(result)
    if result.get("ok") and result.get("followers") is not None:
        record_followers(result["followers"], result.get("following"))
    return result


def apply(result: dict) -> None:
    """Re-apply a saved check result on top of freshly pulled files (used when a push races)."""
    write_status(result)
    if result.get("ok") and result.get("followers") is not None:
        record_followers(result["followers"], result.get("following"))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--apply":
        apply(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
        sys.exit(0)
    r = run()
    print(json.dumps(r, indent=2))
    if len(sys.argv) == 3 and sys.argv[1] == "--save":
        Path(sys.argv[2]).write_text(json.dumps(r), encoding="utf-8")
    sys.exit(0 if r.get("ok") else 1)
