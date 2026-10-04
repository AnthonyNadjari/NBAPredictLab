"""
NBAVision Engine — Main entry point.
Authentication via cookies, then engine execution.
"""
import io
import os
import sys
import traceback

# Force UTF-8 output so emoji/unicode in tweets and LLM replies don't crash
# the process on Windows consoles that default to cp1252/charmap.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from auth import launch_and_auth, save_session_state
from engine import run_session
from notify import notify_auth_failure
from profile_stats import run_at_start as profile_stats_run_at_start
from session_log import write_session_log, build_session_log
from config import (
    MAX_REPLIES,
    CYCLE_INTERVAL_MINUTES,
    MAX_CONSECUTIVE_ERRORS,
    MAX_POSTING_FAILURES,
    DRY_RUN,
    KEYWORDS_PER_CYCLE,
)


def _write_failure_log(reason: str, run_id: str) -> None:
    """Write a minimal session log on auth failure so artifacts always exist."""
    from datetime import datetime
    from config import TZ
    now = datetime.now(TZ).isoformat()
    log_data = build_session_log(
        start_time=now,
        end_time=now,
        total_scraped=0,
        total_filtered=0,
        total_scored=0,
        total_llm_calls=0,
        total_replied=0,
        total_skipped=0,
        skip_reasons={},
        avg_response_length=0,
        avg_engagement_velocity=0,
        replies_posted=[],
        run_id=run_id or None,
        events=[{"step": "auth_failure", "at": now, "detail": {"reason": reason}}],
    )
    path = write_session_log(log_data)
    print(f"Failure log written to {path}", flush=True)


def main() -> int:
    run_id = os.environ.get("NBAVISION_RUN_ID", "")
    print(f"NBAVision Engine starting. Run ID: {run_id or '(local)'}", flush=True)
    print(
        f"Config: max_replies={MAX_REPLIES}, cycle_interval_min={CYCLE_INTERVAL_MINUTES}, "
        f"max_consecutive_errors={MAX_CONSECUTIVE_ERRORS}, max_posting_failures={MAX_POSTING_FAILURES}, "
        f"keywords_per_cycle={KEYWORDS_PER_CYCLE}, dry_run={DRY_RUN}",
        flush=True,
    )
    print(f"Working directory: {os.getcwd()}", flush=True)

    result = launch_and_auth()
    if len(result) == 5 and result[0] is None:
        reason = result[4]
        if reason == "no_cookies":
            msg = "No cookies. Set TWITTER_COOKIES_JSON (repo secret or env)."
        elif reason == "cookies_expired":
            msg = "Critical cookies are expired or missing. Re-export from browser."
        elif reason == "browser_launch_failed":
            msg = "Could not launch Chromium. Check Playwright installation."
        else:
            msg = "Cookies present but session invalid or expired. Re-export cookies from the browser where you're logged in to X."
        print(f"ERR: {msg}", flush=True)
        _write_failure_log(reason, run_id)
        notify_auth_failure(reason)
        return 1

    pw, browser, context, page = result[0], result[1], result[2], result[3]
    print("Auth OK. Session started.", flush=True)

    profile_stats_run_at_start(page)

    try:
        log_data = run_session(page, context, browser=browser, playwright_instance=pw)
        print("Session ended.", flush=True)
        save_session_state(context)
        return 0
    except Exception as e:
        print(f"Session error: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        _write_failure_log(f"session_crash: {e}", run_id)
        return 1
    finally:
        try:
            save_session_state(context)
        except Exception:
            pass
        try:
            if browser:
                browser.close()
        except Exception:
            pass
        try:
            if pw:
                pw.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
