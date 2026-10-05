"""
NBAVision Engine — Main entry point.
Authentication via cookies, then engine execution.
"""
from __future__ import annotations
import io
import os
import sys
import time
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
from status import record_run, summarize_session
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


STALL_LIMIT_SEC = 15 * 60


def _start_watchdog(run_id: str | None) -> None:
    """End a session that stopped making progress (a dead Playwright driver makes calls spin
    forever, the 5 Oct session sat 2 h on one post): record what was posted, then exit."""
    import threading
    import heartbeat

    def watch():
        heartbeat.beat()
        while True:
            time.sleep(30)
            if heartbeat.idle_seconds() > STALL_LIMIT_SEC:
                print(f"Watchdog: no progress for {STALL_LIMIT_SEC // 60} min, ending the session", flush=True)
                try:
                    replies = list(heartbeat.REPLIES)
                    record_run({"run_id": run_id or None, "auth": "ok", "replied": len(replies),
                                "replies": replies[-25:], "stop_reason": "stalled"})
                except Exception as e:
                    print(f"Watchdog: could not record the run ({e})", flush=True)
                sys.stdout.flush()
                os._exit(3)

    threading.Thread(target=watch, daemon=True, name="watchdog").start()


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

    from config import get_llm_api_key
    login_check = os.getenv("NBAVISION_LOGIN_CHECK", "").lower() in ("1", "true", "yes")
    if not login_check and not get_llm_api_key():
        # Without a model the old code posted canned template replies: never do that.
        msg = "LLM_API_KEY missing: add the Groq key as a repo secret. No session run."
        print(f"ERR: {msg}", flush=True)
        record_run({"run_id": run_id or None, "auth": "no_llm_key", "auth_message": msg})
        return 1

    result = launch_and_auth()
    if len(result) == 5 and result[0] is None:
        reason = result[4]
        if reason == "no_cookies":
            msg = "Not logged in to X (no profile session, no backup cookies). On the runner PC run: python vision/tools/connect_x.py"
        elif reason == "cookies_expired":
            msg = "Profile logged out and backup cookies expired. On the runner PC run: python vision/tools/connect_x.py"
        elif reason == "browser_launch_failed":
            msg = "Could not launch Chromium. Check Playwright installation."
        else:
            msg = "Not logged in to X. On the runner PC run: python vision/tools/connect_x.py"
        print(f"ERR: {msg}", flush=True)
        _write_failure_log(reason, run_id)
        record_run({"run_id": run_id or None, "auth": reason, "auth_message": msg})
        notify_auth_failure(reason)
        return 1

    pw, browser, context, page = result[0], result[1], result[2], result[3]
    print("Auth OK. Session started.", flush=True)

    profile_stats_run_at_start(page)

    if login_check:
        # Connection test only: log in, record followers, save the session, post nothing
        from auth import logged_in_handle
        save_session_state(context)
        handle = logged_in_handle(page)
        print(f"Login check OK ({handle or '?'}). Nothing posted.", flush=True)
        record_run({"run_id": run_id or None, "auth": "ok", "login_check": True, "handle": handle})
        try:
            context.close()
            pw.stop()
        except Exception:
            pass
        return 0

    _start_watchdog(run_id)
    try:
        log_data = run_session(page, context, browser=browser, playwright_instance=pw)
        print("Session ended.", flush=True)
        record_run(summarize_session(log_data, auth="ok"))
        if log_data.get("browser_gone"):
            # Any Playwright call would now hang: leave without cleanup (systemd reaps the cgroup)
            print("Browser gone: exiting without cleanup.", flush=True)
            sys.stdout.flush()
            os._exit(0)
        save_session_state(context)
        return 0
    except Exception as e:
        print(f"Session error: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        _write_failure_log(f"session_crash: {e}", run_id)
        record_run({"run_id": run_id or None, "auth": "ok", "crash": str(e)[:300]})
        return 1
    finally:
        try:
            save_session_state(context)
        except Exception:
            pass
        try:
            if browser:
                browser.close()
            elif context:
                context.close()
        except Exception:
            pass
        try:
            if pw:
                pw.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
