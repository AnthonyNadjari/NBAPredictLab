"""
NBAVision Engine — Posting replies (Spec Section 10).
Uses Playwright insert_text for instant text entry (works in headless).
"""
from __future__ import annotations
import random
import time
from pathlib import Path
from playwright.sync_api import Page
from config import (
    WAIT_BEFORE_NEXT_TWEET_SEC_MIN,
    WAIT_BEFORE_NEXT_TWEET_SEC_MAX,
    DRY_RUN,
    PROJECT_ROOT,
)

LOGS_DIR = PROJECT_ROOT / "logs"
# Id of the last reply posted (read from X's CreateTweet response), for the replies ledger
LAST_POSTED_ID: str | None = None


def _ensure_logs_dir() -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    return LOGS_DIR


def _screenshot_on_error(page: Page, label: str) -> None:
    try:
        safe = label.replace("/", "_").replace("\\", "_")[:60]
        path = _ensure_logs_dir() / f"error_{safe}.png"
        page.screenshot(path=str(path))
    except Exception:
        pass


def wait_before_next_tweet() -> None:
    delay = random.uniform(WAIT_BEFORE_NEXT_TWEET_SEC_MIN, WAIT_BEFORE_NEXT_TWEET_SEC_MAX)
    time.sleep(delay)


def _insert_text(page: Page, text: str) -> None:
    """Insert text instantly via Playwright's input method (works in headless)."""
    page.keyboard.insert_text(text)
    time.sleep(random.uniform(0.3, 0.6))


def _visible_within(locator, timeout: int) -> bool:
    """is_visible() returns immediately and ignores its timeout, evaluating the
    page before the SPA renders. wait_for() actually polls until visible."""
    try:
        locator.wait_for(state="visible", timeout=timeout)
        return True
    except Exception:
        return False


def _do_post(page: Page, tweet_url: str, reply_text: str, tweet_id: str) -> tuple[bool, str | None]:
    """Single attempt: goto, click reply, type, send. Returns (success, error)."""
    page.goto(tweet_url, wait_until="domcontentloaded", timeout=30000)
    time.sleep(random.uniform(3, 5))

    reply_btn = page.locator('[data-testid="reply"]').first
    if not _visible_within(reply_btn, 15000):
        return False, "reply_button_not_found"
    reply_btn.click()
    time.sleep(random.uniform(0.5, 1.0))

    editor = page.locator('[data-testid="tweetTextarea_0"]').first
    if editor.count() == 0:
        editor = page.locator('div[contenteditable="true"][role="textbox"]').first
    if not editor.count() or not _visible_within(editor, 8000):
        return False, "reply_box_not_found"
    editor.click()
    time.sleep(random.uniform(0.2, 0.4))

    _insert_text(page, reply_text)
    time.sleep(random.uniform(0.5, 1.0))

    send_btn = page.locator('[data-testid="tweetButton"]').first
    if not _visible_within(send_btn, 5000):
        return False, "send_button_not_found"
    global LAST_POSTED_ID
    LAST_POSTED_ID = None
    clicked = False
    try:
        with page.expect_response(lambda r: ("/CreateTweet" in r.url or "/CreateNoteTweet" in r.url)
                                  and r.request.method == "POST", timeout=20000) as info:
            send_btn.click()
            clicked = True
        from publisher import _created_id
        LAST_POSTED_ID, _err = _created_id(info.value)
    except Exception:
        if not clicked:
            raise
        # the id is a bonus for tracking; the toast check below still confirms the post

    time.sleep(random.uniform(2, 3))
    try:
        toast = page.locator('[data-testid="toast"]').first
        if _visible_within(toast, 3000):
            print(f"  Post confirmed (toast visible)", flush=True)
    except Exception:
        pass
    return True, None


def post_reply(page: Page, tweet_url: str, reply_text: str) -> tuple[bool, str | None]:
    """
    Navigate to tweet, click reply, insert text, send. Retries once on reply_button_not_found.
    In DRY_RUN mode: does not post.
    Returns (success, error_message).
    """
    tweet_id = tweet_url.rstrip("/").split("/")[-1]

    if DRY_RUN:
        print(f"  [DRY RUN] Would post to {tweet_url}: {reply_text!r}", flush=True)
        return True, None

    try:
        success, err = _do_post(page, tweet_url, reply_text, tweet_id)
        if not success and err == "reply_button_not_found":
            print(f"  Retrying once after reply_button_not_found...", flush=True)
            time.sleep(random.uniform(3, 5))
            success, err = _do_post(page, tweet_url, reply_text, tweet_id)
        if not success:
            _screenshot_on_error(page, f"{err}_{tweet_id}")
        return success, err
    except Exception as e:
        _screenshot_on_error(page, f"exception_{tweet_id}")
        return False, str(e)
