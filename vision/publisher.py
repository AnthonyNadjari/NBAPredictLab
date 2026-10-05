"""
Post a thread (text + optional image per tweet) through the logged-in browser.

Same approach as the reply bot (no API, no per-post cost): the first tweet is a
new post from the composer, every next tweet is a reply to the previous one.

Safety, in order of importance:
- every CreateTweet request is checked BEFORE it leaves the browser (text, the tweet it
  replies to, image attached); anything unexpected is aborted, so nothing wrong is sent
- the click is never retried; everything before it is (reload and try again)
- each tweet's id is read from X's own response; when the outcome of a click is unknown
  the thread stops as "uncertain" so nobody re-posts it blindly
- progress is reported before each click and after each success (on_progress), so a
  killed process still leaves a trace of what may be live
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Callable

from playwright.sync_api import BrowserContext, Page, Route

COMPOSE_URL = "https://x.com/compose/post"
BOT_HANDLE = "NBAPredictLab"
LOGS_DIR = Path(__file__).resolve().parent / "logs"
CREATE_RE = re.compile(r"/(CreateTweet|CreateNoteTweet)(\?|$)")
PREPARE_ATTEMPTS = 3

# Draft.js pastes multi-line text as separate blocks; this is how a person pastes into X.
_PASTE_JS = """(el, text) => {
  el.focus();
  const dt = new DataTransfer();
  dt.setData('text/plain', text);
  el.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true}));
}"""


class UncertainPost(RuntimeError):
    """The Post button was clicked but X's answer is unknown: the tweet may be live."""


def _visible(locator, timeout: int) -> bool:
    try:
        locator.wait_for(state="visible", timeout=timeout)
        return True
    except Exception:
        return False


def _lines(s: str) -> list[str]:
    """Non-empty lines, inner spaces collapsed: catches merged or lost lines, tolerates blank-line rendering."""
    return [re.sub(r"\s+", " ", ln).strip() for ln in s.replace("\r", "").split("\n") if ln.strip()]


# ------------------------------------------------------------------ outgoing request guard
class _Guard:
    """Checks each CreateTweet request against what we meant to send, and aborts it otherwise."""

    def __init__(self):
        self.expect: dict | None = None
        self.refused: str | None = None

    def arm(self, text: str, reply_to: str | None, with_image: bool) -> None:
        self.expect = {"lines": _lines(text), "reply_to": reply_to, "image": with_image}
        self.refused = None

    def handle(self, route: Route) -> None:
        exp, self.expect = self.expect, None      # one request per armed tweet
        problem = None
        if exp is None:
            problem = "unexpected post request"
        else:
            try:
                v = (route.request.post_data_json or {}).get("variables") or {}
            except Exception:
                v = {}
            if not v:
                problem = "unknown request shape"
            elif _lines(str(v.get("tweet_text", ""))) != exp["lines"]:
                problem = "text differs from the preview"
            elif ((v.get("reply") or {}).get("in_reply_to_tweet_id") or None) != exp["reply_to"]:
                problem = f"would reply to {(v.get('reply') or {}).get('in_reply_to_tweet_id')} instead of {exp['reply_to']}"
            elif exp["image"] and '"media_id"' not in json.dumps(v.get("media") or {}):
                problem = "image not attached"
        if problem:
            self.refused = problem
            route.abort()
        else:
            route.continue_()


# ------------------------------------------------------------------ composer steps (retryable)
def _composer(page: Page, timeout: int = 15000):
    """The open compose/reply dialog. Never the inline composers behind it."""
    dialog = page.locator('[role="dialog"]').filter(has=page.locator('[data-testid="tweetTextarea_0"]')).last
    if not _visible(dialog, timeout):
        raise RuntimeError("composer dialog not found")
    return dialog


def _type(page: Page, editor, text: str) -> None:
    """Put `text` in the editor: paste first, then line by line (Enter = new line) if X dropped it."""
    editor.click()
    time.sleep(random.uniform(0.3, 0.6))
    editor.evaluate(_PASTE_JS, text)
    time.sleep(random.uniform(0.8, 1.2))
    if _lines(editor.inner_text()) == _lines(text):
        return
    editor.click()
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    for i, line in enumerate(text.split("\n")):
        if i:
            page.keyboard.press("Enter")
        if line:
            page.keyboard.insert_text(line)
        time.sleep(random.uniform(0.05, 0.15))
    time.sleep(random.uniform(0.5, 0.9))
    if _lines(editor.inner_text()) != _lines(text):
        raise RuntimeError("text did not land in the composer as written")


def _prepare(page: Page, text: str, image: str | None, reply_to: str | None):
    """Open the right composer, fill it, attach the image. Returns the enabled Post button."""
    if reply_to is None:
        page.goto(COMPOSE_URL, wait_until="domcontentloaded", timeout=45000)
    else:
        page.goto(f"https://x.com/{BOT_HANDLE}/status/{reply_to}", wait_until="domcontentloaded", timeout=45000)
        # only the reply button of that exact tweet (the page also shows the tweets above it)
        reply = page.locator(f'article:has(a[href$="/status/{reply_to}"]) [data-testid="reply"]').first
        if not _visible(reply, 25000):
            raise RuntimeError("reply button of the previous tweet not found")
        reply.click()
    time.sleep(random.uniform(1.5, 3.0))
    root = _composer(page)
    editor = root.locator('[data-testid="tweetTextarea_0"]').first
    if not _visible(editor, 10000):
        raise RuntimeError("text box not found")
    _type(page, editor, text)
    if image:
        root.locator('input[data-testid="fileInput"]').first.set_input_files(image)
        if not _visible(root.locator('[data-testid="attachments"]').first, 30000):
            raise RuntimeError("image did not attach")
        time.sleep(random.uniform(2.5, 4.0))   # upload; the request guard checks the media id is there
    button = root.locator('[data-testid="tweetButton"]').first
    if not _visible(button, 10000):
        raise RuntimeError("post button not found")
    for _ in range(40):
        if button.is_enabled():
            return button
        time.sleep(1)
    raise RuntimeError("post button stayed disabled (text too long or upload stuck)")


def _created_id(response) -> tuple[str | None, str | None]:
    """(tweet id, X error) from the CreateTweet response; (None, None) = sent but id unreadable."""
    try:
        data = response.json()
    except Exception:
        return None, None
    errors = data.get("errors") or []
    try:
        res = (data["data"].get("create_tweet") or data["data"].get("notetweet_create"))["tweet_results"]["result"]
        res = res.get("tweet", res)          # TweetWithVisibilityResults wraps the tweet
        return str(res["rest_id"]), None
    except Exception:
        pass
    if errors:
        return None, "; ".join(str(e.get("message", e)) for e in errors)[:200]
    return None, None


# ------------------------------------------------------------------ thread
def post_thread(context: BrowserContext, posts: list[dict], dry_run: bool = False,
                on_progress: Callable[[list[str], int | None], None] | None = None) -> tuple[list[str], str | None]:
    """posts: [{"text": str, "image": path|None}].

    Returns (posted ids, error). Stops at the first failure (a half thread stays online:
    the opener is the important part). Raises UncertainPost when a click's outcome is
    unknown. on_progress(ids, sending_index) is called before each click and after each
    success. Dry run: everything but the click (replies are tried on PUBLISH_DRY_REPLY_TO).
    """
    page = context.new_page()
    page.on("dialog", lambda d: d.accept())
    guard = _Guard()
    page.route(CREATE_RE, guard.handle)
    ids: list[str] = []
    error = None
    progress = on_progress or (lambda _ids, _sending: None)
    try:
        for i, p in enumerate(posts):
            label = f"tweet {i + 1}/{len(posts)}"
            reply_to = None if i == 0 else (ids[-1] if not dry_run else (os.getenv("PUBLISH_DRY_REPLY_TO") or ""))
            if reply_to == "":
                print(f"  [DRY RUN] {label}: no tweet to try the reply composer on, skipped", flush=True)
                continue
            button, last = None, None
            for attempt in range(PREPARE_ATTEMPTS):
                try:
                    button = _prepare(page, p["text"], p.get("image"), reply_to)
                    break
                except Exception as e:      # nothing sent yet: safe to start this tweet over
                    last = e
                    print(f"  {label}: attempt {attempt + 1} failed before sending ({e})", flush=True)
                    time.sleep(random.uniform(3, 6))
            if button is None:
                raise RuntimeError(f"{label}: {last}")
            if dry_run:
                print(f"  [DRY RUN] {label}: ready ({len(p['text'])} chars, image={bool(p.get('image'))}), not sent",
                      flush=True)
                continue

            guard.arm(p["text"], reply_to, bool(p.get("image")))
            progress(ids, i)
            try:
                with page.expect_response(lambda r: bool(CREATE_RE.search(r.url)) and r.request.method == "POST",
                                          timeout=60000) as info:
                    button.click()
                response = info.value
            except Exception as e:
                if guard.refused:
                    progress(ids, None)
                    raise RuntimeError(f"{label}: not sent, {guard.refused}")
                raise UncertainPost(f"{label}: no answer from X after clicking Post ({e})")
            if guard.refused:
                progress(ids, None)
                raise RuntimeError(f"{label}: not sent, {guard.refused}")
            tweet_id, err = _created_id(response)
            if err:
                progress(ids, None)
                raise RuntimeError(f"{label}: X refused the post ({err})")
            if not tweet_id:
                raise UncertainPost(f"{label}: X answered HTTP {response.status} without a tweet id")
            ids.append(tweet_id)
            progress(ids, None)
            print(f"  Posted {label}: {tweet_id}", flush=True)
            time.sleep(random.uniform(4, 8))
    except UncertainPost:
        _screenshot(page)
        raise
    except Exception as e:
        error = str(e)[:300]
        print(f"  Thread stopped: {error}", flush=True)
        _screenshot(page)
    finally:
        try:
            page.close()
        except Exception:
            pass
    return ids, error


def _screenshot(page: Page) -> None:
    try:
        LOGS_DIR.mkdir(exist_ok=True)
        page.screenshot(path=str(LOGS_DIR / "publish_error.png"))
    except Exception:
        pass


def write_progress(path: str | Path, total: int, dry_run: bool):
    """on_progress callback that keeps THREAD_RESULT_FILE current while posting."""
    def _write(ids: list[str], sending: int | None) -> None:
        Path(path).write_text(json.dumps({
            "posted": len(ids), "total": total, "first_id": ids[0] if ids else None, "ids": ids,
            "sending": sending, "dry_run": dry_run, "error": None,
        }))
    return _write
