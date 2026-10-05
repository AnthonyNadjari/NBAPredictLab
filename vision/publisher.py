"""
Post a thread (text + optional image per tweet) through the logged-in browser.

Same approach as the reply bot (no API, no per-post cost): the first tweet is a
new post from the composer, every next tweet is a reply to the previous one.
Each tweet's id is read from X's own CreateTweet response, never guessed.
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path

from playwright.sync_api import BrowserContext, Page

COMPOSE_URL = "https://x.com/compose/post"
BOT_HANDLE = "NBAPredictLab"
LOGS_DIR = Path(__file__).resolve().parent / "logs"

# Draft.js pastes multi-line text as separate blocks; this is how a person pastes into X.
_PASTE_JS = """(el, text) => {
  el.focus();
  const dt = new DataTransfer();
  dt.setData('text/plain', text);
  el.dispatchEvent(new ClipboardEvent('paste', {clipboardData: dt, bubbles: true, cancelable: true}));
}"""


def _visible(locator, timeout: int) -> bool:
    try:
        locator.wait_for(state="visible", timeout=timeout)
        return True
    except Exception:
        return False


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _composer(page: Page):
    """The open compose/reply dialog (falls back to the page when X renders it inline)."""
    dialog = page.locator('[role="dialog"]').last
    return dialog if _visible(dialog, 5000) else page


def _type(page: Page, editor, text: str) -> None:
    """Put `text` in the editor: paste first, then line by line (Enter = new line) if X dropped it."""
    editor.click()
    time.sleep(random.uniform(0.3, 0.6))
    editor.evaluate(_PASTE_JS, text)
    time.sleep(random.uniform(0.8, 1.2))
    if _norm(editor.inner_text()) == _norm(text):
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
    got = _norm(editor.inner_text())
    if got != _norm(text):
        raise RuntimeError(f"text did not land in the composer (got {len(got)} of {len(_norm(text))} chars)")


def _created_id(response) -> tuple[str | None, str | None]:
    """(tweet id, X error message) from the CreateTweet GraphQL response."""
    try:
        data = response.json()
    except Exception:
        return None, f"HTTP {response.status}, unreadable response"
    try:
        res = data["data"]["create_tweet"]["tweet_results"]["result"]
        res = res.get("tweet", res)          # TweetWithVisibilityResults wraps the tweet
        return str(res["rest_id"]), None
    except Exception:
        errors = data.get("errors") or []
        msg = "; ".join(str(e.get("message", e)) for e in errors)[:200] if errors else f"HTTP {response.status}"
        return None, msg


def _fill_and_send(page: Page, text: str, image: str | None, dry_run: bool, label: str) -> str:
    """Fill the open composer (new post or reply) and send it. Returns the new tweet id."""
    root = _composer(page)
    editor = root.locator('[data-testid="tweetTextarea_0"]').first
    if not _visible(editor, 15000):
        raise RuntimeError(f"{label}: composer not found")
    _type(page, editor, text)

    if image:
        root.locator('input[data-testid="fileInput"]').first.set_input_files(image)
        if not _visible(root.locator('[data-testid="attachments"]').first, 30000):
            raise RuntimeError(f"{label}: image did not attach")
        time.sleep(random.uniform(2.5, 4.0))    # X processes the upload before enabling the button

    button = root.locator('[data-testid="tweetButton"]').first
    if not _visible(button, 10000):
        raise RuntimeError(f"{label}: post button not found")
    for _ in range(40):
        if button.is_enabled():
            break
        time.sleep(1)
    else:
        raise RuntimeError(f"{label}: post button stayed disabled (text too long or upload stuck)")
    if dry_run:
        print(f"  [DRY RUN] {label}: composer ready ({len(text)} chars, image={bool(image)}), not sent", flush=True)
        return "dry"

    with page.expect_response(lambda r: "CreateTweet" in r.url and r.request.method == "POST",
                              timeout=60000) as info:
        button.click()
    tweet_id, err = _created_id(info.value)
    if not tweet_id:
        raise RuntimeError(f"{label}: X refused the post ({err})")
    return tweet_id


def _open_reply(page: Page, tweet_id: str, label: str) -> None:
    page.goto(f"https://x.com/{BOT_HANDLE}/status/{tweet_id}", wait_until="domcontentloaded", timeout=45000)
    reply = page.locator(f'article:has(a[href*="/status/{tweet_id}"]) [data-testid="reply"]').first
    if not _visible(reply, 20000):
        reply = page.locator('[data-testid="reply"]').first
        if not _visible(reply, 5000):
            raise RuntimeError(f"{label}: reply button not found")
    reply.click()


def post_thread(context: BrowserContext, posts: list[dict], dry_run: bool = False) -> tuple[list[str], str | None]:
    """posts: [{"text": str, "image": path|None}].

    Returns (posted ids, error). Stops at the first failure: a half thread stays
    online (the opener is the important part) and the error says where it stopped.
    In dry run nothing is sent; replies are tried against PUBLISH_DRY_REPLY_TO (one of
    our own tweets) when set, so the reply composer is exercised too.
    """
    page = context.new_page()
    page.on("dialog", lambda d: d.accept())
    ids: list[str] = []
    error = None
    try:
        for i, p in enumerate(posts):
            label = f"tweet {i + 1}/{len(posts)}"
            if i == 0:
                page.goto(COMPOSE_URL, wait_until="domcontentloaded", timeout=45000)
            else:
                target = ids[-1] if not dry_run else os.getenv("PUBLISH_DRY_REPLY_TO", "")
                if not target:
                    print(f"  [DRY RUN] {label}: no tweet to try the reply composer on, skipped", flush=True)
                    continue
                _open_reply(page, target, label)
            time.sleep(random.uniform(1.5, 3.0))
            tweet_id = _fill_and_send(page, p["text"], p.get("image"), dry_run, label)
            if not dry_run:
                ids.append(tweet_id)
                print(f"  Posted {label}: {tweet_id}", flush=True)
            time.sleep(random.uniform(4, 8))
    except Exception as e:
        error = str(e)[:300]
        print(f"  Thread stopped: {error}", flush=True)
        try:
            LOGS_DIR.mkdir(exist_ok=True)
            page.screenshot(path=str(LOGS_DIR / "publish_error.png"))
        except Exception:
            pass
    finally:
        page.close()
    return ids, error
