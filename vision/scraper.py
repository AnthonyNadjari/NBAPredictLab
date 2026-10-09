"""
NBAVision Engine — Scraping Twitter via Playwright DOM (Spec Section 4).
Uses multiple browser tabs to scrape keywords in parallel batches,
cutting total scrape time by ~60%.
"""
from __future__ import annotations
import os
import json
import random
import re
import time
from urllib.parse import quote_plus

from playwright.sync_api import Page, BrowserContext

from config import (
    KEYWORDS,
    KEYWORDS_PER_CYCLE,
    TWITTER_SEARCH_BASE,
    SEARCH_WAIT_SEC_MIN,
    SEARCH_WAIT_SEC_MAX,
    SCROLL_DELTA_MIN,
    SCROLL_DELTA_MAX,
    SCROLL_WAIT_SEC_MIN,
    SCROLL_WAIT_SEC_MAX,
    SCROLL_COUNT,
)

HIGH_PRIORITY_KEYWORDS = {
    "NBA", "NBA trade", "NBA injury", "LeBron", "Steph Curry", "Luka Doncic", "Wembanyama",
}

PARALLEL_TABS = int(os.getenv("PARALLEL_TABS", "3"))  # 1 on the shared server (memory cap)


def _wait_random(low: float, high: float) -> None:
    time.sleep(random.uniform(low, high))


def _parse_int_from_aria(aria_label: str | None) -> int:
    if not aria_label:
        return 0
    digits = re.sub(r"[^\d]", "", aria_label)
    return int(digits) if digits else 0


def _safe_locator_attr(locator, attr: str, default=None):
    try:
        if locator.count():
            return locator.get_attribute(attr)
    except Exception:
        pass
    return default


def extract_tweets_from_page(page: Page) -> list[dict]:
    try:
        articles = page.locator('article[data-testid="tweet"]')
        count = articles.count()
    except Exception:
        return []

    tweets = []
    for i in range(count):
        try:
            art = articles.nth(i)

            link_el = art.locator('a[href*="/status/"]').first
            href = _safe_locator_attr(link_el, "href")
            tweet_id = None
            if href:
                match = re.search(r"/status/(\d+)", href)
                if match:
                    tweet_id = match.group(1)

            text_el = art.locator('div[data-testid="tweetText"]').first
            try:
                text = (text_el.inner_text() if text_el.count() else "") or ""
            except Exception:
                text = ""

            user_el = art.locator('a[href^="/"][role="link"]').first
            user_href = _safe_locator_attr(user_el, "href")
            username = (user_href or "").strip("/").split("/")[0] if user_href else ""

            like_el = art.locator('[data-testid="like"]').first
            reply_el = art.locator('[data-testid="reply"]').first
            retweet_el = art.locator('[data-testid="retweet"]').first
            likes = _parse_int_from_aria(_safe_locator_attr(like_el, "aria-label"))
            replies = _parse_int_from_aria(_safe_locator_attr(reply_el, "aria-label"))
            retweets = _parse_int_from_aria(_safe_locator_attr(retweet_el, "aria-label"))

            time_el = art.locator("time").first
            timestamp = _safe_locator_attr(time_el, "datetime") or ""

            has_media = False
            try:
                media_loc = art.locator('[data-testid="tweetPhoto"], [data-testid="videoPlayer"]')
                has_media = media_loc.count() > 0
            except Exception:
                pass

            if tweet_id and username:
                tweets.append({
                    "tweet_id": tweet_id,
                    "text": text,
                    "username": username,
                    "likes": likes,
                    "replies": replies,
                    "retweets": retweets,
                    "timestamp": timestamp,
                    "followers": None,
                    "has_media": has_media,
                })
        except Exception:
            continue
    return tweets


def _scrape_single_tab(page: Page, keyword: str) -> list[dict]:
    """Navigate one tab to a keyword search, scroll, extract."""
    url = search_url(keyword)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print(f"      Navigation timeout for {keyword!r}: {e}", flush=True)
        return []

    _wait_random(SEARCH_WAIT_SEC_MIN, SEARCH_WAIT_SEC_MAX)

    for _ in range(SCROLL_COUNT):
        delta = random.randint(SCROLL_DELTA_MIN, SCROLL_DELTA_MAX)
        page.mouse.wheel(0, delta)
        _wait_random(SCROLL_WAIT_SEC_MIN, SCROLL_WAIT_SEC_MAX)

    tweets = extract_tweets_from_page(page)
    if not tweets:
        _wait_random(2, 4)
        tweets = extract_tweets_from_page(page)
    return tweets


def _scrape_batch_parallel(context: BrowserContext, main_page: Page, keywords: list[str]) -> list[tuple[str, list[dict]]]:
    """
    Scrape a batch of keywords using multiple tabs in parallel.
    Opens extra tabs, navigates all simultaneously, then scrolls+extracts round-robin.
    Returns list of (keyword, tweets) pairs.
    """
    n = len(keywords)
    if n == 0:
        return []

    if n == 1:
        return [(keywords[0], _scrape_single_tab(main_page, keywords[0]))]

    pages: list[Page] = [main_page]
    extra_pages: list[Page] = []
    for _ in range(n - 1):
        try:
            p = context.new_page()
            pages.append(p)
            extra_pages.append(p)
        except Exception:
            break

    results: list[tuple[str, list[dict]]] = []

    # Navigate all tabs (fast — goto returns quickly with domcontentloaded)
    for i, (pg, kw) in enumerate(zip(pages, keywords)):
        url = search_url(kw)
        try:
            pg.goto(url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"      Navigation timeout for {kw!r}: {e}", flush=True)

    # Single shared wait (all pages are loading simultaneously)
    _wait_random(SEARCH_WAIT_SEC_MAX, SEARCH_WAIT_SEC_MAX + 1)

    # Scroll and extract from each tab
    for pg, kw in zip(pages, keywords):
        try:
            for _ in range(SCROLL_COUNT):
                delta = random.randint(SCROLL_DELTA_MIN, SCROLL_DELTA_MAX)
                pg.mouse.wheel(0, delta)
                _wait_random(SCROLL_WAIT_SEC_MIN, SCROLL_WAIT_SEC_MAX)

            tweets = extract_tweets_from_page(pg)
            if not tweets:
                _wait_random(1, 2)
                tweets = extract_tweets_from_page(pg)
            results.append((kw, tweets))
        except Exception as e:
            print(f"      Scrape error for {kw!r}: {e}", flush=True)
            results.append((kw, []))

    for p in extra_pages:
        try:
            p.close()
        except Exception:
            pass

    return results


def search_url(keyword: str) -> str:
    """Live search URL. Watchlist queries (from:...) skip the min_faves filter: we want them fresh."""
    from config import SEARCH_SUFFIX
    q = keyword if keyword.startswith("(from:") else keyword + SEARCH_SUFFIX
    return TWITTER_SEARCH_BASE.format(query=quote_plus(q))


DISCOVERED = None  # loaded lazily: {handle: {"hits": n, "best": likes, "seen": iso}}


def _discovered_file():
    from config import LOCAL_STATE_DIR
    return LOCAL_STATE_DIR / "discovered_accounts.json"


def _load_discovered() -> dict:
    global DISCOVERED
    if DISCOVERED is None:
        try:
            DISCOVERED = json.loads(_discovered_file().read_text(encoding="utf-8"))
        except Exception:
            DISCOVERED = {}
    return DISCOVERED


def learn_accounts(tweets: list[dict], min_likes: int = 100) -> int:
    """Remember every account whose NBA tweet reached `min_likes`: the watchlist grows by itself."""
    from datetime import datetime, timezone
    d, added = _load_discovered(), 0
    for t in tweets:
        u, likes = (t.get("username") or "").strip(), int(t.get("likes") or 0)
        if not u or likes < min_likes:
            continue
        e = d.setdefault(u, {"hits": 0, "best": 0})
        added += e["hits"] == 0
        e["hits"] += 1
        e["best"] = max(e["best"], likes)
        e["seen"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    try:
        _discovered_file().write_text(json.dumps(d), encoding="utf-8")
    except Exception:
        pass
    return added


def _watchlist_queries(cycle_index: int) -> list[str]:
    """Fixed big accounts + up to 400 discovered ones (most often viral first), rotated: each cycle
    searches WATCHLIST_QUERIES_PER_CYCLE queries of 6 accounts, a different slice every cycle."""
    from config import WATCHLIST_ACCOUNTS, WATCHLIST_QUERIES_PER_CYCLE
    disc = sorted(_load_discovered().items(), key=lambda kv: (-kv[1]["hits"], -kv[1]["best"]))
    fixed = [WATCHLIST_ACCOUNTS[i:i + 6] for i in range(0, len(WATCHLIST_ACCOUNTS), 6)]
    extra_pool = [u for u, _ in disc if u not in WATCHLIST_ACCOUNTS][:400]
    extra = [extra_pool[i:i + 6] for i in range(0, len(extra_pool), 6)]
    half = WATCHLIST_QUERIES_PER_CYCLE // 2 if extra else WATCHLIST_QUERIES_PER_CYCLE
    pick = [fixed[(cycle_index * half + k) % len(fixed)] for k in range(min(half, len(fixed)))]
    if extra:
        n = WATCHLIST_QUERIES_PER_CYCLE - len(pick)
        pick += [extra[(cycle_index * n + k) % len(extra)] for k in range(min(n, len(extra)))]
    return ["(" + " OR ".join(f"from:{a}" for a in g) + ")" for g in pick]


def _select_keywords(cycle_index: int) -> list[str]:
    sample_size = min(KEYWORDS_PER_CYCLE, len(KEYWORDS))

    hp_in_keywords = [k for k in KEYWORDS if k in HIGH_PRIORITY_KEYWORDS]
    other = [k for k in KEYWORDS if k not in HIGH_PRIORITY_KEYWORDS]

    hp_count = min(5, len(hp_in_keywords))
    random.shuffle(hp_in_keywords)
    selected_hp = hp_in_keywords[:hp_count]

    remaining_slots = sample_size - len(selected_hp)
    selected_other = random.sample(other, min(remaining_slots, len(other)))

    result = selected_hp + selected_other
    random.shuffle(result)
    return _watchlist_queries(cycle_index) + result


class BrowserGone(RuntimeError):
    """The browser or the Playwright driver is gone; the session can't continue."""


def browser_gone(ex: Exception) -> bool:
    msg = str(ex)
    return any(m in msg for m in ("Connection closed", "has been closed", "Browser closed",
                                  "Target closed", "driver"))


def scrape_all_keywords(page: Page, context: BrowserContext, cycle_index: int = 0) -> list[dict]:
    """
    Scrape keywords in parallel batches of PARALLEL_TABS tabs.
    Deduplicates by tweet_id.
    """
    keywords = _select_keywords(cycle_index)
    print(f"    Scraping {len(keywords)} keywords ({PARALLEL_TABS} tabs in parallel)", flush=True)

    seen_ids: set[str] = set()
    all_tweets: list[dict] = []
    consecutive_kw_errors = 0
    empty_count = 0

    # Split keywords into batches
    batches = [keywords[i:i + PARALLEL_TABS] for i in range(0, len(keywords), PARALLEL_TABS)]

    for batch_idx, batch in enumerate(batches):
        if consecutive_kw_errors >= 5:
            print("    Circuit breaker: 5 consecutive keyword errors, pausing 60s", flush=True)
            _wait_random(55, 65)
            consecutive_kw_errors = 0

        import heartbeat
        heartbeat.beat()
        batch_start = batch_idx * PARALLEL_TABS + 1
        kw_labels = ", ".join(f"{kw!r}" for kw in batch)
        print(f"    Batch {batch_idx + 1}/{len(batches)} (kw {batch_start}-{batch_start + len(batch) - 1}): {kw_labels}", flush=True)

        try:
            results = _scrape_batch_parallel(context, page, batch)

            batch_had_error = True
            for kw, tweets in results:
                new_in_batch = 0
                for t in tweets:
                    tid = t.get("tweet_id")
                    if tid and tid not in seen_ids:
                        seen_ids.add(tid)
                        all_tweets.append(t)
                        new_in_batch += 1
                if len(tweets) == 0:
                    empty_count += 1
                else:
                    batch_had_error = False
                print(f"      {kw!r}: {len(tweets)} tweets, {new_in_batch} new", flush=True)

            if batch_had_error and all(len(t) == 0 for _, t in results):
                consecutive_kw_errors += len(batch)
            else:
                consecutive_kw_errors = 0

        except Exception as ex:
            if browser_gone(ex):
                # The Playwright driver or the browser died: every next call would hang, stop now
                raise BrowserGone(str(ex)) from ex
            consecutive_kw_errors += len(batch)
            print(f"      Batch error ({consecutive_kw_errors}/5): {ex}", flush=True)
            continue

    if empty_count > len(keywords) * 0.7:
        print(f"    WARNING: {empty_count}/{len(keywords)} keywords returned 0 tweets — session may be degraded", flush=True)

    new = learn_accounts(all_tweets)
    print(f"    Scrape done: {len(all_tweets)} unique tweets ({new} new accounts to watch, "
          f"{len(_load_discovered())} discovered).", flush=True)
    return all_tweets
