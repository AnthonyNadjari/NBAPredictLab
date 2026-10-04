"""
NBAVision Engine — Scrape bot profile for follower/following count and update daily stats.
Run at the beginning of each run; one value per day (overwritten if same day).
"""
import json
import re
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Page

from config import BOT_PROFILE_USERNAME, STATS_FILE, TZ


def _parse_count(text: str) -> int | None:
    """Parse '1,234', '1.2K', '1.2M', '12.5K' etc. to int."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip().replace(",", "").replace(" ", "")
    if not text:
        return None
    text_upper = text.upper()
    mult = 1
    if text_upper.endswith("K"):
        mult = 1000
        text = text[:-1].strip()
    elif text_upper.endswith("M"):
        mult = 1_000_000
        text = text[:-1].strip()
    try:
        return int(float(text) * mult)
    except (ValueError, TypeError):
        return None


def _extract_number_from_text(text: str) -> int | None:
    """Pull leading number from strings like '276 Following', '1,234 Followers', '12.5K'."""
    if not text:
        return None
    m = re.match(r"([\d,.]+)\s*([KkMm])?", text.strip())
    if not m:
        return None
    raw = m.group(1)
    suffix = (m.group(2) or "").upper()
    return _parse_count(raw + suffix)


JS_EXTRACT = """
() => {
    const result = {followers: null, following: null, debug: []};
    const links = document.querySelectorAll('a[href]');
    for (const a of links) {
        const href = a.getAttribute('href') || '';
        const aria = a.getAttribute('aria-label') || '';
        const text = (a.innerText || '').trim();
        // Check followers (href contains /followers or /verified_followers)
        if (/\\/(?:verified_)?followers$/i.test(href)) {
            result.debug.push({type: 'followers_link', href, aria, text});
            // Try aria-label first: "1,234 Followers"
            const am = aria.match(/([\d,.]+)\s*([KkMm])?\s*[Ff]ollowers/i);
            if (am) {
                result.followers = am[0];
            } else if (text) {
                result.followers = text;
            }
        }
        // Check following (href ends with /following)
        if (/\\/following$/i.test(href)) {
            result.debug.push({type: 'following_link', href, aria, text});
            const am = aria.match(/([\d,.]+)\s*([KkMm])?\s*[Ff]ollowing/i);
            if (am) {
                result.following = am[0];
            } else if (text) {
                result.following = text;
            }
        }
    }
    // Fallback: look for any element with aria-label containing Followers/Following
    if (!result.followers) {
        const els = document.querySelectorAll('[aria-label*="Follower"]');
        for (const el of els) {
            const aria = el.getAttribute('aria-label') || '';
            result.debug.push({type: 'aria_follower_el', aria, tag: el.tagName});
            const am = aria.match(/([\d,.]+)\s*([KkMm])?\s*[Ff]ollowers/i);
            if (am) { result.followers = am[0]; break; }
        }
    }
    if (!result.following) {
        const els = document.querySelectorAll('[aria-label*="Following"]');
        for (const el of els) {
            const aria = el.getAttribute('aria-label') || '';
            result.debug.push({type: 'aria_following_el', aria, tag: el.tagName});
            const am = aria.match(/([\d,.]+)\s*([KkMm])?\s*[Ff]ollowing/i);
            if (am) { result.following = am[0]; break; }
        }
    }
    return result;
}
"""


def scrape_profile_stats(page: Page, username: str) -> dict | None:
    """
    Navigate to profile, scrape followers (and following). Returns {'followers': int, 'following': int} or None.
    """
    url = f"https://x.com/{username}"
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(3000)
    except Exception as e:
        print(f"Profile stats: failed to load profile: {e}", flush=True)
        return None

    result = {"followers": None, "following": None}

    # Strategy 1: JavaScript DOM extraction (most reliable)
    try:
        js_result = page.evaluate(JS_EXTRACT)
        print(f"Profile stats JS debug: {json.dumps(js_result.get('debug', []))}", flush=True)

        raw_followers = js_result.get("followers")
        raw_following = js_result.get("following")

        if raw_followers:
            val = _extract_number_from_text(str(raw_followers))
            if val is not None:
                result["followers"] = val
                print(f"Profile stats: JS extracted followers={val} from '{raw_followers}'", flush=True)
        if raw_following:
            val = _extract_number_from_text(str(raw_following))
            if val is not None:
                result["following"] = val
                print(f"Profile stats: JS extracted following={val} from '{raw_following}'", flush=True)
    except Exception as e:
        print(f"Profile stats: JS extraction failed: {e}", flush=True)

    # Strategy 2: Playwright locators as fallback
    if result["followers"] is None or result["following"] is None:
        for label in ["followers", "verified_followers", "following"]:
            key = "following" if label == "following" else "followers"
            if result[key] is not None:
                continue
            try:
                link = page.locator(f'a[href$="/{label}"]').first
                if link.count() == 0:
                    continue
                aria = link.get_attribute("aria-label") or ""
                text = link.inner_text() or ""
                print(f"Profile stats: locator a[href$='/{label}'] aria='{aria}' text='{text}'", flush=True)
                val = _extract_number_from_text(aria) or _extract_number_from_text(text.split("\n")[0])
                if val is not None:
                    result[key] = val
            except Exception:
                pass

    if result["followers"] is None and result["following"] is None:
        print("Profile stats: could not find followers/following on profile", flush=True)
        return None
    return result


def load_stats() -> list[dict]:
    """Load stats from STATS_FILE. Returns list of {date, followers, following?}."""
    if not STATS_FILE.is_file():
        return []
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Profile stats: could not load {STATS_FILE}: {e}", flush=True)
        return []
    if not isinstance(data, list):
        return []
    return data


def save_stats(entries: list[dict]) -> None:
    """Write stats to STATS_FILE. Ensures docs/ exists."""
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"Profile stats: saved to {STATS_FILE}", flush=True)


def update_today(entries: list[dict], followers: int | None, following: int | None) -> list[dict]:
    """Update or append today's entry. One value per day (last run of the day wins)."""
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    new_entry = {"date": today, "followers": followers, "following": following}
    out = [e for e in entries if isinstance(e, dict) and e.get("date") != today]
    out.append(new_entry)
    out.sort(key=lambda e: e.get("date") or "")
    return out


def run_at_start(page: Page) -> None:
    """
    Scrape profile stats and update docs/stats.json with today's value.
    No-op if BOT_PROFILE_USERNAME is not set.
    """
    if not BOT_PROFILE_USERNAME:
        return
    print(f"Profile stats: scraping @{BOT_PROFILE_USERNAME}...", flush=True)
    stats = scrape_profile_stats(page, BOT_PROFILE_USERNAME)
    if not stats:
        return
    print(f"Profile stats: followers={stats.get('followers')}, following={stats.get('following')}", flush=True)
    entries = load_stats()
    entries = update_today(entries, stats.get("followers"), stats.get("following"))
    save_stats(entries)
