"""
NBAVision Engine — Twitter authentication via cookies with stealth & persistence.
"""
import json
import os
import time
import traceback
from pathlib import Path
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

from config import (
    get_twitter_cookies_json,
    TWITTER_HOME_URL,
    BROWSER_USER_AGENT,
    BROWSER_VIEWPORT,
    STATE_FILE,
    PROJECT_ROOT,
)

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5],
});
window.chrome = { runtime: {}, loadTimes: () => {}, csi: () => {} };
const origQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (params) =>
    params.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : origQuery(params);
"""

LOGS_DIR = PROJECT_ROOT / "logs"


def _ensure_logs_dir() -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    return LOGS_DIR


def _normalize_cookie_domain(domain: str | None) -> str:
    if not domain or not isinstance(domain, str):
        return ".x.com"
    d = domain.strip().lower()
    if d in ("x.com", ".x.com") or d.endswith(".x.com"):
        return ".x.com"
    if d in ("twitter.com", ".twitter.com") or d.endswith(".twitter.com"):
        return ".twitter.com"
    return domain if domain.startswith(".") else "." + domain


def parse_cookies(raw: str) -> list[dict]:
    """Parse TWITTER_COOKIES_JSON into Playwright cookie dicts."""
    if not raw or raw.strip() in ("", "[]"):
        return []
    raw_list = json.loads(raw)
    out = []
    for c in raw_list:
        if not isinstance(c, dict) or not c.get("name") or c.get("value") is None:
            continue
        p = {
            "name": str(c["name"]),
            "value": str(c["value"]),
            "domain": _normalize_cookie_domain(c.get("domain")),
            "path": c.get("path") or "/",
        }
        if c.get("httpOnly") is not None:
            p["httpOnly"] = bool(c["httpOnly"])
        if c.get("secure") is not None:
            p["secure"] = bool(c["secure"])
        exp = c.get("expirationDate") or c.get("expires")
        if exp is not None:
            p["expires"] = int(float(exp))
        if c.get("sameSite"):
            s = str(c["sameSite"]).lower()
            if s == "no_restriction":
                p["sameSite"] = "None"
            elif s in ("strict", "lax", "none"):
                p["sameSite"] = s.capitalize() if s != "none" else "None"
        out.append(p)
    return out


def validate_cookie_expiry(cookies: list[dict]) -> list[str]:
    """Return warnings for critical cookies that are expired or near-expiry."""
    now = time.time()
    warnings = []
    critical = {"auth_token", "ct0", "twid", "kdt"}
    found_critical = set()
    for c in cookies:
        name = c.get("name", "")
        if name in critical:
            found_critical.add(name)
        if name not in critical:
            continue
        exp = c.get("expires") or c.get("expirationDate")
        if exp is None:
            continue
        exp_ts = float(exp)
        if exp_ts < now:
            warnings.append(f"EXPIRED: {name} expired {int(now - exp_ts)}s ago")
        elif exp_ts - now < 86400:
            hours_left = (exp_ts - now) / 3600
            warnings.append(f"EXPIRING SOON: {name} expires in {hours_left:.1f}h")
    missing = critical - found_critical
    if "auth_token" in missing:
        warnings.append("MISSING: auth_token cookie not found")
    if "ct0" in missing:
        warnings.append("MISSING: ct0 cookie not found")
    return warnings


def save_session_state(context: BrowserContext) -> None:
    """Persist cookies + storage after a successful session so next run can reuse them."""
    try:
        context.storage_state(path=str(STATE_FILE))
        print(f"Auth: Session state saved to {STATE_FILE.name}", flush=True)
    except Exception as e:
        print(f"Auth: Could not save session state: {e}", flush=True)


def check_session_alive(page: Page) -> bool:
    """Quick check if the session is still valid (useful mid-run)."""
    try:
        page.goto(TWITTER_HOME_URL, wait_until="domcontentloaded", timeout=20000)
    except Exception:
        return False
    avatar = page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').first
    try:
        avatar.wait_for(state="visible", timeout=8000)
        return True
    except Exception:
        pass
    timeline = page.locator('[data-testid="primaryColumn"]').first
    try:
        timeline.wait_for(state="visible", timeout=5000)
        return True
    except Exception:
        return False


def launch_and_auth() -> tuple:
    """
    1. Launch headless Chromium with stealth patches
    2. Load cookies (prefer persisted state.json, fall back to TWITTER_COOKIES_JSON)
    3. Navigate to X home, verify login
    4. Save updated session state on success
    Returns (playwright, browser, context, page) or (*None, reason_str).
    """
    _ensure_logs_dir()

    raw = get_twitter_cookies_json()
    cookies = parse_cookies(raw)
    if not cookies:
        return None, None, None, None, "no_cookies"

    cookie_warnings = validate_cookie_expiry(cookies)
    for w in cookie_warnings:
        print(f"Auth: {w}", flush=True)

    expired_critical = [w for w in cookie_warnings if w.startswith("EXPIRED")]
    missing_critical = [w for w in cookie_warnings if w.startswith("MISSING")]
    if expired_critical or missing_critical:
        print("Auth: Critical cookies are expired or missing. Re-export from browser.", flush=True)
        return None, None, None, None, "cookies_expired"

    print(f"Auth: Loaded {len(cookies)} cookies.", flush=True)

    pw = sync_playwright().start()
    print("Auth: Launching headless Chromium (stealth)...", flush=True)
    try:
        browser: Browser = pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
    except Exception as e:
        print(f"Auth: Failed to launch Chromium: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        pw.stop()
        return None, None, None, None, "browser_launch_failed"

    use_state = STATE_FILE.is_file()
    context = None

    if use_state:
        print(f"Auth: Restoring session from {STATE_FILE.name}", flush=True)
        try:
            context = browser.new_context(
                storage_state=str(STATE_FILE),
                user_agent=BROWSER_USER_AGENT,
                viewport=BROWSER_VIEWPORT,
                locale="en-US",
            )
        except Exception as e:
            print(f"Auth: state.json invalid ({e}), falling back to raw cookies", flush=True)
            use_state = False
            context = None

    if context is None:
        context = browser.new_context(
            user_agent=BROWSER_USER_AGENT,
            viewport=BROWSER_VIEWPORT,
            locale="en-US",
        )
        context.add_cookies(cookies)

    context.add_init_script(STEALTH_JS)
    page: Page = context.new_page()

    def _check_logged_in() -> bool:
        # is_visible() returns immediately and ignores its timeout, so it would
        # evaluate X's loading spinner before the SPA renders. wait_for() polls
        # and actually blocks until the element is visible (or the timeout).
        avatar = page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').first
        try:
            avatar.wait_for(state="visible", timeout=10000)
            return True
        except Exception:
            pass
        timeline = page.locator('[data-testid="primaryColumn"]').first
        try:
            timeline.wait_for(state="visible", timeout=8000)
            return True
        except Exception:
            return False

    print("Auth: Navigating to X home...", flush=True)
    try:
        page.goto(TWITTER_HOME_URL, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print(f"Auth: Navigation failed: {e}", flush=True)

    if _check_logged_in():
        print("Auth: Session valid (timeline/avatar visible).", flush=True)
        save_session_state(context)
        return pw, browser, context, page

    # If state.json failed, retry with raw cookies
    if use_state:
        print("Auth: state.json session failed, retrying with raw cookies...", flush=True)
        try:
            page.close()
            context.close()
        except Exception:
            pass
        context = browser.new_context(
            user_agent=BROWSER_USER_AGENT,
            viewport=BROWSER_VIEWPORT,
            locale="en-US",
        )
        context.add_cookies(cookies)
        context.add_init_script(STEALTH_JS)
        page = context.new_page()
        try:
            page.goto(TWITTER_HOME_URL, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"Auth: Navigation failed on retry: {e}", flush=True)
        if _check_logged_in():
            print("Auth: Session valid with raw cookies.", flush=True)
            save_session_state(context)
            return pw, browser, context, page

    print("Auth: First load failed, retrying once...", flush=True)
    try:
        page.reload(wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(6000)
    except Exception as e:
        print(f"Auth: Reload failed: {e}", flush=True)
    if _check_logged_in():
        print("Auth: Session valid after retry.", flush=True)
        save_session_state(context)
        return pw, browser, context, page

    # Screenshot for debugging — capture what the browser actually shows
    try:
        ss_path = LOGS_DIR / "auth_failure.png"
        page.screenshot(path=str(ss_path), full_page=True)
        print(f"Auth: Failure screenshot saved to {ss_path}", flush=True)
        page_title = page.title()
        page_url = page.url
        print(f"Auth: Page title='{page_title}', url='{page_url}'", flush=True)
    except Exception as e:
        print(f"Auth: Could not capture screenshot: {e}", flush=True)

    print("Auth: Session invalid or expired.", flush=True)
    try:
        browser.close()
        pw.stop()
    except Exception:
        pass
    return None, None, None, None, "session_invalid"
