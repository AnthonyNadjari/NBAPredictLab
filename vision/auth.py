"""
NBAVision Engine — Twitter authentication via cookies with stealth & persistence.
"""
from __future__ import annotations
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
    PROFILE_DIR,
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
    return is_logged_in(page, 10000)


def _chrome_channel() -> str | None:
    """Use the real installed Chrome when present (more human fingerprint than bundled Chromium)."""
    ch = os.getenv("NBAVISION_BROWSER_CHANNEL", "chrome").strip()
    return ch or None


def _real_user_agent(pw, channel: str | None) -> str:
    """Desktop UA matching the actual browser version (headless Chrome advertises 'HeadlessChrome')."""
    try:
        b = pw.chromium.launch(headless=True, channel=channel)
        version = b.version
        b.close()
        return ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                f"(KHTML, like Gecko) Chrome/{version} Safari/537.36")
    except Exception:
        return BROWSER_USER_AGENT


def open_profile(pw, headless: bool = True) -> BrowserContext:
    """Persistent browser profile shared by every run on this machine (and by tools/connect_x.py)."""
    channel = _chrome_channel()
    kwargs = dict(
        headless=headless,
        viewport=BROWSER_VIEWPORT,
        locale="en-US",
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"],
    )
    if headless:
        kwargs["user_agent"] = _real_user_agent(pw, channel)
    try:
        return pw.chromium.launch_persistent_context(str(PROFILE_DIR), channel=channel, **kwargs)
    except Exception as e:
        if channel:
            print(f"Auth: channel '{channel}' unavailable ({e}); using bundled Chromium", flush=True)
            return pw.chromium.launch_persistent_context(str(PROFILE_DIR), **kwargs)
        raise


def is_logged_in(page: Page, timeout_ms: int = 10000) -> bool:
    """True when the logged-in side nav is visible (the logged-out page has no account switcher)."""
    try:
        page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').first.wait_for(
            state="visible", timeout=timeout_ms)
        return True
    except Exception:
        return False


def logged_in_handle(page: Page) -> str | None:
    """@handle of the logged-in account, read from the side nav."""
    try:
        txt = page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').first.inner_text(timeout=3000)
        for part in txt.split():
            if part.startswith("@"):
                return part
    except Exception:
        pass
    return None


def launch_and_auth() -> tuple:
    """
    1. Open the persistent profile (stays logged in between runs, same IP/fingerprint)
    2. If logged out, inject TWITTER_COOKIES_JSON as a fallback and retry
    3. Return (playwright, None, context, page) or (None, None, None, None, reason)
    """
    _ensure_logs_dir()
    pw = sync_playwright().start()
    try:
        context = open_profile(pw, headless=True)
    except Exception as e:
        print(f"Auth: Failed to launch browser: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        pw.stop()
        return None, None, None, None, "browser_launch_failed"
    context.add_init_script(STEALTH_JS)
    page: Page = context.pages[0] if context.pages else context.new_page()

    def _goto_home():
        try:
            page.goto(TWITTER_HOME_URL, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"Auth: Navigation failed: {e}", flush=True)

    print(f"Auth: Opening persistent profile {PROFILE_DIR}", flush=True)
    _goto_home()
    if is_logged_in(page, 15000):
        print(f"Auth: Session valid via profile ({logged_in_handle(page) or '?'}).", flush=True)
        save_session_state(context)
        return pw, None, context, page

    cookies = parse_cookies(get_twitter_cookies_json())
    reason = "session_invalid"
    if not cookies:
        reason = "no_cookies"
    else:
        warnings = validate_cookie_expiry(cookies)
        for w in warnings:
            print(f"Auth: {w}", flush=True)
        if any(w.startswith(("EXPIRED", "MISSING")) for w in warnings):
            reason = "cookies_expired"
        else:
            print(f"Auth: Profile logged out; injecting {len(cookies)} cookies from secret...", flush=True)
            context.add_cookies(cookies)
            _goto_home()
            if is_logged_in(page, 15000):
                print(f"Auth: Session valid with secret cookies ({logged_in_handle(page) or '?'}).", flush=True)
                save_session_state(context)
                return pw, None, context, page

    try:
        ss_path = LOGS_DIR / "auth_failure.png"
        page.screenshot(path=str(ss_path), full_page=True)
        print(f"Auth: Failure screenshot saved to {ss_path} (title='{page.title()}', url='{page.url}')", flush=True)
    except Exception as e:
        print(f"Auth: Could not capture screenshot: {e}", flush=True)
    print("Auth: Not logged in. Run `python vision/tools/connect_x.py` on the runner PC to log in once.", flush=True)
    try:
        context.close()
        pw.stop()
    except Exception:
        pass
    return None, None, None, None, reason
