#!/usr/bin/env python3
"""
Log the reply bot into X once, on the PC that hosts the GitHub Actions runner.

Opens a visible Chrome window on the bot's persistent profile. Log in by hand
(2FA included); the window closes by itself once the home timeline shows.
Every later run reuses this profile, so cookies stay fresh with the same IP and
browser fingerprint.

    python vision/tools/connect_x.py                 # log in / check
    python vision/tools/connect_x.py --update-secret # also store cookies as the
                                                     # TWITTER_COOKIES_JSON backup secret
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright  # noqa: E402

from auth import is_logged_in, logged_in_handle, open_profile  # noqa: E402
from config import PROFILE_DIR, TWITTER_HOME_URL  # noqa: E402

REPO = "AnthonyNadjari/NBAPredictLab"
WAIT_MINUTES = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-secret", action="store_true", help="push cookies to the repo secret (needs gh)")
    args = ap.parse_args()

    with sync_playwright() as pw:
        ctx = open_profile(pw, headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(TWITTER_HOME_URL, wait_until="domcontentloaded")
        if not is_logged_in(page, 8000):
            print(f"Log in to X in the browser window (waiting up to {WAIT_MINUTES} min)...")
            page.goto("https://x.com/i/flow/login", wait_until="domcontentloaded")
            deadline = time.time() + WAIT_MINUTES * 60
            while time.time() < deadline:
                if "/home" in page.url and is_logged_in(page, 3000):
                    break
                time.sleep(2)
            else:
                print("Timed out: not logged in.")
                ctx.close()
                return 1
        handle = logged_in_handle(page) or "?"
        print(f"Logged in as {handle}. Profile saved in {PROFILE_DIR}")

        if args.update_secret:
            cookies = [c for c in ctx.cookies() if "x.com" in c["domain"] or "twitter.com" in c["domain"]]
            payload = json.dumps(cookies, separators=(",", ":"))
            r = subprocess.run(["gh", "secret", "set", "TWITTER_COOKIES_JSON", "-R", REPO],
                               input=payload, text=True, capture_output=True)
            print("Backup secret updated." if r.returncode == 0 else f"gh failed: {r.stderr.strip()}")
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
