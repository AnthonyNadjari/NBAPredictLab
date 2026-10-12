#!/usr/bin/env python3
"""Set the profile bio through the logged-in browser: python tools/set_bio.py "new bio" (prints the old one)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auth import launch_and_auth  # noqa: E402


def main() -> int:
    bio = sys.argv[1].strip()
    if not bio or len(bio) > 160:
        print("bio must be 1-160 characters")
        return 1
    pw, _, context, page, *err = launch_and_auth()
    if context is None:
        print(f"login failed: {err}")
        return 1
    try:
        page.goto("https://x.com/settings/profile", wait_until="domcontentloaded", timeout=45000)
        box = page.locator('textarea[name="description"]').first
        box.wait_for(state="visible", timeout=30000)
        print(f"old bio: {box.input_value()!r}")
        box.fill(bio)
        time.sleep(1.5)
        page.locator('[data-testid="Profile_Save_Button"]').first.click(timeout=15000)
        time.sleep(5)
        print(f"new bio: {bio!r}")
        return 0
    finally:
        context.close()
        pw.stop()


if __name__ == "__main__":
    sys.exit(main())
