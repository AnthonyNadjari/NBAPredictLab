#!/usr/bin/env python3
"""
Convert a Netscape-format cookies file (cookies.txt) to the JSON array
expected by TWITTER_COOKIES_JSON. Keeps only x.com / .x.com / twitter.com cookies.

Usage:
  python scripts/netscape_cookies_to_json.py path/to/cookies.txt

Output: JSON array to stdout. Copy it and paste into GitHub → Settings →
Secrets and variables → Actions → TWITTER_COOKIES_JSON (New/Update secret).

Do not commit cookies.txt or the output to the repo.
"""

import json
import sys
from pathlib import Path


# Domains we keep (Twitter/X)
ALLOWED_DOMAINS = (".x.com", "x.com", ".twitter.com", "twitter.com")


def _domain_ok(domain: str) -> bool:
    d = (domain or "").strip().lower()
    return d in ALLOWED_DOMAINS or d.endswith(".x.com") or d.endswith(".twitter.com")


def parse_netscape(path: Path) -> list[dict]:
    """
    Parse Netscape cookie format:
    domain  flag  path  secure  expiration  name  value
    (tab-separated; value is rest of line, may contain tabs)
    """
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            domain = parts[0].strip()
            if not _domain_ok(domain):
                continue
            path_str = (parts[2] or "/").strip()
            secure_flag = (parts[3] or "").strip().upper() == "TRUE"
            try:
                expiration = int(float(parts[4]))
            except (ValueError, IndexError):
                expiration = 0
            name = (parts[5] or "").strip()
            value = "\t".join(parts[6:]).strip()
            if not name:
                continue
            # Normalize domain for our app (auth.py uses .x.com / .twitter.com)
            if domain in ("x.com", "twitter.com") and not domain.startswith("."):
                domain = "." + domain
            out.append({
                "name": name,
                "value": value,
                "domain": domain,
                "path": path_str,
                "expirationDate": expiration,
                "secure": secure_flag,
            })
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/netscape_cookies_to_json.py <cookies.txt>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    cookies = parse_netscape(path)
    if not cookies:
        print("No x.com / twitter.com cookies found in the file.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(cookies, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
