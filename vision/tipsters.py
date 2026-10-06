#!/usr/bin/env python3
"""
Tipster tape: what the betting accounts on X are picking (run by the server tick between reply
sessions, every 3 h, with the logged-in browser).

1. Live X searches for NBA pick posts ("best bet", "POTD", "NBA picks"...), engagement-filtered.
2. The LLM turns each post into structured picks: game, side, market (moneyline / spread / total /
   player prop), line, odds, units. Posts that are not picks are dropped.
3. Appended to TAPE_DIR/tipsters/<UTC date>.jsonl (one line per pick, with the author, likes and
   tweet id). Later: is the tipster consensus a signal against the result and the closing line,
   and which accounts are actually good? (research, once a few weeks of data exist).
Nothing is posted; nothing is pushed to git.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import get_llm_api_key, get_llm_model
from scraper import _scrape_single_tab

TAPE_DIR = Path(os.getenv("NBA_TAPE_DIR", "/opt/nba-vision/tape")) / "tipsters"
SEEN = TAPE_DIR / "seen_ids.json"
QUERIES = [
    'NBA ("best bet" OR "lock of the day" OR POTD OR "play of the day")',
    '"NBA picks" OR "NBA pick" OR "NBA bets"',
    'NBA (moneyline OR "ML" OR spread) tonight (units OR "1u" OR "2u")',
    'NBA (over OR under) (points OR rebounds OR assists) "prop"',
]
PARSE_PROMPT = """You read posts from sports-betting accounts on X and extract their NBA picks.
Return JSON: {"picks": [{"i": <post index>, "game": "AWAY @ HOME team names if known", "pick": "the side/selection as written",
"market": "moneyline|spread|total|player_prop|parlay|other", "line": number or null, "odds": "as written or null",
"units": number or null, "game_date": "YYYY-MM-DD if stated, else null"}]}
Only real NBA picks the author is making (not recaps of past results, not ads without a pick, not other sports).
A post can contain several picks. No pick in a post: emit nothing for it. Do not invent missing fields."""


def _llm(posts: list[dict]) -> list[dict]:
    key, model = get_llm_api_key(), get_llm_model()
    if not key or not posts:
        return []
    url = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/") + "/chat/completions"
    body = "\n\n".join(f"[{i}] @{p['username']}: {p['text'][:600]}" for i, p in enumerate(posts))
    for attempt in range(4):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, timeout=90, json={
                "model": model, "temperature": 0, "max_tokens": 4000,
                **({"reasoning_effort": "low"} if "gpt-oss" in model else {}),
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": PARSE_PROMPT}, {"role": "user", "content": body}]})
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            js = json.loads(r.json()["choices"][0]["message"]["content"])
            return [x for x in js.get("picks", []) if isinstance(x, dict) and isinstance(x.get("i"), int)]
        except Exception as e:
            print(f"Tipsters: LLM parse failed ({e})", flush=True)
            time.sleep(10)
    return []


def run(page) -> dict:
    TAPE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        seen = set(json.loads(SEEN.read_text(encoding="utf-8")))
    except Exception:
        seen = set()
    posts = {}
    for q in QUERIES:
        for t in _scrape_single_tab(page, q):
            if t.get("tweet_id") and t["tweet_id"] not in seen and len(t.get("text") or "") > 20:
                posts[t["tweet_id"]] = t
        time.sleep(random.uniform(4, 9))
    posts = list(posts.values())
    picks_out, now = [], datetime.now(timezone.utc)
    for start in range(0, len(posts), 12):                          # 12 posts per LLM call
        chunk = posts[start:start + 12]
        for p in _llm(chunk):
            if 0 <= p["i"] < len(chunk):
                src = chunk[p.pop("i")]
                picks_out.append({**p, "author": src["username"], "likes": src.get("likes"), "tweet_id": src["tweet_id"],
                                  "posted": src.get("timestamp"), "seen_at": now.isoformat(timespec="seconds"),
                                  "text": (src.get("text") or "")[:400]})
        time.sleep(3)
    if picks_out:
        with open(TAPE_DIR / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            for p in picks_out:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    SEEN.write_text(json.dumps(sorted(seen | {p["tweet_id"] for p in posts})[-20000:]), encoding="utf-8")
    return {"posts": len(posts), "picks": len(picks_out)}


def main() -> int:
    from auth import launch_and_auth
    res = launch_and_auth()
    if res[0] is None:
        print(f"Tipsters: not logged in ({res[-1]})", flush=True)
        return 1
    pw, _, context, page = res
    try:
        print("Tipsters: " + json.dumps(run(page)), flush=True)
        return 0
    finally:
        try:
            context.close()
            pw.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
