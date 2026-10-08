#!/usr/bin/env python3
"""Replay real tweets through the CURRENT reply pipeline (LLM + facts + validator), posting nothing.

    python vision/tools/replay_lab.py OUT.jsonl [--limit N]

Input: every tweet we already answered (docs/vision/replies/*.json), so old and new replies can be
compared on the same tweets. Output, one line per tweet: tweet, old reply, new decision/reason,
new reply, facts given, fact used, validator verdict.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPO_ROOT  # noqa: E402
from llm_client import call_llm, verify_reply  # noqa: E402
import os  # noqa: E402
from reply_validator import validate_reply  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = []
    for f in sorted((REPO_ROOT / "docs" / "vision" / "replies").glob("20*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    seen, tweets = set(), []
    for r in rows:
        if r.get("tweet_text") and r["tweet_text"] not in seen:
            seen.add(r["tweet_text"])
            tweets.append(r)
    if a.limit:
        tweets = tweets[: a.limit]
    done = set()
    out = Path(a.out)
    if out.exists():
        done = {json.loads(l)["tweet_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    print(f"{len(tweets)} tweets, {len(done)} already done", flush=True)
    for i, r in enumerate(tweets):
        if r.get("tweet_id") in done:
            continue
        res = call_llm(r["tweet_text"], r.get("author") or "") or {}
        resp = (res.get("response") or "").strip()
        valid, why = (None, None)
        if "REPLY" in (res.get("decision") or "").upper() and resp:
            ctx = "\n".join([r["tweet_text"]] + list(res.get("facts") or []))
            valid, why = validate_reply(resp, [], tweet_text=ctx)
            if valid and os.getenv("REPLAY_VERIFY") == "1":       # second pass, as in production
                ok, vwhy = verify_reply(r["tweet_text"], res.get("facts") or [], resp)
                if not ok:
                    valid, why = False, "fact_check: " + vwhy
        line = {"tweet_id": r.get("tweet_id"), "author": r.get("author"), "tweet": r["tweet_text"],
                "old_reply": r.get("reply_text"), "old_grade": r.get("auto_grade"),
                "decision": res.get("decision"), "reason": res.get("reason"), "reply": resp,
                "facts": res.get("facts"), "fact_used": res.get("fact_used"), "valid": valid, "invalid_why": why}
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"[{i + 1}/{len(tweets)}] {line['decision']} {('' if valid in (None, True) else 'BLOCKED ' + str(why))}", flush=True)
        # Groq free tier: 8,000 tokens/min shared with the live bot (~2,500 per call)
        time.sleep(float(__import__('os').getenv('REPLAY_DELAY', '20')) + random.uniform(0, 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
