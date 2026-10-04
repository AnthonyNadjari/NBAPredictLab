#!/usr/bin/env python3
"""
Reply lab: run the reply pipeline (facts -> LLM -> fact check -> validator) on
sample tweets and print what the bot WOULD answer. Nothing is posted, no browser.

    LLM_API_KEY=... python vision/tools/reply_lab.py                 # Groq, default model
    LLM_MODEL=llama-3.3-70b-versatile LLM_API_KEY=... python vision/tools/reply_lab.py
    LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen3:4b python vision/tools/reply_lab.py
    python vision/tools/reply_lab.py tweets.txt                       # one tweet per line ("@author: text")
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_client import call_llm  # noqa: E402
from reply_validator import validate_reply  # noqa: E402

SAMPLES = [
    ("ShamsCharania", "Knicks guard Jalen Brunson is expected to play tonight vs. the Celtics after missing two games with an ankle sprain, sources say."),
    ("NBACentral", "Victor Wembanyama tonight: 31 points, 17 rebounds, 6 blocks. Spurs win."),
    ("hoopsfan22", "the lakers are washed and everyone is too scared to say it"),
    ("BleacherReport", "Who's the best team in the West right now? 🤔"),
    ("randomguy", "happy birthday to my brother!! love you man"),
    ("statmuse", "Most 40-point games this season:\n\nSGA — 6\nLuka — 5\nGiannis — 4"),
    ("LegionHoops", "Thunder vs Spurs tonight. Who you got?"),
    ("someone", "this man is HIM"),
]


def main() -> int:
    samples = SAMPLES
    if len(sys.argv) > 1:
        samples = []
        for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                author, _, text = line.partition(":")
                samples.append((author.strip().lstrip("@"), text.strip()))
    session: list[str] = []
    for author, text in samples:
        print("=" * 80)
        print(f"@{author}: {text}")
        r = call_llm(text, author)
        if not r:
            print("  -> LLM error")
            continue
        facts = r.get("facts") or []
        if facts:
            print("  facts given:", *[f"\n    - {f}" for f in facts])
        if (r.get("decision") or "").upper() != "REPLY":
            print(f"  -> SKIP ({r.get('reason')})")
            continue
        reply = (r.get("response") or "").strip()
        ok, why = validate_reply(reply, session, tweet_text="\n".join([text] + facts))
        print(f"  -> {'REPLY' if ok else 'BLOCKED (' + str(why) + ')'}: {reply}"
              + (f"\n     fact used: {r.get('fact_used')}" if r.get("fact_used") else ""))
        if ok:
            session.append(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
