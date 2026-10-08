#!/usr/bin/env python3
"""Run the second-pass fact check (llm_client.verify_reply) on replies already drafted by replay_lab.py.

    python vision/tools/verify_replay.py REPLAY.jsonl OUT.jsonl
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_client import verify_reply  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
todo = [r for r in rows if r.get("valid")]
print(f"{len(todo)} replies to check", flush=True)
with open(out, "w", encoding="utf-8") as fh:
    for i, r in enumerate(todo):
        ok, why = verify_reply(r["tweet"], r.get("facts") or [], r["reply"])
        fh.write(json.dumps({"tweet": r["tweet"], "reply": r["reply"], "ok": ok, "why": why}, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"[{i + 1}/{len(todo)}] {'OK ' if ok else 'REJECT'} {r['reply'][:70]!r} {'' if ok else '-> ' + why[:120]}", flush=True)
        time.sleep(8 + random.uniform(0, 2))
