#!/usr/bin/env python3
"""Automatic checks on a replay_lab.py output: numbers the reply asserts that are in neither the
tweet nor the facts given, correction phrasing, length, skip/block rates, old vs new.

    python vision/tools/replay_check.py replay.jsonl
"""
import json
import re
import sys
from collections import Counter

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:[-–]\d{1,3})?%?|\d+\.\d+)(?![\w.])")
CORRECTION = re.compile(r"(?i)\b(actually|last i (saw|checked)|not a clean|roster says|rumou?r|that's not|not quite|wrong)\b")


def nums(s):
    return {n.replace("–", "-") for n in NUM.findall(s or "") if not re.fullmatch(r"[0-9]", n)}


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rep = [r for r in rows if "REPLY" in (r.get("decision") or "").upper() and r.get("reply")]
    ok = [r for r in rep if r.get("valid")]
    print(f"{len(rows)} tweets | new: {len(rep)} replies drafted, {len(ok)} pass the validator, "
          f"{len(rows) - len(rep)} skipped")
    print("skip reasons:", Counter(r.get("reason") for r in rows if r not in rep).most_common(8))
    print("blocked by validator:", Counter(r.get("invalid_why") for r in rep if not r.get("valid")).most_common(6))
    L = [len(r["reply"]) for r in ok]
    Lo = [len(r["old_reply"] or "") for r in rows if r.get("old_reply")]
    print(f"length: new avg {sum(L) / max(len(L), 1):.0f} chars, old avg {sum(Lo) / max(len(Lo), 1):.0f}")
    flags = []
    for r in ok:
        support = r["tweet"] + " " + " ".join(r.get("facts") or [])
        extra = nums(r["reply"]) - nums(support)
        if extra:
            flags.append(("unsupported number " + ",".join(sorted(extra)), r))
        if CORRECTION.search(r["reply"]):
            flags.append(("correction phrasing", r))
    print(f"\n{len(flags)} flagged replies (of {len(ok)} that would be posted):")
    for why, r in flags:
        print(f"- [{why}] @{r['author']}: {r['tweet'][:110]!r}\n    -> {r['reply']!r}\n    facts: {r.get('facts')}")


if __name__ == "__main__":
    main(sys.argv[1])
