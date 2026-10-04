#!/usr/bin/env python3
"""Merge a session's stats.json / runs.json (copied to DIR) into the checked-out docs/vision files.

stats.json: one entry per date (newest value wins). runs.json: union by (run_id, at).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "docs" / "vision"


def load(p: Path) -> list:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:
        return []


def main(src: str) -> None:
    src = Path(src)
    stats = {e["date"]: e for e in load(ROOT / "stats.json") if e.get("date")}
    for e in load(src / "stats.json"):
        if e.get("date") and e.get("followers") is not None:
            stats[e["date"]] = {**stats.get(e["date"], {}), **e}
    (ROOT / "stats.json").write_text(json.dumps(sorted(stats.values(), key=lambda e: e["date"]), indent=2),
                                     encoding="utf-8")
    runs = {(r.get("run_id"), r.get("at")): r for r in load(ROOT / "runs.json")}
    for r in load(src / "runs.json"):
        runs[(r.get("run_id"), r.get("at"))] = r
    merged = sorted(runs.values(), key=lambda r: r.get("at") or "")[-300:]
    (ROOT / "runs.json").write_text(json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
