import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)


def _ledger(tmp_path, monkeypatch):
    import replies_ledger as L
    monkeypatch.setattr(L, "LEDGER_DIR", tmp_path / "replies")
    return L


def test_parse_metrics_from_x_aria_label():
    import replies_ledger as L
    assert L.parse_metrics("3 replies, 1 repost, 12 likes, 2 bookmarks, 1,234 views") == \
        {"replies": 3, "reposts": 1, "likes": 12, "bookmarks": 2, "views": 1234}
    assert L.parse_metrics("1 reply, 1.2K views")["views"] == 1200
    assert L.parse_metrics("")["likes"] == 0


def test_add_snapshot_grade_and_follow(tmp_path, monkeypatch):
    L = _ledger(tmp_path, monkeypatch)
    posted = (NOW - timedelta(hours=30)).isoformat()
    e = L.add({"tweet_url": "https://x.com/a/status/1", "author": "TheAuthor", "reply_text": "Hard to argue.",
               "reply_id": "99", "posted_at": posted, "reason": "agree"}, run_id="r1")
    assert e["id"] == "99" and e["reply_url"].endswith("/99")
    assert L.grade(e) is None                              # nothing measured yet
    L.snapshot(e, {"likes": 0, "replies": 0, "views": 40}, NOW)
    assert "h24" in e["metrics"] and "h72" not in e["metrics"]
    assert L.grade(e) == "bad"
    L.snapshot(e, {"likes": 2, "replies": 0, "views": 300}, NOW + timedelta(hours=1))
    assert e["metrics"]["h24"]["likes"] == 0 and L.grade(e) == "normal"   # h24 frozen, grade uses latest
    assert L.attribute_follows([e], {"@theauthor"}, NOW) == 1 and L.grade(e) == "good_follow"
    files = L.recent(now=NOW)
    assert [x["id"] for es in files.values() for x in es] == ["99"]


def test_export_for_the_panel(tmp_path, monkeypatch):
    L = _ledger(tmp_path, monkeypatch)
    L.add({"tweet_url": "https://x.com/a/status/1", "reply_text": "x", "posted_at": NOW.isoformat()})
    out = L.export(tmp_path / "docs")
    assert [p.name for p in out] == ["2026-10.json"]
    assert json.loads((tmp_path / "docs" / "index.json").read_text()) == ["2026-10.json"]
    # no reply id captured: a pending id the tracker later replaces by matching the text
    assert json.loads(out[0].read_text(encoding="utf-8"))[0]["id"] == "pending-1"
