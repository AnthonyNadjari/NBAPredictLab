import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))
pytest.importorskip("playwright")
import conversations  # noqa: E402


class FakePage:
    def __init__(self, pages):
        self.pages, self.url = pages, None

    def goto(self, url, **kw):
        self.url = url

    def wait_for_selector(self, *a, **kw):
        pass

    def evaluate(self, js):
        return self.pages[self.url]


def art(user, sid, text, mins_ago):
    at = (datetime.now(timezone.utc) - timedelta(minutes=mins_ago)).isoformat().replace("+00:00", "Z")
    return {"user": user, "link": f"/{user}/status/{sid}", "text": text, "at": at}


def test_mentions_filters(monkeypatch):
    monkeypatch.setattr(conversations.time, "sleep", lambda s: None)
    page = FakePage({"https://x.com/notifications/mentions": [
        art("fan1", "1", "nah he was better last year honestly", 30),
        art("NBAPredictLab", "2", "our own tweet here ok", 10),
        art("fan2", "3", "lol", 10),                                   # too short
        art("fan3", "4", "this aged badly my friend trust me", 60 * 8),  # too old
        art("fan4", "5", "already answered this one before ok", 5),
        art("hater", "6", "Ai account based out of France get outta here", 5),
    ]})
    st = {"answered": ["5"], "people": {}, "count": 0}
    got = conversations.mentions(page, st, datetime.now(timezone.utc))
    assert [m["id"] for m in got] == ["1"]
    assert [f["user"] for f in conversations.LAST_FLAGS] == ["hater"] and st["flagged"] == ["6"]


def test_thread_needs_our_reply_just_above(monkeypatch):
    monkeypatch.setattr(conversations.time, "sleep", lambda s: None)
    m = {"user": "fan1", "id": "9"}
    url = "https://x.com/fan1/status/9"
    ours = [art("ShamsCharania", "7", "big news", 90), art("NBAPredictLab", "8", "our take", 60), art("fan1", "9", "nah", 30)]
    assert conversations.thread(FakePage({url: ours}), m)["ours"] == "our take"
    other = [art("ShamsCharania", "7", "big news", 90), art("someone", "8", "x", 60), art("fan1", "9", "nah", 30)]
    assert conversations.thread(FakePage({url: other}), m) is None
