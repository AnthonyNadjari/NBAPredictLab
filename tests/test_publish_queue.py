import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))

NOW = datetime(2026, 10, 20, 22, 0, tzinfo=timezone.utc)


def _setup(tmp_path, monkeypatch, requests):
    from tools import publish_queue as pq
    queue = tmp_path / "publish_queue.json"
    queue.write_text(json.dumps({"requests": requests}))
    monkeypatch.setattr(pq, "QUEUE", queue)
    monkeypatch.setattr(pq, "LOG", tmp_path / "publish_log.json")
    monkeypatch.setattr(pq, "DONE", tmp_path / "state" / "done.txt")
    return pq


def test_each_request_is_handed_out_once_oldest_first(tmp_path, monkeypatch):
    at = NOW.isoformat()
    pq = _setup(tmp_path, monkeypatch, [
        {"id": "1-1", "game_id": "DAL_at_HOU", "texts": ["a", "b"], "requested_at": at},
        {"id": "2-1", "game_id": "announce-season", "dry": True, "requested_at": at},
    ])
    texts = tmp_path / "texts.json"
    assert pq.next_request(texts, NOW) == "1-1 DAL_at_HOU false"
    assert json.loads(texts.read_text(encoding="utf-8")) == ["a", "b"]
    assert pq.next_request(texts, NOW) == "2-1 announce-season true"
    assert texts.read_text(encoding="utf-8") == ""
    assert pq.next_request(texts, NOW) == "none"


def test_stale_or_malformed_requests_are_skipped(tmp_path, monkeypatch):
    old = (NOW - timedelta(hours=13)).isoformat()
    pq = _setup(tmp_path, monkeypatch, [
        {"id": "1-1", "game_id": "DAL_at_HOU", "requested_at": old},           # forgotten: never post late
        {"id": "2-1", "game_id": "x; rm -rf /", "requested_at": NOW.isoformat()},
        {"id": "", "game_id": "ok", "requested_at": NOW.isoformat()},
    ])
    assert pq.next_request(tmp_path / "t.json", NOW) == "none"


def test_log_records_outcome_for_the_panel(tmp_path, monkeypatch):
    pq = _setup(tmp_path, monkeypatch, [])
    res = tmp_path / "thread_result.json"
    res.write_text(json.dumps({"posted": 6, "total": 6, "first_id": "123", "dry_run": False, "error": None}))
    e = pq.log_result("1-1", "DAL_at_HOU", res, NOW)
    assert e["ok"] and e["url"] == "https://x.com/NBAPredictLab/status/123"
    res.write_text(json.dumps({"posted": 0, "total": 6, "error": "X refused the post (duplicate)"}))
    e = pq.log_result("2-1", "DAL_at_HOU", res, NOW)
    assert not e["ok"] and "duplicate" in e["error"]
    e = pq.log_result("3-1", "DAL_at_HOU", tmp_path / "missing.json", NOW)   # publisher crashed
    assert not e["ok"] and e["error"]
    log = json.loads((tmp_path / "publish_log.json").read_text(encoding="utf-8"))
    assert [x["id"] for x in log] == ["1-1", "2-1", "3-1"]
