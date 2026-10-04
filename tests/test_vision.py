import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

VISION = Path(__file__).resolve().parents[1] / "vision"
sys.path.insert(0, str(VISION))


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NBAVISION_STATE_DIR", str(tmp_path / "state"))


def test_schedule_gate_fires_once_in_window():
    import config
    from tools.schedule_gate import due_slot
    sched = {"schedules": [{"time": "23:00", "enabled": True}, {"time": "01:00", "enabled": False}]}
    tz = config.TZ
    assert due_slot(datetime(2026, 10, 21, 23, 5, tzinfo=tz), sched)
    assert due_slot(datetime(2026, 10, 21, 22, 59, tzinfo=tz), sched) is None
    assert due_slot(datetime(2026, 10, 21, 23, 15, tzinfo=tz), sched) is None
    assert due_slot(datetime(2026, 10, 22, 1, 5, tzinfo=tz), sched) is None  # disabled slot


def test_merge_status_keeps_both_sides(tmp_path, monkeypatch):
    from tools import merge_status
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "stats.json").write_text(json.dumps([{"date": "2026-10-01", "followers": 150}]))
    (docs / "runs.json").write_text(json.dumps([{"run_id": "1", "at": "a"}]))
    src = tmp_path / "src"
    src.mkdir()
    (src / "stats.json").write_text(json.dumps([{"date": "2026-10-02", "followers": 155}]))
    (src / "runs.json").write_text(json.dumps([{"run_id": "2", "at": "b"}]))
    monkeypatch.setattr(merge_status, "ROOT", docs)
    merge_status.main(str(src))
    stats = json.loads((docs / "stats.json").read_text())
    runs = json.loads((docs / "runs.json").read_text())
    assert [s["followers"] for s in stats] == [150, 155]
    assert {r["run_id"] for r in runs} == {"1", "2"}


def test_context_detection_and_facts(monkeypatch):
    import nba_context as c
    monkeypatch.setattr(c, "_rosters", lambda: {"jalen brunson": ["Jalen Brunson", "NYK"],
                                                "jalen williams": ["Jalen Williams", "OKC"]})
    monkeypatch.setattr(c, "_history", lambda: [
        {"game_date": "2026-10-20", "season": "2026-27", "season_type": "Regular Season",
         "home": "NYK", "away": "PHI", "home_pts": "110", "away_pts": "100"},
        {"game_date": "2026-10-22", "season": "2026-27", "season_type": "Regular Season",
         "home": "BOS", "away": "NYK", "home_pts": "99", "away_pts": "105"},
    ])
    monkeypatch.setattr(c, "_pending", lambda: [])
    teams, players = c.detect("Brunson and the Knicks rolling, Williams quiet")
    assert teams == {"NYK"}
    assert players == {"Jalen Brunson": "NYK"}  # "Williams" alone is ambiguous: ignored
    facts = c.facts_for("Knicks are for real")
    assert "New York Knicks are 2-0 in the 2026-27 regular season" in facts
    assert "New York Knicks have won 2 straight" in facts


def test_validator_accepts_names_from_facts():
    from reply_validator import validate_reply
    ok, _ = validate_reply("Knicks are 2-0 and Brunson looks locked in", [],
                           tweet_text="who is the best team in the east\nNew York Knicks are 2-0")
    bad, reason = validate_reply("Celtics would never let that happen", [], tweet_text="who is the best team in the east")
    assert ok and not bad and reason == "reply_adds_entity_not_in_tweet"


def test_x_account_record_followers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src import x_account
    p = tmp_path / "stats.json"
    p.write_text(json.dumps([{"date": "2026-01-01", "followers": 10}]))
    x_account.record_followers(12, 5, path=p)
    x_account.record_followers(13, 5, path=p)  # same day: overwritten
    data = json.loads(p.read_text())
    assert len(data) == 2 and data[-1]["followers"] == 13
