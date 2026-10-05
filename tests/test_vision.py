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


def test_schedule_gate_window():
    import config
    from tools.schedule_gate import due_slot
    sched = {"schedules": [{"time": "23:00", "enabled": True}, {"time": "01:00", "enabled": False}]}
    tz = config.TZ
    assert due_slot(datetime(2026, 10, 21, 23, 5, tzinfo=tz), sched)
    assert due_slot(datetime(2026, 10, 21, 22, 59, tzinfo=tz), sched) is None
    assert due_slot(datetime(2026, 10, 21, 23, 40, tzinfo=tz), sched)  # GitHub cron runs late
    assert due_slot(datetime(2026, 10, 21, 23, 46, tzinfo=tz), sched) is None
    assert due_slot(datetime(2026, 10, 22, 1, 5, tzinfo=tz), sched) is None  # disabled slot


def test_schedule_gate_runs_each_slot_once(tmp_path, monkeypatch):
    from tools import schedule_gate
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps([{"slot": "2026-10-21T23:00:00+02:00"}]))
    monkeypatch.setattr(schedule_gate, "RUNS_FILE", runs)
    assert "2026-10-21T23:00:00+02:00" in schedule_gate.done_slots()


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
    facts = c.facts_for("Knicks have the best record in the East")
    # test data dates are 2026-10-20/22: recent relative to the 2026-27 season
    monkeypatch.setattr(c, "_now", lambda: datetime(2026, 10, 25, 12, 0))
    facts = c.facts_for("Knicks have the best record in the East")
    assert "New York Knicks are 2-0 so far this season (2026-27)" in facts
    assert "New York Knicks have won 2 straight" in facts
    # in the offseason the same record is phrased as last season, without stale form lines
    monkeypatch.setattr(c, "_now", lambda: datetime(2027, 9, 1, 12, 0))
    facts = c.facts_for("Knicks have the best record in the East")
    assert facts == ["New York Knicks finished last season (2026-27) 2-0"]


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


class _Resp:
    def __init__(self, payload, status=200, text=""):
        self._p, self.status_code, self.text, self.headers = payload, status, text, {}

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)


def _llm_reply(obj):
    return _Resp({"choices": [{"message": {"content": json.dumps(obj)}}]})


def test_llm_client_injects_facts_and_rejects_invented_ones(monkeypatch):
    import llm_client
    import nba_context
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setattr(nba_context, "facts_for", lambda t: ["New York Knicks are 2-0 so far this season (2026-27)"])
    monkeypatch.setattr(nba_context, "today_line", lambda: "Today is 2026-10-25.")
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["body"] = json
        return _llm_reply({"decision": "REPLY", "reason": "agree", "response": "2-0 and rolling.",
                           "fact_used": "New York Knicks are 2-0 so far this season (2026-27)"})
    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    r = llm_client.call_llm("Knicks look good", "fan")
    assert r["decision"] == "REPLY" and r["context_used"]
    user_msg = sent["body"]["messages"][1]["content"]
    assert "VERIFIED FACTS" in user_msg and "2-0 so far this season" in user_msg
    assert sent["body"]["response_format"] == {"type": "json_object"}

    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: _llm_reply(
        {"decision": "REPLY", "reason": "x", "response": "They are 10-0.", "fact_used": "Knicks are 10-0"}))
    r = llm_client.call_llm("Knicks look good", "fan")
    assert r["decision"] == "SKIP" and r["reason"] == "unverified_fact"


def test_llm_client_falls_back_when_model_is_gone(monkeypatch):
    import llm_client
    import nba_context
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "retired-model")
    monkeypatch.setattr(nba_context, "facts_for", lambda t: [])
    monkeypatch.setattr(nba_context, "today_line", lambda: "Today.")
    models = []

    def fake_post(url, headers=None, json=None, timeout=None):
        models.append(json["model"])
        if json["model"] == "retired-model":
            return _Resp({}, status=404, text='{"error":{"message":"The model `retired-model` does not exist"}}')
        return _llm_reply({"decision": "SKIP", "reason": "nothing_to_add", "response": ""})
    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    r = llm_client.call_llm("NBA tonight", "fan")
    assert models == ["retired-model", llm_client.FALLBACK_MODEL] and r["decision"] == "SKIP"


def test_context_ignores_ambiguous_words_and_off_topic_tweets(monkeypatch):
    import nba_context as c
    monkeypatch.setattr(c, "_rosters", lambda: {})
    teams, _ = c.detect("Hawks smash the Magpies, Heat wave in Melbourne, Kings Cross tonight")
    assert teams == set()                        # AFL / weather / places, not NBA
    teams, _ = c.detect("Miami Heat and Orlando Magic in the East")
    assert teams == {"MIA", "ORL"}
    monkeypatch.setattr(c, "_history", lambda: [])
    monkeypatch.setattr(c, "_pending", lambda: [])
    assert c.facts_for("Celtics' new jersey looks clean") == []   # not about results: no records


def test_records_only_for_result_tweets_whole_words():
    import nba_context as nc
    assert not nc.PERFORMANCE_RE.search("Bring it during the scoring drills, former teammate Frank says")
    assert nc.PERFORMANCE_RE.search("Lakers lost again, washed?")
    assert nc.PERFORMANCE_RE.search("Who wins the West this season")


def test_record_pattern_catches_win_loss_numbers():
    from engine import RECORD_RE
    assert RECORD_RE.findall("49-33 last season, 2026-27 is new") == ["49-33"]
