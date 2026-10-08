import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_injury_alert_draft_measures_the_market_move(tmp_path, monkeypatch):
    import alerts
    keys = {"123": {"name": "Star Player", "team": "PHI", "min": 34.0, "pts": 27.5}}
    monkeypatch.setattr(alerts, "key_players", lambda: keys)
    monkeypatch.setattr(alerts, "TAPE_DIR", tmp_path)
    monkeypatch.setattr(alerts, "OUT_DIR", tmp_path / "alerts")
    monkeypatch.setattr(alerts, "STATE", tmp_path / "alerts" / "state.json")
    now = datetime.now(timezone.utc)
    day = tmp_path / now.strftime("%Y-%m-%d")
    t0, t1, t2 = (now - timedelta(minutes=m) for m in (30, 20, 2))
    end = (now + timedelta(hours=8)).isoformat()
    hdr = ["ts", "event", "ticker", "team", "yes_bid", "yes_ask", "last", "volume", "expected_end"]
    ev = "KXNBAGAME-26OCT21PHINYK"
    _write(day / "kalshi.csv", hdr, [
        [t0.isoformat(), ev, ev + "-PHI", "Philadelphia", "0.61", "0.63", "", "", end],
        [t2.isoformat(), ev, ev + "-PHI", "Philadelphia", "0.46", "0.48", "", "", end]])
    _write(day / "injuries.csv", ["ts", "team", "player", "status", "comment"],
           [[t1.isoformat(), "Philadelphia 76ers", "Star Player", "Out", "Ruled out (knee)"]])
    (tmp_path / "alerts").mkdir()
    (tmp_path / "alerts" / "state.json").write_text(json.dumps({"last_ts": t0.isoformat(), "pending": [], "done": []}))
    alerts.main()
    drafts = [json.loads(l) for l in (tmp_path / "alerts" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(drafts) == 1 and drafts[0]["opp"] == "NYK"
    assert "62% → 47%" in drafts[0]["text"] and "76ers" in drafts[0]["text"]
    alerts.main()                                    # never twice for the same news
    assert len((tmp_path / "alerts" / "drafts.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_alert_text_is_clean_and_short():
    import alerts
    a = {"player": "Joel Embiid", "team": "PHI", "opp": "NYK", "pts": 27.5, "min": 34.0, "before": 0.62,
         "event": "KXNBAGAME-26OCT21PHINYK"}
    t = alerts.alert_text(a, 0.47, 12)
    assert t.startswith("🚨 INJURY ALERT") and "62% → 47%" in t and "🔥" in t and "#Sixers #NBA" in t
    assert len(t) <= 260


def test_post_alerts_only_when_switched_on_fresh_and_once(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timedelta, timezone
    from tools import post_alerts as pa
    monkeypatch.setattr(pa, "TAPE", tmp_path)
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    monkeypatch.setattr(pa, "REPO", repo)
    now = datetime.now(timezone.utc)
    (tmp_path / "drafts.jsonl").write_text("\n".join(json.dumps(d) for d in [
        {"tag": "a", "measured_at": (now - timedelta(minutes=5)).isoformat(), "text": "x", "player": "A"},
        {"tag": "b", "measured_at": (now - timedelta(hours=2)).isoformat(), "text": "y", "player": "B"}]))
    (repo / "docs" / "autopilot.json").write_text('{"injury_alerts": false}')
    assert pa.due(now) == []
    (repo / "docs" / "autopilot.json").write_text('{"injury_alerts": true}')
    assert [d["tag"] for d in pa.due(now)] == ["a"]                  # stale one never posted
    (tmp_path / "posted.jsonl").write_text(json.dumps({"tag": "a", "at": now.isoformat()}) + "\n")
    assert pa.due(now) == []                                          # once
