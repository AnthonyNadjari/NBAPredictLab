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
