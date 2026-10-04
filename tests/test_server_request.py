import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))


def test_panel_request_runs_once(tmp_path, monkeypatch, capsys):
    from tools import server_request as sr
    req = tmp_path / "request.json"
    req.write_text(json.dumps({"id": "1", "kind": "dry", "max_replies": 3}))
    monkeypatch.setattr(sr, "REQUEST", req)
    monkeypatch.setattr(sr, "DONE", tmp_path / "done.txt")
    sr.main()
    assert capsys.readouterr().out.strip() == "dry 3"
    sr.main()
    assert capsys.readouterr().out.strip() == "none"  # same id: not again
    req.write_text(json.dumps({"id": "2", "kind": "rm -rf"}))
    sr.main()
    assert capsys.readouterr().out.strip() == "none"  # unknown kind ignored
