#!/usr/bin/env python3
"""
The morning after: grade last night's threads in public, and show off the good nights.

  * one quote post per thread we posted last night: "✅ Called it" / "❌ Missed this one", with the
    final score and our probability, quoting the thread (its URL at the end makes X show it as a quote);
  * when last night went well (at least 4 graded picks and 2 out of 3 right), a results post: the
    record card + a table of every pick, and an invitation to DM us for the picks before tip-off.

Every number comes from docs/dashboard.json (refreshed by the 09:00 UTC run, which runs this right
after). Specials go to docs/pending_games.json and requests to docs/vision/publish_queue.json with
"not_before" = 16:00 UTC (noon in New York): the server posts them then, 4 minutes apart.

    python scripts/results_post.py [--date YYYY-MM-DD (US date of the games)] [--dry]
"""
import argparse
import json
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine.teams import to_code  # noqa: E402
from src.social.thread import nickname  # noqa: E402

DASHBOARD = ROOT / "docs" / "dashboard.json"
PENDING = ROOT / "docs" / "pending_games.json"
QUEUE = ROOT / "docs" / "vision" / "publish_queue.json"
LOG = ROOT / "docs" / "vision" / "publish_log.json"
POST_AT = time(16, 0)          # UTC
MIN_PICKS, MIN_RATE = 4, 2 / 3
CTA = "Want the picks before tip-off? DM us 📩"


def _read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _teams(game_id: str) -> tuple:
    """'Washington_Wizards_vs_Cleveland_Cavaliers_2026-10-28' -> ({'WAS', 'CLE'}, '2026-10-28')"""
    name, _, day = game_id.rpartition("_")
    a, _, b = name.partition("_vs_")
    return {to_code(a.replace("_", " ")), to_code(b.replace("_", " "))}, day


def _score(r: dict) -> str:
    pts = sorted((int(x) for x in "".join(c if c.isdigit() else " " for c in str(r.get("score") or "")).split()), reverse=True)
    return f"{pts[0]}-{pts[1]}" if len(pts) == 2 else ""


def quote_text(r: dict, url: str) -> str:
    pick, opp = r["pick"], (r["away"] if r["pick"] == r["home"] else r["home"])
    p, score = round(r["p"] * 100), _score(r)
    if r["correct"]:
        return f"✅ Called it.\n\nThe {nickname(pick)} beat the {nickname(opp)} {score}. We had them at {p}%.\n\n{url}"
    return (f"❌ Missed this one.\n\nThe {nickname(opp)} beat the {nickname(pick)} {score}. We had {nickname(pick)} at {p}%. "
            f"Every miss stays on the record.\n\n{url}")


def results_thread(rows: list, season: dict, day: date):
    w, n = sum(r["correct"] for r in rows), len(rows)
    if n < MIN_PICKS or w / n < MIN_RATE:
        return None
    label = day.strftime("%b %d").upper().replace(" 0", " ")
    s_n, s_ok = season.get("n") or 0, season.get("correct") or 0
    season_txt = f"{s_ok}-{s_n - s_ok}" if s_n else "–"
    exp = sum(r["p"] for r in rows) / n
    hits = sorted((r for r in rows if r["correct"]), key=lambda r: r["p"])
    best = hits[0] if hits else None
    best_txt = (f"{nickname(best['pick'])} at {round(best['p'] * 100)}%" if best else "–")
    t1 = (f"📈 {w}/{n} on our picks last night.\n\n"
          + (f"Season: {s_ok}/{s_n} ({s_ok / s_n * 100:.0f}%), every pick graded after the buzzer.\n\n" if s_n else "")
          + CTA)
    card1 = {"type": "recap", "period": label, "eyebrow": "Last night", "wins": w, "losses": n - w,
             "hit_rate": f"{w / n * 100:.0f}%", "expected": f"{exp * 100:.0f}%", "season": season_txt,
             "best_label": "Toughest hit", "best": best_txt}
    table = sorted(rows, key=lambda r: (-r["correct"], -r["p"]))[:10]
    card2 = {"type": "ranking", "kicker": label, "eyebrow": "Every pick, graded", "numbered": False,
             "title_html": f"<em>{w}/{n}</em> last night",
             "rows": [{"team": r["pick"], "value": f"{'W' if r['correct'] else 'L'} {_score(r)}",
                       "name": f"{nickname(r['pick'])} {round(r['p'] * 100)}%",
                       "note": f"vs {nickname(r['away'] if r['pick'] == r['home'] else r['home'])}"} for r in table],
             "foot": "Our probability on the team we picked"}
    t2 = "Every pick from last night, wins and misses 👇"
    return [{"text": t1, "card": card1}, {"text": t2, "card": card2}]


def build(day: date) -> list:
    """[(special_id, title, thread)] for the games of US date `day`."""
    dash = _read(DASHBOARD, {})
    rows = [r for r in dash.get("recent", []) if r.get("date") == day.isoformat() and r.get("correct") is not None]
    out = []
    for e in _read(LOG, []):
        if not (e.get("ok") and e.get("url") and str(e.get("id", "")).startswith("autopilot-")):
            continue
        try:
            teams, gday = _teams(e["game_id"])
        except Exception:
            continue
        r = next((r for r in rows if gday == day.isoformat() and {r["home"], r["away"]} == teams), None)
        if r:
            out.append((f"result-{e['game_id']}", f"Résultat {r['away']} @ {r['home']}",
                        [{"text": quote_text(r, e["url"]), "card": None}]))
    thread = results_thread(rows, dash.get("season", {}), day)
    if thread:
        out.append((f"night-{day.isoformat()}", f"Bilan de la nuit {day.isoformat()}", thread))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    day = date.fromisoformat(a.date) if a.date else now.date() - timedelta(days=1)
    posts = build(day)
    for sid, _, thread in posts:
        print(f"== {sid}\n" + "\n--\n".join(t["text"] for t in thread))
    if a.dry or not posts:
        print("nothing to queue" if not posts else "dry run")
        return 0
    not_before = datetime.combine(now.date(), POST_AT, tzinfo=timezone.utc)
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT).returncode
    for attempt in range(3):
        pend = _read(PENDING, {})
        done = {s["id"] for s in pend.get("specials", []) if s.get("published")}
        new = [(sid, title, th) for sid, title, th in posts if sid not in done]
        ids = {sid for sid, _, _ in new}
        pend["specials"] = [s for s in pend.get("specials", []) if s.get("id") not in ids] + [
            {"id": sid, "type": "announcement", "title": title, "matchup": title, "date": day.isoformat(),
             "published": False, "thread": th} for sid, title, th in new]
        PENDING.write_text(json.dumps(pend, indent=2, ensure_ascii=False), encoding="utf-8")
        q = _read(QUEUE, {"requests": []})
        have = {r.get("id") for r in q["requests"] if isinstance(r, dict)}
        q["requests"] = (q["requests"] + [{"id": sid, "game_id": sid, "texts": None, "dry": False,
                                           "requested_at": now.isoformat(timespec="seconds"),
                                           "not_before": not_before.isoformat(timespec="seconds"),
                                           "source": "results_post"} for sid in ids if sid not in have])[-20:]
        QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
        git("add", "docs/pending_games.json", "docs/vision/publish_queue.json")
        git("-c", "user.name=github-actions[bot]", "-c", "user.email=github-actions[bot]@users.noreply.github.com",
            "commit", "-q", "-m", f"Results posts queued for {day.isoformat()} ({len(ids)})")
        if git("push", "-q", "origin", "HEAD:main") == 0:
            return 0
        git("fetch", "-q", "origin", "main")
        git("reset", "-q", "--hard", "origin/main")
    return 1


if __name__ == "__main__":
    sys.exit(main())
