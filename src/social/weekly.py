"""Weekly recap post: last Monday-Sunday (US dates) of graded picks."""
import sqlite3
from datetime import date, timedelta
from typing import Dict, Optional

from src.engine.teams import to_code
from src.social.thread import nickname, _fit

MIN_PICKS = 5


def last_week(today: date) -> tuple:
    monday = today - timedelta(days=today.weekday() + 7)
    return monday, monday + timedelta(days=6)


def build_weekly(db_path: str, today: date, season_since: str) -> Optional[Dict]:
    start, end = last_week(today)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""SELECT game_date, home_team, away_team, predicted_home_prob, correct
                           FROM predictions WHERE correct IS NOT NULL AND game_date BETWEEN ? AND ?""",
                        (start.isoformat(), end.isoformat())).fetchall()
    season = conn.execute("SELECT COUNT(*), SUM(correct) FROM predictions WHERE correct IS NOT NULL "
                          "AND game_date >= ?", (season_since,)).fetchone()
    conn.close()
    if len(rows) < MIN_PICKS:
        return None

    def pick(r):
        home_pick = r[3] >= 0.5
        team = to_code(r[1] if home_pick else r[2])
        opp = to_code(r[2] if home_pick else r[1])
        return team, opp, (r[3] if home_pick else 1 - r[3])

    wins = sum(r[4] for r in rows)
    losses = len(rows) - wins
    expected = sum(pick(r)[2] for r in rows) / len(rows)
    hits = [r for r in rows if r[4]]
    misses = [r for r in rows if not r[4]]
    best = min(hits, key=lambda r: pick(r)[2]) if hits else None
    worst = max(misses, key=lambda r: pick(r)[2]) if misses else None
    period = f"{start.strftime('%b %d')} – {end.strftime('%b %d')}".upper().replace(" 0", " ")

    tiers = []
    for lo, hi, label in ((0.7, 1.01, "70%+"), (0.6, 0.7, "60-70%"), (0.0, 0.6, "under 60%")):
        t = [r for r in rows if lo <= pick(r)[2] < hi]
        if t:
            w = sum(r[4] for r in t)
            tiers.append(f"{label}: {w}-{len(t) - w}")

    s_n, s_ok = season[0] or 0, season[1] or 0
    season_txt = f"{s_ok}-{s_n - s_ok} ({s_ok / s_n * 100:.1f}%)" if s_n else "–"
    best_txt = (f"{nickname(pick(best)[0])} over {nickname(pick(best)[1])} at {pick(best)[2] * 100:.0f}%"
                if best else "–")
    t1 = (f"📊 WEEK IN REVIEW · {period}\n\n"
          f"{wins}-{losses} on our picks ({wins / len(rows) * 100:.0f}%)\n"
          f"Our probabilities expected {expected * 100:.0f}%.\n\n"
          + (f"Season: {season_txt}\n\n" if s_n else "")
          + "Every pick, graded 👇")
    t2 = "🎯 BY CONFIDENCE\n\n" + "\n".join(tiers)
    if best:
        t2 += f"\n\nToughest call that hit: {best_txt}."
    if worst:
        t2 += f"\nBiggest miss: {nickname(pick(worst)[0])} at {pick(worst)[2] * 100:.0f}% vs {nickname(pick(worst)[1])}."
    card = {"type": "recap", "period": period, "eyebrow": "Week in review", "wins": wins, "losses": losses,
            "hit_rate": f"{wins / len(rows) * 100:.0f}%", "expected": f"{expected * 100:.0f}%",
            "season": season_txt.split(" ")[0], "best_label": "Toughest hit", "best": best_txt if best else "–"}
    return {
        "id": f"weekly-{start.isoformat()}", "type": "weekly", "title": f"Week in review {period}",
        "matchup": f"Bilan de la semaine ({period.title()})", "date": end.isoformat(), "published": False,
        "thread": [{"text": _fit(t1), "card": card}, {"text": _fit(t2), "card": None}],
    }
