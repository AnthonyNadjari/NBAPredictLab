"""Build the X thread for one game: hook + 3 reasons + the risk + track record.

Every tweet is a dict {"text": str, "card": dict | None}; cards are rendered to
PNG by src/social/render.py from docs/cards/card.html (same template the control
panel uses for its preview), so what you preview is what gets posted.
"""
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from src.engine.teams import TEAMS, to_code
from src.social import angles as A

ET = ZoneInfo("America/New_York")
MAX_LEN = 280
EMOJI = {"elo": "📊", "season": "📈", "form": "🔥", "offense": "🎯", "defense": "🛡️", "shooting": "🏹",
         "rest": "😴", "schedule": "🗓️", "venue": "🏟️", "streak": "⚡", "injuries": "🩹", "model": "🤖"}


def nickname(code: str) -> str:
    full = TEAMS.get(code, code)
    return "76ers" if code == "PHI" else ("Trail Blazers" if code == "POR" else full.split()[-1])


def _when(start_utc: Optional[str]) -> str:
    if not start_utc:
        return ""
    t = datetime.fromisoformat(start_utc).astimezone(ET)
    return t.strftime("%a %b %d · %I:%M %p ET").replace(" 0", " ").upper()


def _fit(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_LEN:
        return text
    cut = text[:MAX_LEN - 1]
    return cut[:cut.rfind(" ")].rstrip(" ,.;:") + "…"


def build_thread(pred: Dict, record_line: str) -> Dict:
    """pred: a predictions row (full team names) with 'features' (dict) and probabilities."""
    f = pred.get("features") or {}
    home, away = to_code(pred["home_team"]), to_code(pred["away_team"])
    ph = float(pred["predicted_home_prob"])
    pick_home = ph >= 0.5
    p = ph if pick_home else 1 - ph
    pick, opp = (home, away) if pick_home else (away, home)
    P = A.Side(f, "home" if pick_home else "away", pick, nickname(pick))
    O = A.Side(f, "away" if pick_home else "home", opp, nickname(opp))

    odds_pick = pred.get("home_odds") if pick_home else pred.get("away_odds")
    src = f.get("probability_source")
    market = src in ("market", "blend")
    side = (lambda x: x if pick_home else 1 - x)
    own, mkt = f.get("own_home_prob"), f.get("market_home_prob")
    if src == "blend":
        source = f"Our model {round(100 * side(own))}% + books {round(100 * side(mkt))}%"
    elif market:
        source = "Books' consensus" + (f" · our model {round(100 * side(own))}%" if own is not None else "")
    elif src == "own":
        source = "Our model (no odds yet)"
    else:
        source = "Stats model (no odds yet)"
    matchup = f"{nickname(away)} at {nickname(home)}"
    when = _when(f.get("start_utc"))

    def rec(side):
        gp, w = f.get(f"{side}_games_played") or 0, f.get(f"{side}_season_win_pct")
        if gp and gp >= 3 and w is not None:
            wins = round(w * gp)
            return f"{wins}-{int(gp) - wins}"
        return ""

    when_txt = ""
    if f.get("start_utc"):
        when_txt = datetime.fromisoformat(f["start_utc"]).astimezone(ET).strftime("%a %b %d, %I:%M %p ET").replace(" 0", " ")
    hook = (f"🏀 {TEAMS[away]} at {TEAMS[home]}\n"
            f"{'🕗 ' + when_txt if when_txt else ''}\n\n"
            f"🎯 Pick: {TEAMS[pick]} ({p * 100:.0f}%)\n"
            + (f"📊 Our model {round(100 * side(own))}% · books {round(100 * side(mkt))}%\n" if src == "blend" else "")
            + (f"💰 Odds {float(odds_pick):.2f} · {'books’ consensus' if market else 'model price'}\n" if odds_pick else "")
            + "\nThe case, and the risk 🧵")
    tweets = [{"text": _fit(hook), "card": {
        "type": "match", "when": when, "away": away, "home": home, "away_prob": 1 - ph,
        "away_sub": rec("away") or "away", "home_sub": rec("home") or "home",
        "pick_name": nickname(pick), "odds": f"{float(odds_pick):.2f}" if odds_pick else "", "source": source,
        "tape": _tape(f)}}]

    all_angles = A.build(f, P, O, pick_home)
    support = [a for a in A.ranked(all_angles) if a.strength > 0.25 and a.key != "model"][:3]
    if len(support) < 3:
        extra = [a for a in A.ranked(all_angles) if a not in support and a.strength > 0]
        support += extra[:3 - len(support)]
    for i, a in enumerate(support, 1):
        text = f"{EMOJI.get(a.key, '•')} {a.title.upper()}\n\n{a.text}"
        card = None
        if a.rows:
            card = {"type": "edge", "away": away, "home": home, "matchup": f"{matchup} · {when}".strip(" ·"),
                    "eyebrow": f"Reason {i} of {len(support)} · {nickname(pick)}", "title": a.title,
                    "sub": a.text, "rows": a.rows}
        tweets.append({"text": _fit(text), "card": card})

    against = [a for a in all_angles if a.strength < -0.3]
    risk_odds = 100 - round(p * 100)
    if against:
        worst = min(against, key=lambda a: a.strength)
        risk = (f"⚠️ THE RISK\n\n{worst.text}\n\n"
                f"Teams priced like {O.name} tonight still win about {risk_odds} times in 100.")
    else:
        risk = (f"⚠️ THE RISK\n\nNothing big flags against the {P.name}. "
                f"But {p * 100:.0f}% is not 100%: priced like this, {O.name} win about {risk_odds} times in 100.")
    tweets.append({"text": _fit(risk), "card": None})

    tweets.append({"text": _fit(f"📈 {record_line}\n\nEvery pick is posted before tip-off and graded after the "
                                f"final buzzer. No deleting the misses.\n\n🔔 Follow for the full slate every game day."),
                   "card": None})
    return {"tweets": tweets, "pick": pick, "prob": p}


def _tape(f: Dict) -> List[Dict]:
    """Tale of the tape for the match card: away value first, home second."""
    def g(side, k):
        return f.get(f"{side}_{k}")
    tape = []
    if g("away", "last10_win_pct") is not None:
        w = lambda s: round((g(s, "last10_win_pct") or 0) * 10)
        tape.append({"label": "Last 10", "away": f"{w('away')}-{10 - w('away')}", "home": f"{w('home')}-{10 - w('home')}"})
    if g("away", "last10_net_rating") is not None:
        tape.append({"label": "Net rtg L10", "away": f"{g('away', 'last10_net_rating'):+.1f}",
                     "home": f"{g('home', 'last10_net_rating'):+.1f}"})
    if g("away", "rest_days") is not None:
        r = lambda s: "B2B" if g(s, "back_to_back") else f"{int(g(s, 'rest_days'))}d"
        tape.append({"label": "Rest", "away": r("away"), "home": r("home")})
    if g("away", "elo") is not None:
        tape.append({"label": "Elo", "away": f"{g('away', 'elo'):.0f}", "home": f"{g('home', 'elo'):.0f}"})
    return tape


def record_line(db_path: str, since: str, backtest_acc: float = 0.686) -> str:
    from src.engine.pipeline import track_record
    rec = track_record(db_path, since=since)
    if rec["n"] >= 30:
        return f"This season: {rec['correct']}-{rec['n'] - rec['correct']} on picks ({rec['accuracy'] * 100:.1f}%)."
    return f"Our probabilities hit {backtest_acc * 100:.1f}% of winners over 4 seasons of backtests."
