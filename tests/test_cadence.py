import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))
import cadence  # noqa: E402
from config import TZ  # noqa: E402


def simulate(days=5):
    runs, t, n = [], datetime(2026, 10, 12, tzinfo=TZ), 0
    per_day, hours = Counter(), Counter()
    while t < datetime(2026, 10, 12, tzinfo=TZ) + timedelta(days=days):
        d = cadence.decide(t, runs)
        if d["due"]:
            n += 1
            end = t + timedelta(minutes=d["max"] * 2.6 + 8)
            runs.append({"auth": "ok", "end": end.isoformat(), "replied": d["max"], "run_id": f"r{n}"})
            per_day[t.date()] += d["max"]
            hours[t.hour] += d["max"]
            t = end
        t += timedelta(minutes=5)
    return runs, per_day, hours


def test_daily_volume_and_quiet_hours():
    runs, per_day, hours = simulate()
    full_days = list(per_day.values())[1:-1]
    assert all(10 <= v <= 30 for v in full_days), per_day
    assert sum(hours[h] for h in (9, 10, 11, 12)) == 0      # US night: nothing
    # never more than the hourly cap inside any hour after a session ended
    for r in runs:
        t = datetime.fromisoformat(r["end"])
        assert sum(x["replied"] for x in runs if timedelta(0) <= t - datetime.fromisoformat(x["end"]) < timedelta(hours=1)) <= cadence.HOURLY_MAX + cadence.SESSION_SIZE[1]


def test_prime_time_gets_more():
    _, _, hours = simulate()
    prime = sum(hours[h] for h in cadence.PEAK_HOURS) / len(cadence.PEAK_HOURS)
    day = sum(hours[h] for h in range(14, 21)) / 7
    assert prime > day


def test_same_plan_on_every_tick():
    now = datetime(2026, 10, 12, 22, 0, tzinfo=TZ)
    runs = [{"auth": "ok", "end": (now - timedelta(minutes=10)).isoformat(), "replied": 8, "run_id": "x"}]
    assert cadence.decide(now, runs) == cadence.decide(now, runs)
    assert cadence.decide(datetime(2026, 10, 12, 11, 0, tzinfo=TZ), [])["why"] == "quiet hours"
