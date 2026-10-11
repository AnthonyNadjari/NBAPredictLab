import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vision"))
from reply_validator import humanize, stat_opener_again  # noqa: E402


def test_humanize_drops_most_final_periods_and_keeps_names():
    rng = random.Random(1)
    out = [humanize("Luka with the keys is scary for the West.", rng) for _ in range(200)]
    assert sum(o.endswith(".") for o in out) < 60
    assert all(o.startswith("Luka") for o in out)
    assert humanize("wait for it...", rng) == "wait for it..."


def test_stat_opener_again():
    assert stat_opener_again("17 on 13 shots", ["nice", "25 on 12 shots"])
    assert not stat_opener_again("17 on 13 shots", ["nice", "sure"])
    assert not stat_opener_again("Tatum is back", ["25 on 12"])
