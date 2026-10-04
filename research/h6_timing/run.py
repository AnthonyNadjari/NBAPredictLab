"""H6 timing: end-to-end run.

    python research/h6_timing/run.py              # downloads (cached) + all analyses
    python research/h6_timing/run.py --no-fetch   # analyses only, from the on-disk cache

First run downloads ~5.6k Kalshi candle files (2 req/s, ~50 min), ~1k NBA injury-report PDFs
(2 req/s, ~25 min) and probes ~40 ESPN events (~5 min). Reruns take ~1-2 minutes.
Console output is also written to research/h6_timing/results/run_log.txt.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    (HERE / "results").mkdir(exist_ok=True)
    log = open(HERE / "results/run_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    fetch = "--no-fetch" not in sys.argv

    import probe_espn
    print("=== ESPN intraday-history / injury probe ===")
    probe_espn.main()  # cached after the first run

    if fetch:
        import fetch_kalshi
        print("\n=== Kalshi download ===")
        fetch_kalshi.main()

    import injury_timeline
    injury_timeline.analyze()  # downloads missing PDFs unless injury_rows.csv is cached

    import analyze
    sys.argv = [sys.argv[0], "--rebuild"]
    analyze.main()
    sys.stdout = sys.__stdout__
    log.close()


if __name__ == "__main__":
    main()
