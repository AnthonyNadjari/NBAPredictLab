"""Session heartbeat: the engine beats on every step; a watchdog in main.py ends a session
that stopped making progress (a dead Playwright driver makes calls spin forever)."""
import time

LAST = time.time()
REPLIES: list = []      # the session's replies_posted list (shared, for a partial record)


def beat() -> None:
    global LAST
    LAST = time.time()


def idle_seconds() -> float:
    return time.time() - LAST
