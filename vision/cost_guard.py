"""
LLM spend tracking and a hard daily cap, from the provider's real balance (DeepSeek /user/balance).

The first reading of the UTC day is kept in the state dir; spent today = first reading - now.
Once spent today reaches LLM_DAILY_BUDGET_USD (default 0.60), sessions stop calling the LLM.
Every reading is appended to docs/vision/llm_cost.json (one entry per day) for the control panel.
Providers without a balance endpoint (Groq): no tracking, never blocks.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import requests

from config import LOCAL_STATE_DIR, REPO_ROOT

STATE = LOCAL_STATE_DIR / "llm_balance.json"
PANEL = REPO_ROOT / "docs" / "vision" / "llm_cost.json"


def _balance() -> float | None:
    if "deepseek" not in os.getenv("LLM_BASE_URL", ""):
        return None
    try:
        js = requests.get("https://api.deepseek.com/user/balance", timeout=15,
                          headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '')}"}).json()
        usd = [b for b in js.get("balance_infos", []) if b.get("currency") == "USD"]
        return float(usd[0]["total_balance"]) if usd else None
    except Exception as e:
        print(f"Cost: balance unavailable ({e})", flush=True)
        return None


def check() -> dict:
    """{balance, spent_today, budget, over} from the real balance, and record it."""
    budget = float(os.getenv("LLM_DAILY_BUDGET_USD", "0.60"))
    bal = _balance()
    if bal is None:
        return {"balance": None, "spent_today": None, "budget": budget, "over": False}
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        st = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        st = {}
    if st.get("day") != day:
        # new day: start from the last balance we saw (so spend between days is not lost), or now
        st = {"day": day, "start": st.get("last", bal)}
    if bal > st["start"]:                       # top-up during the day
        st["start"] = bal
    st["last"] = bal
    STATE.write_text(json.dumps(st), encoding="utf-8")
    spent = round(st["start"] - bal, 4)
    try:
        log = json.loads(PANEL.read_text(encoding="utf-8"))
    except Exception:
        log = {}
    log[day] = {"spent_usd": spent, "balance_usd": bal, "budget_usd": budget,
                "at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    PANEL.write_text(json.dumps(dict(sorted(log.items())[-90:]), indent=1), encoding="utf-8")
    return {"balance": bal, "spent_today": spent, "budget": budget, "over": spent >= budget}
