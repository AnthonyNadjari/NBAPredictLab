"""
NBAVision Engine — Failure notification via Discord webhook (optional).
"""
import json
import os
import urllib.request
from config import DISCORD_WEBHOOK_URL


def send_discord_notification(title: str, message: str, *, color: int = 0xFF0000) -> bool:
    """
    Send an embed to the configured Discord webhook.
    Returns True if sent (or if no webhook is configured — silent no-op).
    """
    url = DISCORD_WEBHOOK_URL
    if not url:
        return True

    run_id = os.environ.get("NBAVISION_RUN_ID", "local")
    payload = {
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "footer": {"text": f"Run ID: {run_id}"},
        }]
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status in (200, 204)
    except Exception as e:
        print(f"Notify: Discord webhook failed: {e}", flush=True)
        return False


def notify_auth_failure(reason: str) -> None:
    send_discord_notification(
        "NBAVision: Auth Failed",
        f"Session could not be established.\n**Reason:** {reason}\n\nRe-export cookies and update the `TWITTER_COOKIES_JSON` secret.",
    )


def notify_session_summary(total_replied: int, total_skipped: int, errors: dict | None = None) -> None:
    desc = f"Replied: **{total_replied}** | Skipped: **{total_skipped}**"
    if errors:
        desc += "\n**Issues:** " + ", ".join(f"{k}={v}" for k, v in errors.items())
    color = 0x00FF00 if total_replied > 0 else 0xFFA500
    send_discord_notification("NBAVision: Session Complete", desc, color=color)
