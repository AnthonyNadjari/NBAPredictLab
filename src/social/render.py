"""Render thread cards to PNG with headless Chromium (Playwright) from docs/cards/card.html."""
import base64
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

TEMPLATE = Path(__file__).resolve().parents[2] / "docs" / "cards" / "card.html"


def card_url(card: Dict) -> str:
    data = base64.urlsafe_b64encode(json.dumps(card).encode("utf-8")).decode("ascii")
    return TEMPLATE.as_uri() + "#" + data


def render_cards(cards: List[Optional[Dict]], out_dir: Optional[str] = None, scale: float = 2.0) -> List[Optional[str]]:
    """PNG path for each card (None where the card is None or rendering failed)."""
    from playwright.sync_api import sync_playwright

    out = Path(out_dir or tempfile.mkdtemp(prefix="cards_"))
    paths: List[Optional[str]] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 675}, device_scale_factor=scale)
        for i, card in enumerate(cards):
            if not card:
                paths.append(None)
                continue
            try:
                page.goto(card_url(card), wait_until="domcontentloaded")
                page.reload(wait_until="domcontentloaded")  # hash-only navigations don't rerun scripts
                page.wait_for_function("window.__READY__ === true", timeout=20000)
                p = out / f"card_{i}_{card.get('type', 'x')}.png"
                page.locator("#card").screenshot(path=str(p))
                paths.append(str(p))
            except Exception as e:  # a missing image must not block the thread
                print(f"Card {i} failed: {e}")
                paths.append(None)
        browser.close()
    return paths
