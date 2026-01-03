#!/usr/bin/env python3
"""
Mark a game as published in the pending_games.json file.

This script is called by GitHub Actions after successfully posting a thread.

Usage:
    python scripts/mark_published.py GAME_ID

Example:
    python scripts/mark_published.py LAL_vs_BOS_2026-01-03
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mark_published(game_id: str) -> bool:
    """
    Mark a game as published in the JSON file.

    Args:
        game_id: Game identifier to mark as published

    Returns:
        True if successful, False otherwise
    """
    try:
        json_path = Path('docs/pending_games.json')

        if not json_path.exists():
            logger.error(f"File not found: {json_path}")
            return False

        # Read current data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Find and mark the game
        game_found = False
        for game in data.get('games', []):
            if game['id'] == game_id:
                game['published'] = True
                game['published_at'] = datetime.now().isoformat()
                game_found = True
                logger.info(f"✓ Marked game as published: {game['matchup']}")
                break

        if not game_found:
            logger.warning(f"Game {game_id} not found in JSON")
            return False

        # Write updated data
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Updated {json_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to mark game as published: {e}", exc_info=True)
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/mark_published.py GAME_ID")
        print("Example: python scripts/mark_published.py LAL_vs_BOS_2026-01-03")
        sys.exit(1)

    game_id = sys.argv[1]

    logger.info(f"Marking game as published: {game_id}")

    success = mark_published(game_id)

    if success:
        logger.info("✓ SUCCESS")
        sys.exit(0)
    else:
        logger.error("✗ FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
