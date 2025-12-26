#!/usr/bin/env python3
"""
Send real daily report email to both recipients (production mode)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.email_reporter import EmailReporter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Send real daily report to both recipients"""
    logger.info("=" * 60)
    logger.info("SENDING REAL DAILY REPORT - Production Mode")
    logger.info("Recipients: nadjari.anthony@gmail.com, recipient@example.com")
    logger.info("=" * 60)
    
    try:
        email_reporter = EmailReporter(db_path='data/nba_predictor.db')
        
        # Send real email (not test mode)
        success = email_reporter.send_daily_report(test_mode=False)
        
        if success:
            logger.info("=" * 60)
            logger.info("✅ REAL EMAIL SENT SUCCESSFULLY TO BOTH RECIPIENTS!")
            logger.info("=" * 60)
        else:
            logger.error("=" * 60)
            logger.error("❌ EMAIL FAILED")
            logger.error("=" * 60)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

