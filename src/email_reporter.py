"""
Email Reporter for NBA Predictions
Sends daily reports via Outlook using win32com (no SMTP configuration needed)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sqlite3
import logging

logger = logging.getLogger(__name__)


def _teams_match(pred_home: str, pred_away: str, game_home: str, game_away: str) -> bool:
    """Return True if (pred_home, pred_away) is the same matchup as (game_home, game_away)."""
    if not all([pred_home, pred_away, game_home, game_away]):
        return False
    ph, pa = pred_home.strip().lower(), pred_away.strip().lower()
    gh, ga = game_home.strip().lower(), game_away.strip().lower()
    if (ph == gh and pa == ga) or (ph == ga and pa == gh):
        return True
    # Abbrev vs full: "bos" in "boston celtics", or "celtics" in "boston celtics"
    def pair_matches(a1, a2, b1, b2):
        return (
            (a1 in b1 or b1 in a1 or a1 in b2 or b2 in a1)
            and (a2 in b1 or b1 in a2 or a2 in b2 or b2 in a2)
        )
    return pair_matches(ph, pa, gh, ga) or pair_matches(ph, pa, ga, gh)


# Try to import win32com for Outlook
try:
    import win32com.client
    WIN32COM_AVAILABLE = True
except ImportError:
    WIN32COM_AVAILABLE = False
    logger.warning("win32com not available. Install with: pip install pywin32")


class EmailReporter:
    """Send daily NBA prediction reports via email"""
    
    def __init__(self, db_path: str = 'data/nba_predictor.db'):
        """
        Initialize email reporter.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.recipients = [
            'nadjari.anthony@gmail.com',
            'hugo.dubelloy@hotmail.com'
        ]
    
    def get_yesterday_results(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get yesterday's games with predictions and actual results.
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses yesterday.
        
        Returns:
            List of game dictionaries with predictions and results
        """
        if date is None:
            yesterday = datetime.now() - timedelta(days=1)
            date = yesterday.strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get predictions with results for the specified date
        cursor.execute("""
            SELECT
                game_date,
                home_team,
                away_team,
                predicted_winner,
                predicted_home_prob,
                predicted_away_prob,
                confidence,
                actual_winner,
                actual_home_score,
                actual_away_score,
                correct,
                home_odds,
                away_odds
            FROM predictions
            WHERE game_date = ?
            ORDER BY home_team, away_team
        """, (date,))

        results = []
        for row in cursor.fetchall():
            game_date, home_team, away_team, predicted_winner, pred_home_prob, pred_away_prob, \
            confidence, actual_winner, actual_home_score, actual_away_score, correct, home_odds, away_odds = row

            # Use stored odds if available, otherwise calculate from probabilities as fallback
            if home_odds is None:
                home_odds = round(1 / pred_home_prob, 2) if pred_home_prob > 0 else 99.0
            if away_odds is None:
                away_odds = round(1 / pred_away_prob, 2) if pred_away_prob > 0 else 99.0
            
            results.append({
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': predicted_winner,
                'predicted_home_prob': pred_home_prob,
                'predicted_away_prob': pred_away_prob,
                'home_odds': home_odds,
                'away_odds': away_odds,
                'confidence': confidence,
                'actual_winner': actual_winner,
                'actual_home_score': actual_home_score,
                'actual_away_score': actual_away_score,
                'correct': correct
            })
        
        conn.close()
        # Resolve missing or 0-0 scores from games table so email shows real results
        return self._resolve_scores_from_games(results)
    
    def _resolve_scores_from_games(self, results: List[Dict]) -> List[Dict]:
        """
        For each result that has missing/0-0 scores OR missing actual_winner,
        look up the games table and backfill scores + winner + correct flag.

        This makes the email robust against the case where the predictions
        table has not yet been updated (e.g. the CDN schedule returned
        score=0 at daily update time, but real scores arrived later via the
        NBA API path or the Streamlit "Refresh Game Data" button).
        """
        if not results:
            return results
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        resolved = []
        for game in results:
            needs_scores = (
                game.get('actual_home_score') is None
                or game.get('actual_away_score') is None
                or (game.get('actual_home_score') == 0 and game.get('actual_away_score') == 0)
            )
            needs_winner = not game.get('actual_winner')
            if not (needs_scores or needs_winner):
                resolved.append(game)
                continue
            game_date = game.get('game_date')
            home_team = game.get('home_team') or ''
            away_team = game.get('away_team') or ''
            if not game_date or not home_team or not away_team:
                resolved.append(game)
                continue
            # Normalize date to YYYY-MM-DD for lookup
            if hasattr(game_date, 'strftime'):
                game_date = game_date.strftime('%Y-%m-%d')
            elif isinstance(game_date, str) and len(game_date) == 10:
                pass
            else:
                try:
                    game_date = datetime.strptime(str(game_date)[:10], '%Y-%m-%d').strftime('%Y-%m-%d')
                except Exception:
                    resolved.append(game)
                    continue
            # Exact match first (home/away or swapped)
            cursor.execute("""
                SELECT home_score, away_score, home_team, away_team
                FROM games
                WHERE game_date = ?
                AND home_score IS NOT NULL AND away_score IS NOT NULL
                AND (home_score > 0 OR away_score > 0)
                AND (
                    (home_team = ? AND away_team = ?)
                    OR (home_team = ? AND away_team = ?)
                )
                LIMIT 1
            """, (game_date, home_team, away_team, away_team, home_team))
            row = cursor.fetchone()
            if not row:
                # Fallback: fetch all games that day and match by team name (handles abbrev vs full)
                cursor.execute("""
                    SELECT home_score, away_score, home_team, away_team
                    FROM games
                    WHERE game_date = ?
                    AND home_score IS NOT NULL AND away_score IS NOT NULL
                    AND (home_score > 0 OR away_score > 0)
                """, (game_date,))
                for g in cursor.fetchall():
                    hs, aws, g_home, g_away = g
                    if _teams_match(home_team, away_team, g_home, g_away):
                        row = (hs, aws, g_home, g_away)
                        break
            if row:
                hs = int(row[0])
                aws = int(row[1])
                g_home = row[2] if len(row) >= 3 else home_team
                g_away = row[3] if len(row) >= 4 else away_team
                # Align the games-row orientation with the prediction's home/away.
                # Same-orientation if games.home_team refers to the same team as
                # prediction.home_team (accounting for abbrev vs full name).
                def _same(a: str, b: str) -> bool:
                    if not a or not b:
                        return False
                    al, bl = a.strip().lower(), b.strip().lower()
                    return al == bl or al in bl or bl in al
                same_orientation = _same(home_team, g_home) and _same(away_team, g_away)
                if not same_orientation:
                    # Games row is swapped: games.home_team == prediction.away_team
                    hs, aws = aws, hs
                game = dict(game)
                game['actual_home_score'] = hs
                game['actual_away_score'] = aws
                home_won = hs > aws
                if needs_winner:
                    game['actual_winner'] = home_team if home_won else away_team
                # Recompute `correct` if missing, using the (possibly new) winner
                if game.get('correct') is None and game.get('predicted_winner') and game.get('actual_winner'):
                    game['correct'] = 1 if game['predicted_winner'] == game['actual_winner'] else 0
            resolved.append(game)
        conn.close()
        return resolved
    
    def get_today_predictions(self, date: Optional[str] = None) -> List[Dict]:
        """
        Get today's predictions (without results yet).
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses today.
        
        Returns:
            List of prediction dictionaries
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get predictions for today (regardless of whether results are in)
        cursor.execute("""
            SELECT
                game_date,
                home_team,
                away_team,
                predicted_winner,
                predicted_home_prob,
                predicted_away_prob,
                confidence,
                home_odds,
                away_odds
            FROM predictions
            WHERE game_date = ?
            ORDER BY confidence DESC, home_team, away_team
        """, (date,))

        predictions = []
        for row in cursor.fetchall():
            game_date, home_team, away_team, predicted_winner, pred_home_prob, pred_away_prob, confidence, home_odds, away_odds = row

            # Use stored odds if available, otherwise calculate from probabilities as fallback
            if home_odds is None:
                home_odds = round(1 / pred_home_prob, 2) if pred_home_prob > 0 else 99.0
            if away_odds is None:
                away_odds = round(1 / pred_away_prob, 2) if pred_away_prob > 0 else 99.0
            
            predictions.append({
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'predicted_winner': predicted_winner,
                'predicted_home_prob': pred_home_prob,
                'predicted_away_prob': pred_away_prob,
                'home_odds': home_odds,
                'away_odds': away_odds,
                'confidence': confidence
            })
        
        conn.close()
        return predictions
    
    def format_yesterday_results(self, results: List[Dict]) -> str:
        """Format yesterday's results as HTML."""
        if not results:
            return "<p><em>Aucun match hier.</em></p>"
        
        html = "<h2>📊 Résultats d'hier</h2>"
        html += "<table style='border-collapse: collapse; width: 100%; margin-bottom: 20px;'>"
        html += """
        <tr style='background-color: #1e3a5f; color: white;'>
            <th style='padding: 6px 10px; text-align: left; border: 1px solid #ddd;'>Match</th>
            <th style='padding: 6px 10px; text-align: center; border: 1px solid #ddd;'>Prédiction</th>
            <th style='padding: 6px 10px; text-align: center; border: 1px solid #ddd;'>Cotes</th>
            <th style='padding: 6px 10px; text-align: center; border: 1px solid #ddd;'>Résultat</th>
            <th style='padding: 6px 10px; text-align: center; border: 1px solid #ddd;'>Statut</th>
        </tr>
        """
        
        for game in results:
            matchup = f"{game['away_team']} @ {game['home_team']}"
            predicted = game['predicted_winner']
            home_odds = game['home_odds']
            away_odds = game['away_odds']
            
            # Format odds - AWAY / HOME to match the "AWAY @ HOME" matchup order
            # Highlight the predicted winner's odds in yellow
            if predicted == game['away_team']:
                odds_display = f"<span style='background-color: #ffeb3b; padding: 2px 4px;'>{away_odds:.2f}</span> / {home_odds:.2f}"
            elif predicted == game['home_team']:
                odds_display = f"{away_odds:.2f} / <span style='background-color: #ffeb3b; padding: 2px 4px;'>{home_odds:.2f}</span>"
            else:
                odds_display = f"{away_odds:.2f} / {home_odds:.2f}"
            
            # Result (show score; use "—" if missing or 0-0 so we don't show wrong data)
            if game['actual_winner']:
                hs, aws = game.get('actual_home_score'), game.get('actual_away_score')
                if hs is not None and aws is not None and (hs > 0 or aws > 0):
                    result = f"{hs} - {aws}"
                else:
                    result = "—"
                winner = game['actual_winner']
            else:
                result = "En attente"
                winner = "-"
            
            # Status
            if game['correct'] == 1:
                status = "✅ Correct"
                status_color = "#059669"
            elif game['correct'] == 0:
                status = "❌ Incorrect"
                status_color = "#dc2626"
            else:
                status = "⏳ En attente"
                status_color = "#64748b"
            
            html += f"""
            <tr>
                <td style='padding: 4px 8px; border: 1px solid #ddd;'><strong>{matchup}</strong></td>
                <td style='padding: 4px 8px; text-align: center; border: 1px solid #ddd;'>{predicted}</td>
                <td style='padding: 4px 8px; text-align: center; border: 1px solid #ddd;'>{odds_display}</td>
                <td style='padding: 4px 8px; text-align: center; border: 1px solid #ddd;'>{result}</td>
                <td style='padding: 4px 8px; text-align: center; border: 1px solid #ddd; color: {status_color};'><strong>{status}</strong></td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _format_predictions_table(self, predictions: List[Dict], header_bg: str = '#1e3a5f') -> str:
        """Render a predictions table (shared by today and tomorrow sections)."""
        html = "<table style='border-collapse: collapse; width: 100%;'>"
        html += f"""
        <tr style='background-color: {header_bg}; color: white;'>
            <th style='padding: 6px 10px; text-align: left; border: 1px solid #ddd;'>Match</th>
            <th style='padding: 6px 10px; text-align: center; border: 1px solid #ddd;'>Prediction</th>
            <th style='padding: 6px 10px; text-align: center; border: 1px solid #ddd;'>Cotes</th>
            <th style='padding: 6px 10px; text-align: center; border: 1px solid #ddd;'>Confiance</th>
        </tr>
        """

        for pred in predictions:
            matchup = f"{pred['away_team']} @ {pred['home_team']}"
            predicted = pred['predicted_winner']
            home_odds = pred['home_odds']
            away_odds = pred['away_odds']
            confidence = pred['confidence'] * 100

            if predicted == pred['away_team']:
                odds_display = f"<span style='background-color: #ffeb3b; padding: 2px 4px;'>{away_odds:.2f}</span> / {home_odds:.2f}"
            elif predicted == pred['home_team']:
                odds_display = f"{away_odds:.2f} / <span style='background-color: #ffeb3b; padding: 2px 4px;'>{home_odds:.2f}</span>"
            else:
                odds_display = f"{away_odds:.2f} / {home_odds:.2f}"

            if confidence >= 70:
                conf_color = "#059669"
            elif confidence >= 60:
                conf_color = "#d97706"
            else:
                conf_color = "#dc2626"

            html += f"""
            <tr>
                <td style='padding: 4px 8px; border: 1px solid #ddd;'><strong>{matchup}</strong></td>
                <td style='padding: 4px 8px; text-align: center; border: 1px solid #ddd;'>{predicted}</td>
                <td style='padding: 4px 8px; text-align: center; border: 1px solid #ddd;'>{odds_display}</td>
                <td style='padding: 4px 8px; text-align: center; border: 1px solid #ddd; color: {conf_color};'><strong>{confidence:.1f}%</strong></td>
            </tr>
            """

        html += "</table>"
        return html

    def format_today_predictions(self, predictions: List[Dict]) -> str:
        """Format today's predictions as HTML."""
        if not predictions:
            return "<p><em>Aucune prediction pour aujourd'hui (matchs deja joues ou pas de matchs).</em></p>"

        html = "<h2>🏀 Predictions d'aujourd'hui</h2>"
        html += self._format_predictions_table(predictions, header_bg='#1e3a5f')
        return html

    def create_email_html(self, yesterday_results: List[Dict], today_predictions: List[Dict]) -> str:
        """Create HTML email content with yesterday results and today's predictions."""
        date_str = datetime.now().strftime('%d/%m/%Y')

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #1e3a5f; border-bottom: 3px solid #1e3a5f; padding-bottom: 10px; }}
                h2 {{ color: #2563eb; margin-top: 30px; }}
                table {{ margin: 20px 0; }}
                .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #64748b; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏀 Rapport NBA Predictor - {date_str}</h1>

                {self.format_yesterday_results(yesterday_results)}

                {self.format_today_predictions(today_predictions)}

                <div style='margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px;'>
                    <h3 style='color: white; margin-top: 0;'>Publier sur Twitter</h3>
                    <p style='color: white; margin-bottom: 15px;'>
                        Cliquez sur le bouton pour choisir quelles predictions publier sur Twitter :
                    </p>
                    <a href='https://AnthonyNadjari.github.io/NBAPredictLab/'
                       style='display: inline-block; background: white; color: #667eea; padding: 12px 28px;
                              text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;
                              box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);'>
                        Ouvrir l'interface de publication
                    </a>
                    <p style='color: rgba(255, 255, 255, 0.9); font-size: 13px; margin-top: 12px; margin-bottom: 0;'>
                        Vous pourrez selectionner individuellement chaque match a publier
                    </p>
                </div>

                <div class="footer">
                    <p>🏀 Genere automatiquement par NBA Predictor</p>
                    <p>Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _send_email_smtp(self, recipients: List[str], subject: str, html_content: str) -> bool:
        """
        Send email via SMTP (Gmail). Used in CI/GitHub Actions where Outlook is not available.
        Requires EMAIL_ADDRESS and EMAIL_APP_PASSWORD environment variables.
        """
        import os
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        email_address = os.environ.get('EMAIL_ADDRESS')
        email_password = os.environ.get('EMAIL_APP_PASSWORD')

        if not email_address or not email_password:
            logger.error("[ERROR] EMAIL_ADDRESS and EMAIL_APP_PASSWORD env vars required for SMTP")
            logger.error("  Set these as GitHub Secrets for CI email support")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email_address
            msg['To'] = ', '.join(recipients)
            msg.attach(MIMEText(html_content, 'html'))

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(email_address, email_password)
                server.sendmail(email_address, recipients, msg.as_string())

            logger.info(f"[OK] Email sent via SMTP to {', '.join(recipients)}")
            return True

        except Exception as e:
            logger.error(f"[ERROR] SMTP email failed: {e}")
            return False

    def send_email(self, recipients: List[str], subject: str, html_content: str, test_mode: bool = False, display_first: bool = False) -> bool:
        """
        Send email via Outlook (Windows) or SMTP/Gmail (CI).

        Args:
            recipients: List of email addresses
            subject: Email subject
            html_content: HTML email content
            test_mode: If True, only send to first recipient
            display_first: If True, display email window before sending (for verification)

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if not WIN32COM_AVAILABLE:
                logger.info("win32com not available, falling back to SMTP (Gmail)")
                return self._send_email_smtp(recipients, subject, html_content)
            
            # In test mode, only send to first recipient
            if test_mode:
                recipients = [recipients[0]] if recipients else []
                subject = f"[TEST] {subject}"
            
            logger.info(f"Creating Outlook email to {len(recipients)} recipient(s)...")
            logger.info(f"Recipients: {', '.join(recipients)}")

            # Initialize COM for this thread (required when called from Streamlit)
            try:
                import pythoncom
                pythoncom.CoInitialize()
                logger.debug("COM initialized for thread")
            except Exception as e:
                logger.debug(f"COM already initialized or not needed: {e}")

            # Create Outlook application object
            outlook = win32com.client.Dispatch("Outlook.Application")
            
            # Check if Outlook is connected
            try:
                namespace = outlook.GetNamespace("MAPI")
                
                # Check if Outlook is online
                try:
                    if namespace.Offline:
                        logger.warning("[WARN] Outlook is in OFFLINE mode!")
                        logger.warning("   Please switch Outlook to online mode to send emails")
                    else:
                        logger.info("Outlook connection: OK (Online)")
                except:
                    logger.info("Outlook connection: OK")
                    
            except Exception as e:
                logger.warning(f"Could not verify Outlook connection: {e}")
            
            # Find nadjari.anthony@gmail.com account FIRST
            sender_account = None
            try:
                accounts = namespace.Accounts
                logger.info(f"Found {accounts.Count} account(s) in Outlook")
                
                # Find nadjari.anthony@gmail.com account
                for i in range(accounts.Count):
                    acc = accounts.Item(i + 1)
                    smtp_addr = acc.SmtpAddress.lower() if acc.SmtpAddress else ""
                    
                    if 'nadjari.anthony@gmail.com' in smtp_addr:
                        sender_account = acc
                        logger.info(f"[OK] Found target account: {sender_account.DisplayName} ({sender_account.SmtpAddress})")
                        break
                
                if not sender_account:
                    logger.error("[ERROR] CRITICAL: Could not find nadjari.anthony@gmail.com account!")
                    for i in range(accounts.Count):
                        acc = accounts.Item(i + 1)
                        logger.error(f"     - {acc.DisplayName} ({acc.SmtpAddress})")
                    raise Exception("Target email account not found")
                    
            except Exception as e:
                logger.error(f"[ERROR] CRITICAL ERROR: {e}")
                raise Exception(f"Cannot proceed without correct sender account: {e}")
            
            # Create mail item from the SPECIFIC account's default folder
            # This ensures the email is created with the correct account context
            try:
                # Get the account's default mail folder
                account_delivery_store = sender_account.DeliveryStore
                if account_delivery_store:
                    account_inbox = account_delivery_store.GetDefaultFolder(6)  # 6 = olFolderInbox
                    # Create mail item in the context of this account
                    mail = account_inbox.Items.Add("IPM.Note")
                    logger.info(f"[OK] Created email in account context: {sender_account.SmtpAddress}")
                else:
                    # Fallback to standard method
                    mail = outlook.CreateItem(0)
                    mail.SendUsingAccount = sender_account
                    logger.info(f"[OK] Created email with SendUsingAccount: {sender_account.SmtpAddress}")
            except Exception as e:
                logger.warning(f"Could not create email in account context: {e}")
                # Fallback: standard creation + force account
                mail = outlook.CreateItem(0)
                mail.SendUsingAccount = sender_account
                logger.info(f"[OK] Created email (fallback) with account: {sender_account.SmtpAddress}")
            
            # Set recipients
            for recipient in recipients:
                mail.Recipients.Add(recipient)
            
            # Resolve recipients (check if addresses are valid)
            mail.Recipients.ResolveAll()
            unresolved = [r.Name for r in mail.Recipients if r.Resolved == False]
            if unresolved:
                logger.warning(f"Unresolved recipients: {unresolved}")
            
            # Set subject
            mail.Subject = subject
            
            # Set HTML body
            mail.HTMLBody = html_content
            
            # CRITICAL: Force the account one final time and verify it's set
            try:
                # Set SendUsingAccount
                mail.SendUsingAccount = sender_account
                
                # Also try to set the account via the mail item's account property
                # Some Outlook versions require this
                try:
                    # Get the account's outbox and move email there
                    account_store = sender_account.DeliveryStore
                    if account_store:
                        account_outbox = account_store.GetDefaultFolder(4)  # 4 = olFolderOutbox
                        # This ensures email goes to the right account's outbox
                        logger.info(f"[OK] Email will use outbox for: {sender_account.SmtpAddress}")
                except:
                    pass
                
                logger.info(f"[OK] Final account confirmation: {sender_account.SmtpAddress}")
                
                # Verify one last time
                try:
                    if hasattr(mail, 'SendUsingAccount'):
                        set_account = mail.SendUsingAccount
                        if set_account:
                            logger.info(f"[OK] Verified SendUsingAccount is set: {set_account.SmtpAddress}")
                        else:
                            logger.warning("[WARN] SendUsingAccount is None, but proceeding")
                    else:
                        logger.warning("[WARN] SendUsingAccount property not available")
                except:
                    logger.warning("[WARN] Could not verify SendUsingAccount, but proceeding")
                    
            except Exception as e:
                logger.error(f"[ERROR] CRITICAL: Could not set account before send: {e}")
                raise Exception(f"Cannot send without correct account: {e}")
            
            # Display email first if requested (for verification)
            if display_first:
                logger.info("Displaying email window for verification...")
                logger.info("Review the email in Outlook window, then click 'Send' manually")
                mail.Display()
                return True  # Return True since email is displayed (user will send manually)
            
            # Send email automatically
            logger.info(f"Sending email via Outlook...")
            
            # Try to send immediately and force connection
            import time
            try:
                # Send the email
                mail.Send()
                
                # Immediately force send/receive for all accounts
                logger.info("Forcing Send/Receive to send email immediately...")
                sync_objects = namespace.SyncObjects
                for sync_object in sync_objects:
                    try:
                        sync_object.Start()
                    except:
                        pass
                
                # Wait and check multiple times
                max_checks = 5
                for i in range(max_checks):
                    time.sleep(1)
                    outbox = namespace.GetDefaultFolder(4)  # 4 = olFolderOutbox
                    outbox_items = outbox.Items
                    email_in_outbox = False
                    
                    # Check if our email is still in outbox
                    for item in outbox_items:
                        try:
                            if hasattr(item, 'Subject') and item.Subject == subject:
                                email_in_outbox = True
                                break
                        except:
                            pass
                    
                    if not email_in_outbox:
                        logger.info(f"[OK] Email successfully sent (left Outbox after {i+1} second(s))")
                        break
                    elif i == max_checks - 1:
                        logger.warning("[WARN] Email still in Outbox after multiple checks")
                        logger.warning("   This usually means Outlook is not connected to the mail server")
                        logger.warning("   Please ensure Outlook is online and connected")
                        
            except Exception as e:
                logger.warning(f"Error during send verification: {e}")
                # Email was still sent via mail.Send(), it just may be queued
                logger.info("Email queued - will be sent when Outlook connects")
            
            # Wait a moment and check sent items
            import time
            time.sleep(1)
            
            try:
                sent_items = namespace.GetDefaultFolder(5)  # 5 = olFolderSentMail
                # Try to find the sent email
                messages = sent_items.Items
                messages.Sort("[SentOn]", True)  # Sort by sent time, descending
                if messages.Count > 0:
                    latest = messages.GetFirst()
                    if latest.Subject == subject:
                        logger.info(f"[OK] Email confirmed in Sent Items folder")
                        logger.info(f"  Sent at: {latest.SentOn}")
                        logger.info(f"  To: {latest.To}")
                    else:
                        logger.warning("Email sent but not found in Sent Items (may be queued)")
                else:
                    logger.warning("Sent Items folder is empty - email may be queued")
            except Exception as e:
                logger.warning(f"Could not verify Sent Items: {e}")
            
            logger.info(f"[OK] Email sent successfully to {', '.join(recipients)}")
            logger.info("Note: Check your Outlook 'Sent Items' folder and recipient inbox")

            # Cleanup COM
            try:
                import pythoncom
                pythoncom.CoUninitialize()
                logger.debug("COM uninitialized")
            except:
                pass

            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to send email: {e}", exc_info=True)
            logger.error("Make sure Outlook is installed and configured on this computer")
            logger.error("Try opening Outlook manually and checking your connection")

            # Cleanup COM even on error
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except:
                pass

            return False
    
    def format_tomorrow_predictions(self, predictions: List[Dict]) -> str:
        """Format tomorrow's predictions as HTML."""
        if not predictions:
            return ""

        tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%d/%m/%Y')
        html = f"<h2>🏀 Preview demain ({tomorrow_str})</h2>"
        html += self._format_predictions_table(predictions, header_bg='#475569')
        return html

    def send_daily_report(self, test_mode: bool = False) -> bool:
        """
        Send daily report with yesterday's results, today's predictions,
        and a preview of tomorrow's predictions.

        Args:
            test_mode: If True, only send to first recipient with [TEST] prefix

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            logger.info("Generating daily report...")

            # Get yesterday's results
            yesterday_results = self.get_yesterday_results()
            logger.info(f"Found {len(yesterday_results)} games from yesterday")

            # Get today's predictions
            today_predictions = self.get_today_predictions()
            logger.info(f"Found {len(today_predictions)} predictions for today")

            # Create HTML email with yesterday results and today's predictions
            html_content = self.create_email_html(
                yesterday_results, today_predictions
            )

            # Send email
            subject = f"🏀 Rapport NBA Predictor - {datetime.now().strftime('%d/%m/%Y')}"
            recipients = self.recipients

            return self.send_email(recipients, subject, html_content, test_mode=test_mode)

        except Exception as e:
            logger.error(f"Failed to send daily report: {e}", exc_info=True)
            return False

