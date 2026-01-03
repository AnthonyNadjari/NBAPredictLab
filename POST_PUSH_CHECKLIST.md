# Post-Push Checklist

Complete these steps AFTER you push the code to GitHub.

## ☐ 1. Enable GitHub Pages (2 minutes)

- [ ] Go to: https://github.com/YOUR_USERNAME/nba_predictor/settings/pages
- [ ] Under "Source":
  - [ ] Branch: `main`
  - [ ] Folder: `/docs`
- [ ] Click **Save**
- [ ] Wait 2 minutes for deployment
- [ ] Visit: https://YOUR_USERNAME.github.io/nba_predictor/
- [ ] Confirm page loads (should show "Configuration requise" notice)

## ☐ 2. Create GitHub Personal Access Token (3 minutes)

- [ ] Go to: https://github.com/settings/tokens
- [ ] Click **Generate new token** → **Generate new token (classic)**
- [ ] Settings:
  - [ ] Note: `NBA Predictor Publishing`
  - [ ] Expiration: **No expiration** (or 1 year)
  - [ ] Scopes: Check **repo** (full control)
- [ ] Click **Generate token**
- [ ] **COPY THE TOKEN** (save it somewhere safe - you won't see it again)

Example token format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

## ☐ 3. Update HTML File (2 minutes)

- [ ] Open: `docs/index.html`
- [ ] Find lines ~148-150
- [ ] Replace:
  ```javascript
  const GITHUB_TOKEN = 'YOUR_GITHUB_PAT_HERE';  // ← Paste token here
  const GITHUB_USERNAME = 'YOUR_USERNAME';       // ← Your GitHub username
  const GITHUB_REPO = 'nba_predictor';          // ← Keep as is
  ```
- [ ] Save file

## ☐ 4. Update Email Link (1 minute)

- [ ] Open: `src/email_reporter.py`
- [ ] Find line ~286
- [ ] Replace:
  ```python
  <a href='https://YOUR_USERNAME.github.io/nba_predictor/'  # ← Your username
  ```
- [ ] Save file

## ☐ 5. Add Twitter API Secrets to GitHub (3 minutes)

- [ ] Go to: https://github.com/YOUR_USERNAME/nba_predictor/settings/secrets/actions
- [ ] Click **New repository secret** for EACH:

### Secret 1:
- [ ] Name: `TWITTER_API_KEY`
- [ ] Value: (copy from your `.env` file - the value of `TW_API_KEY`)
- [ ] Click **Add secret**

### Secret 2:
- [ ] Name: `TWITTER_API_SECRET`
- [ ] Value: (copy from `.env` - the value of `TW_API_SECRET`)
- [ ] Click **Add secret**

### Secret 3:
- [ ] Name: `TWITTER_ACCESS_TOKEN`
- [ ] Value: (copy from `.env` - the value of `TW_ACCESS_TOKEN`)
- [ ] Click **Add secret**

### Secret 4:
- [ ] Name: `TWITTER_ACCESS_SECRET`
- [ ] Value: (copy from `.env` - the value of `TW_ACCESS_SECRET`)
- [ ] Click **Add secret**

### Secret 5:
- [ ] Name: `TWITTER_BEARER_TOKEN`
- [ ] Value: (copy from `.env` - the value of `TW_BEARER_TOKEN`)
- [ ] Click **Add secret**

- [ ] Verify all 5 secrets are listed

## ☐ 6. Commit Configuration Changes (1 minute)

```bash
git add docs/index.html src/email_reporter.py
git commit -m "Configure GitHub Pages URL and authentication"
git push
```

- [ ] Run the commands above
- [ ] Confirm push successful

## ☐ 7. Test the System (5 minutes)

### Test A: Export Works
```bash
python -c "from src.daily_games_exporter import DailyGamesExporter; DailyGamesExporter().export_games_for_publishing()"
```
- [ ] Command runs without errors
- [ ] Check `docs/pending_games.json` exists and has games

### Test B: Web Interface Works
- [ ] Visit: https://YOUR_USERNAME.github.io/nba_predictor/
- [ ] Page loads with nice gradient design
- [ ] Shows today's games (if any)
- [ ] Shows "Publier le thread" buttons

### Test C: Publishing Works (CAREFUL - This posts to Twitter!)
- [ ] Click a "Publier le thread" button
- [ ] Button changes to "⏳ Publication en cours..."
- [ ] Go to: https://github.com/YOUR_USERNAME/nba_predictor/actions
- [ ] See "Publish Twitter Thread" workflow running
- [ ] Wait 1-2 minutes for completion
- [ ] Check Twitter - thread should be posted
- [ ] Refresh web page - button should show "✓ Thread publié"

## ☐ 8. Give Friend Access (Optional, 2 minutes)

Choose ONE option:

### Option A: Add as Collaborator (Private Repo)
- [ ] Go to: https://github.com/YOUR_USERNAME/nba_predictor/settings/access
- [ ] Click **Add people**
- [ ] Enter friend's GitHub username
- [ ] Role: **Write**
- [ ] Click **Add [username] to this repository**
- [ ] Friend receives email invitation
- [ ] Friend accepts invitation

### Option B: Make Repository Public
- [ ] Go to: https://github.com/YOUR_USERNAME/nba_predictor/settings
- [ ] Scroll to **Danger Zone**
- [ ] Click **Change visibility**
- [ ] Select **Make public**
- [ ] Type repository name to confirm
- [ ] Click **I understand...**

⚠️ **Note**: If repo is public, anyone with the URL can see your code (but not the GitHub token or Twitter secrets - those are still secure).

## ☐ 9. Share Link with Friend (1 minute)

Send your friend:

```
Hey! Here's how to publish NBA prediction threads to Twitter:

1. Visit: https://YOUR_USERNAME.github.io/nba_predictor/
2. Click "Publier le thread" on any game
3. Wait 1-2 minutes
4. Check Twitter!

You'll also get this link in the daily email each morning.
```

## ☐ 10. Verify Daily Automation (Next Day)

Tomorrow at 11 AM:

- [ ] Daily script runs automatically
- [ ] `docs/pending_games.json` updated with new games
- [ ] Email sent with link to GitHub Pages
- [ ] Your friend can click link and publish

---

## ✅ Setup Complete!

Once all checkboxes are ticked, the system is fully operational.

**What happens daily:**
- 11:00 AM: Predictions made, JSON exported, email sent
- Your friend: Receives email, clicks link, publishes threads
- GitHub Actions: Automatically posts to Twitter

**Zero maintenance required!**

---

## Troubleshooting

### "Configuration incomplète" error on webpage
→ You didn't update `GITHUB_TOKEN` or `GITHUB_USERNAME` in `docs/index.html`

### "GitHub API error (401)" when clicking button
→ Token is invalid. Regenerate and update `docs/index.html`

### "GitHub API error (404)" when clicking button
→ Repository name is wrong, or friend doesn't have access

### Thread not posted after GitHub Actions runs
→ Check Actions logs for errors
→ Verify Twitter secrets are correct
→ Check Twitter rate limits

### Email link doesn't work
→ You didn't update `YOUR_USERNAME` in `src/email_reporter.py`
→ GitHub Pages not enabled yet

Need more help? Read `GITHUB_PAGES_SETUP.md` for detailed troubleshooting.
