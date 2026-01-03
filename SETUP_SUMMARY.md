# Quick Setup Summary - GitHub Pages Twitter Publishing

## What Was Added

✅ **7 New Files Created:**
1. `src/daily_games_exporter.py` - Exports predictions to JSON
2. `docs/index.html` - Beautiful web interface for publishing
3. `docs/pending_games.json` - Daily game data (auto-generated)
4. `.github/workflows/publish_thread.yml` - GitHub Actions workflow
5. `scripts/publish_single_thread.py` - Posts single thread to Twitter
6. `scripts/mark_published.py` - Marks games as published
7. `GITHUB_PAGES_SETUP.md` - Detailed setup guide

✅ **2 Files Modified:**
1. `src/email_reporter.py` - Added publish button in email
2. `daily_auto_prediction.py` - Added export step (Step 6)
3. `.gitignore` - Added temp image exclusion

## What You Need to Do AFTER Pushing

### Step 1: Push to GitHub (ONE TIME)

```bash
git add .
git commit -m "Add GitHub Pages Twitter publishing interface"
git push
```

### Step 2: Enable GitHub Pages (ONE TIME)

1. Go to your repo: https://github.com/YOUR_USERNAME/nba_predictor
2. **Settings** → **Pages**
3. Source: **main** branch, **/docs** folder
4. Save
5. Wait 2 minutes

### Step 3: Create Personal Access Token (ONE TIME)

1. Visit: https://github.com/settings/tokens
2. **Generate new token (classic)**
3. Name: `NBA Predictor Publishing`
4. Check: **repo** scope
5. **Generate token**
6. **COPY IT** (you won't see it again!)

### Step 4: Update 3 Placeholders (ONE TIME)

#### A. In `docs/index.html` (lines ~148-150):

```javascript
const GITHUB_TOKEN = 'ghp_xxxxxxxxxxxxxxxxxxxxx';  // Paste your token here
const GITHUB_USERNAME = 'your-github-username';    // Your GitHub username
const GITHUB_REPO = 'nba_predictor';              // Keep as is
```

#### B. In `src/email_reporter.py` (line ~286):

```python
<a href='https://your-github-username.github.io/nba_predictor/'
```

### Step 5: Add Twitter Secrets to GitHub (ONE TIME)

1. Your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** 5 times:
   - `TWITTER_API_KEY` = (value from your .env)
   - `TWITTER_API_SECRET` = (value from your .env)
   - `TWITTER_ACCESS_TOKEN` = (value from your .env)
   - `TWITTER_ACCESS_SECRET` = (value from your .env)
   - `TWITTER_BEARER_TOKEN` = (value from your .env)

### Step 6: Commit Your Changes Again

```bash
git add docs/index.html src/email_reporter.py
git commit -m "Configure GitHub Pages URL and token"
git push
```

### Step 7: Give Your Friend Access (OPTIONAL)

If they need to publish threads:

**Option A:** Settings → Collaborators → Add their GitHub username

**Option B:** Settings → General → Make repository Public

## Testing

### Test 1: Export Works

```bash
python -c "from src.daily_games_exporter import DailyGamesExporter; DailyGamesExporter().export_games_for_publishing()"
```

Check: `docs/pending_games.json` should have today's games

### Test 2: Web Interface Works

Visit: `https://YOUR_USERNAME.github.io/nba_predictor/`

You should see:
- Beautiful gradient header
- List of today's games
- "Publier le thread" buttons

### Test 3: Publishing Works

1. Click a "Publier le thread" button
2. Go to: Repository → **Actions** tab
3. Watch the workflow run (~1-2 minutes)
4. Check Twitter - thread should be posted!
5. Refresh the page - button should show "✓ Thread publié"

## How Your Friend Will Use It

1. **Receives email** each day (11 AM)
2. **Clicks** "Ouvrir l'interface de publication" button in email
3. **Sees** all today's games in a beautiful interface
4. **Clicks** "Publier le thread" on whichever games they want
5. **Waits** 1-2 minutes
6. **Checks** Twitter - thread is posted! 🎉

## Daily Flow (Automated)

```
11:00 AM - Daily Script Runs
   ├─ Makes predictions
   ├─ Saves to database
   ├─ Exports to docs/pending_games.json  ← NEW
   ├─ Sends email with GitHub Pages link   ← MODIFIED
   └─ Posts best prediction to Twitter

11:05 AM - Your Friend Receives Email
   └─ Clicks link → Opens web interface

11:06 AM - Friend Clicks "Publier le thread"
   └─ Triggers GitHub Actions

11:07 AM - GitHub Actions Runs
   ├─ Generates images
   ├─ Posts to Twitter
   ├─ Marks as published
   └─ Updates JSON

11:08 AM - Thread is Live on Twitter! ✓
```

## Important Notes

⚠️ **Security**: The GitHub token is stored in `docs/index.html`. This is OK if:
- Your repo is **private** (recommended)
- You trust all collaborators
- The token only has `repo` scope

⚠️ **Rate Limits**: Twitter has posting limits. Don't publish too many threads at once.

⚠️ **Token Expiration**: If you set an expiration on the GitHub token, you'll need to regenerate it and update `docs/index.html` when it expires.

✅ **Benefits**:
- Your friend can publish without accessing Streamlit
- Beautiful, user-friendly interface
- One-click publishing
- Automatic image generation
- No server needed (all runs on GitHub)
- Free forever (GitHub Actions free tier is generous)

## Need Help?

Read the detailed guide: `GITHUB_PAGES_SETUP.md`

## What Happens Next?

After you complete the setup:

1. **Tomorrow at 11 AM**: Daily script runs automatically
2. **Email sent** with link to GitHub Pages
3. **Your friend** can click the link and publish any game
4. **You're done!** No more manual work needed

---

**Total Setup Time**: ~10 minutes
**Future Maintenance**: None (fully automated)
