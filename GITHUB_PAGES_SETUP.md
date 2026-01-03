# GitHub Pages Twitter Publishing Setup

This guide explains how to set up the GitHub Pages interface that allows you to publish prediction threads to Twitter by clicking buttons.

## Overview

Your daily email will now include a link to a web interface where you (or your friend) can select which games to publish to Twitter. When a button is clicked, it triggers GitHub Actions to generate images and post the thread.

## One-Time Setup (After First Push)

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages** (in left sidebar)
3. Under **Source**, select:
   - **Branch**: `main`
   - **Folder**: `/docs`
4. Click **Save**
5. Wait 1-2 minutes, then visit: `https://YOUR_USERNAME.github.io/nba_predictor/`

### 2. Create GitHub Personal Access Token

Your friend needs a GitHub account and a Personal Access Token to trigger the workflow:

1. Go to https://github.com/settings/tokens
2. Click **Generate new token** → **Generate new token (classic)**
3. Give it a name: `NBA Predictor Publishing`
4. Check the **repo** scope (full control of private repositories)
5. Set expiration: **No expiration** (or 1 year)
6. Click **Generate token**
7. **COPY THE TOKEN** - you won't see it again!

### 3. Update the HTML File

Edit `docs/index.html` and replace these placeholders:

```javascript
// Line ~148-150
const GITHUB_TOKEN = 'YOUR_GITHUB_PAT_HERE';  // Replace with the token from step 2
const GITHUB_USERNAME = 'YOUR_USERNAME';       // Replace with your GitHub username
const GITHUB_REPO = 'nba_predictor';          // Keep as is
```

**Important**: If your repository is public, consider using a more secure method (see Security Notes below).

### 4. Update Email Link

Edit `src/email_reporter.py` line 286:

```python
<a href='https://YOUR_USERNAME.github.io/nba_predictor/'  # Replace YOUR_USERNAME
```

### 5. Add Twitter API Secrets to GitHub

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each:
   - `TWITTER_API_KEY`
   - `TWITTER_API_SECRET`
   - `TWITTER_ACCESS_TOKEN`
   - `TWITTER_ACCESS_SECRET`
   - `TWITTER_BEARER_TOKEN`
4. Use the same values from your `.env` file

### 6. Give Your Friend Access (Optional)

If your friend wants to publish threads:

**Option A: Make Repository Public**
- Settings → General → Danger Zone → Change visibility → Public
- Your friend can use the interface with their token

**Option B: Add as Collaborator**
- Settings → Collaborators → Add people
- Enter their GitHub username
- They'll get an email invitation

## How It Works

### Daily Flow

1. **11 AM**: `daily_auto_prediction.py` runs:
   - Makes predictions
   - Saves to database
   - **Exports to `docs/pending_games.json`** ← NEW
   - Sends email with link to GitHub Pages

2. **Email Recipient** clicks "Ouvrir l'interface de publication"

3. **Web Interface** loads from GitHub Pages:
   - Shows all today's games
   - Each has a "Publier le thread" button

4. **User Clicks Button**:
   - JavaScript triggers GitHub Actions via API
   - Button shows "Publication en cours..."

5. **GitHub Actions Runs** (1-2 minutes):
   - `scripts/publish_single_thread.py` generates images
   - Posts thread to Twitter
   - `scripts/mark_published.py` marks game as published
   - Commits updated JSON

6. **User Refreshes Page**:
   - Button now shows "✓ Thread publié" (disabled)

## Files Created

```
nba_predictor/
├── .github/
│   └── workflows/
│       └── publish_thread.yml          # GitHub Actions workflow
├── docs/
│   ├── index.html                      # Web interface (GitHub Pages)
│   └── pending_games.json              # Generated daily (game data)
├── scripts/
│   ├── publish_single_thread.py        # Posts one thread to Twitter
│   └── mark_published.py               # Marks game as published
└── src/
    └── daily_games_exporter.py         # Exports games to JSON
```

## Modified Files

- `src/email_reporter.py` - Added publish button section
- `daily_auto_prediction.py` - Added game export step

## Testing

### Test Locally

```bash
# Test the exporter
python -c "from src.daily_games_exporter import DailyGamesExporter; DailyGamesExporter().export_games_for_publishing()"

# Check the output
type docs\pending_games.json

# Test publishing a single thread (dry run first)
# Replace GAME_ID with an actual ID from the JSON
python scripts/publish_single_thread.py LAL_vs_BOS_2026-01-03
```

### Test GitHub Actions

1. Push all changes
2. Visit your GitHub Pages URL
3. Click a "Publier le thread" button
4. Go to **Actions** tab on GitHub
5. Watch the "Publish Twitter Thread" workflow run
6. Check Twitter for the posted thread

## Security Notes

### Storing GitHub Token in HTML

The current implementation stores the GitHub Personal Access Token directly in `docs/index.html`. This is acceptable if:

- Your repository is **private** (token only visible to collaborators)
- You trust all collaborators
- The token only has `repo` scope (limited damage if leaked)

### More Secure Alternatives

**Option 1: OAuth Apps** (Complex)
- Create a GitHub OAuth App
- Use OAuth flow instead of PAT
- Requires backend server

**Option 2: Separate Private Repo** (Recommended for public repos)
- Keep `docs/index.html` in a separate **private** repository
- Your friend clones it locally
- Opens `index.html` as a local file (still works!)

**Option 3: Browser Extension** (Advanced)
- Build a simple browser extension
- Store token securely in extension storage
- Inject publish buttons into a public page

For most use cases, the PAT in a private repo is sufficient.

## Troubleshooting

### "Configuration incomplète" Error
- You forgot to update `GITHUB_TOKEN` or `GITHUB_USERNAME` in `docs/index.html`

### "GitHub API error (401)" or "(403)"
- Token is invalid or expired
- Token doesn't have `repo` scope
- Regenerate token and update HTML

### "GitHub API error (404)"
- Repository name is wrong in HTML
- Repository is private and user doesn't have access

### Thread Not Posted
- Check GitHub Actions logs: Repository → Actions → Latest run
- Verify Twitter API secrets are set correctly
- Check if Twitter rate limits are exceeded

### Email Link Doesn't Work
- You forgot to update `YOUR_USERNAME` in `src/email_reporter.py`
- GitHub Pages isn't enabled yet (Settings → Pages)

### Button Stays "Publishing..." Forever
- Check browser console (F12) for JavaScript errors
- Refresh the page and check GitHub Actions status

## Manual Testing Without Email

You can test the interface directly:

1. Run the exporter manually:
   ```bash
   python -c "from src.daily_games_exporter import DailyGamesExporter; DailyGamesExporter().export_games_for_publishing()"
   ```

2. Commit and push `docs/pending_games.json`

3. Visit your GitHub Pages URL directly

4. Click buttons to test publishing

## Future Enhancements

Possible improvements:

- Add authentication layer (password protection)
- Store publish history (who published what, when)
- Add preview mode (see thread before posting)
- Bulk publish (publish all games at once)
- Schedule publishing (delay until specific time)
- Edit tweets before publishing
- Mobile-responsive improvements

---

**That's it!** Once set up, you just need to:
1. Push the code once
2. Update the 3 placeholders (token, username in HTML, username in email)
3. Add Twitter secrets to GitHub
4. Your friend can start publishing with one click
