# Vercel Setup Guide - No GitHub Account Needed!

This setup uses Vercel (free tier) to create a serverless backend that handles authentication. Your friend only needs a simple password - no GitHub account required!

## Why Vercel?

- ✅ **Free forever** (generous free tier)
- ✅ **No server management** (serverless)
- ✅ **Secure** (GitHub token stored server-side)
- ✅ **Fast** (edge functions worldwide)
- ✅ **Simple** (deploys with one click)

Your friend just needs:
1. The website link (GitHub Pages)
2. A simple password you give them

## One-Time Setup (15 minutes)

### Part A: Deploy to Vercel (5 minutes)

#### Step 1: Create Vercel Account

1. Go to https://vercel.com/signup
2. Sign up with your **GitHub account** (easiest)
3. Authorize Vercel to access your repositories

#### Step 2: Import Project

1. Click **"Add New..."** → **"Project"**
2. Find `nba_predictor` in the list
3. Click **"Import"**
4. **Framework Preset**: None (or Other)
5. **Root Directory**: `.` (keep as is)
6. Click **"Deploy"**
7. Wait 1-2 minutes for deployment
8. You'll get a URL like: `https://nba-predictor-xyz123.vercel.app`

#### Step 3: Configure Environment Variables

1. Go to your project dashboard on Vercel
2. Click **Settings** → **Environment Variables**
3. Add these 3 variables:

**Variable 1:**
- Key: `PUBLISH_PASSWORD`
- Value: A simple password (e.g., `CHANGE_ME`)
- Environment: Production, Preview, Development
- Click **Save**

**Variable 2:**
- Key: `GITHUB_TOKEN`
- Value: Create a Personal Access Token:
  - Go to https://github.com/settings/tokens
  - **Generate new token (classic)**
  - Note: `NBA Predictor Vercel`
  - Expiration: **No expiration**
  - Scopes: Check **repo** only
  - Click **Generate token**
  - Copy the token (starts with `ghp_`)
  - Paste here
- Environment: Production, Preview, Development
- Click **Save**

**Variable 3:**
- Key: `GITHUB_REPO`
- Value: `YOUR_USERNAME/nba_predictor` (replace YOUR_USERNAME)
- Environment: Production, Preview, Development
- Click **Save**

#### Step 4: Redeploy (Force Rebuild)

1. Go to **Deployments** tab
2. Click the **...** menu on the latest deployment
3. Click **Redeploy**
4. Check **"Use existing Build Cache"** = OFF
5. Click **Redeploy**
6. Wait 1 minute

Your backend is now live! 🎉

### Part B: Configure GitHub & GitHub Pages (5 minutes)

#### Step 1: Enable GitHub Pages

1. Go to your repo: https://github.com/YOUR_USERNAME/nba_predictor
2. **Settings** → **Pages**
3. Source: **main** branch, **/docs** folder
4. Click **Save**
5. Wait 2 minutes
6. Your site: `https://YOUR_USERNAME.github.io/nba_predictor/`

#### Step 2: Add Twitter Secrets to GitHub Actions

1. Your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** 5 times:

   - `TWITTER_API_KEY` = (from your .env file)
   - `TWITTER_API_SECRET` = (from your .env file)
   - `TWITTER_ACCESS_TOKEN` = (from your .env file)
   - `TWITTER_ACCESS_SECRET` = (from your .env file)
   - `TWITTER_BEARER_TOKEN` = (from your .env file)

### Part C: Update Configuration Files (5 minutes)

#### Step 1: Update HTML File

Edit `docs/index.html` (lines ~339-342):

```javascript
const PUBLISH_PASSWORD = 'CHANGE_ME';  // Same as Vercel env var
const VERCEL_APP_URL = 'https://nba-predictor-xyz123.vercel.app';  // Your Vercel URL
```

#### Step 2: Update Email File

Edit `src/email_reporter.py` (line ~286):

```python
<a href='https://YOUR_USERNAME.github.io/nba_predictor/'  # Your GitHub Pages URL
```

#### Step 3: Commit and Push

```bash
git add .
git commit -m "Configure Vercel backend for passwordless publishing"
git push
```

## Testing

### Test 1: Backend is Working

Open browser console and run:

```javascript
fetch('https://YOUR_VERCEL_APP.vercel.app/api/publish', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    game_id: 'test',
    password: 'WRONG_PASSWORD'
  })
})
.then(r => r.json())
.then(console.log);
```

Expected response: `{success: false, error: "Invalid password"}`

### Test 2: Frontend is Working

1. Visit: `https://YOUR_USERNAME.github.io/nba_predictor/`
2. Should see nice interface
3. Click a "Publier le thread" button
4. Enter your password when prompted
5. Should trigger GitHub Actions

### Test 3: End-to-End

1. Run daily script to generate games:
   ```bash
   python -c "from src.daily_games_exporter import DailyGamesExporter; DailyGamesExporter().export_games_for_publishing()"
   git add docs/pending_games.json
   git commit -m "Add test games"
   git push
   ```

2. Visit GitHub Pages URL
3. Click "Publier le thread"
4. Enter password
5. Wait 1-2 minutes
6. Check Twitter!

## How It Works

```
User clicks "Publier le thread"
   ↓
JavaScript prompts for password
   ↓
Password validated locally (quick feedback)
   ↓
POST request to Vercel: /api/publish
   {
     game_id: "LAL_vs_BOS_2026-01-03",
     password: "CHANGE_ME"
   }
   ↓
Vercel Function (api/publish.js):
   1. Validates password (server-side)
   2. Uses stored GITHUB_TOKEN (secure!)
   3. Triggers GitHub Actions
   ↓
GitHub Actions runs:
   1. Generates images
   2. Posts to Twitter
   3. Marks as published
   ↓
Success! Thread is live ✅
```

## Security

✅ **GitHub Token**: Stored securely in Vercel (server-side)
✅ **Password**: Simple and changeable anytime
✅ **No GitHub Account**: Your friend doesn't need one
✅ **Rate Limiting**: Could add IP-based limits if needed
✅ **HTTPS**: All communication encrypted

## For Your Friend

Send them this message:

```
Hey! Here's how to publish NBA threads:

1. Visit: https://YOUR_USERNAME.github.io/nba_predictor/
2. Click "Publier le thread" on any game
3. When prompted, enter: CHANGE_ME
4. Wait 1-2 minutes - done!

You'll get the link in the daily email too.
No GitHub account needed - just the password!
```

## Changing the Password

Anytime you want to change it:

1. **Vercel**: Settings → Environment Variables → Edit `PUBLISH_PASSWORD`
2. **HTML**: Update `PUBLISH_PASSWORD` in `docs/index.html`
3. Commit and push
4. Tell your friend the new password

## Troubleshooting

### "Invalid password" error
- Password mismatch between HTML and Vercel
- Check both locations have the same value

### "Server configuration error"
- Vercel environment variables not set
- Missing GITHUB_TOKEN or GITHUB_REPO
- Go to Vercel → Settings → Environment Variables

### "GitHub API error (401)"
- GitHub token invalid or expired
- Regenerate token and update in Vercel

### "CORS error" in browser
- Vercel function not deployed correctly
- Redeploy from Vercel dashboard

### Thread not posting
- Check GitHub Actions logs
- Verify Twitter secrets in GitHub
- Check Twitter rate limits

## Cost

**Vercel Free Tier:**
- 100 GB bandwidth/month
- 100 hours serverless execution/month
- Unlimited requests

**Your usage:**
- ~10 requests/day (when publishing)
- Each request: ~0.1 seconds
- **Total: FREE** (well within limits)

## Advantages Over GitHub Token in HTML

| Feature | Old (GitHub Token in HTML) | New (Vercel Backend) |
|---------|---------------------------|----------------------|
| Friend needs GitHub account | ✅ Yes | ❌ No |
| Friend needs PAT token | ✅ Yes | ❌ No |
| Token visible in HTML | ⚠️ Yes (if repo public) | ✅ No (server-side) |
| Easy to change password | ❌ Need to edit HTML | ✅ Vercel dashboard |
| Setup complexity | Simple | Medium |
| Ongoing maintenance | None | None |

## Files Added

```
nba_predictor/
├── api/
│   └── publish.js          # Vercel serverless function
└── vercel.json            # Vercel configuration
```

## Files Modified

```
docs/index.html            # Uses Vercel backend instead of direct GitHub API
src/email_reporter.py      # (No change needed - still uses GitHub Pages link)
```

---

**That's it!** Your friend can now publish with just a password - no GitHub account needed.
