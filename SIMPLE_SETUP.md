# Quick Setup - Password-Only Publishing (No GitHub Account Needed!)

Your friend only needs a **password** - no GitHub account required!

## What You Get

✅ Beautiful web interface for publishing threads
✅ Your friend just enters a password (no GitHub account)
✅ Secure backend (GitHub token hidden server-side)
✅ Free forever (Vercel + GitHub free tiers)
✅ Zero maintenance after setup

## Setup Steps (15 minutes total)

### Step 1: Push Code to GitHub (2 minutes)

```bash
git add .
git commit -m "Add Vercel-powered Twitter publishing"
git push
```

### Step 2: Deploy Vercel Backend (5 minutes)

1. Go to https://vercel.com/signup
2. Sign up with GitHub
3. Click **"Add New..."** → **"Project"**
4. Import `nba_predictor`
5. Click **"Deploy"**
6. Copy your URL: `https://nba-predictor-xyz123.vercel.app`

### Step 3: Configure Vercel (3 minutes)

Go to Vercel → Settings → Environment Variables, add 3 variables:

```
PUBLISH_PASSWORD = CHANGE_ME  (pick any password)
GITHUB_TOKEN = ghp_xxx...  (create at github.com/settings/tokens - needs 'repo' scope)
GITHUB_REPO = YOUR_USERNAME/nba_predictor
```

Then: Deployments → Redeploy (without cache)

### Step 4: Enable GitHub Pages (2 minutes)

1. Your repo → Settings → Pages
2. Source: **main** branch, **/docs** folder
3. Save

### Step 5: Add Twitter Secrets to GitHub (2 minutes)

Your repo → Settings → Secrets and variables → Actions

Add these 5 secrets (values from your `.env` file):
- `TWITTER_API_KEY`
- `TWITTER_API_SECRET`
- `TWITTER_ACCESS_TOKEN`
- `TWITTER_ACCESS_SECRET`
- `TWITTER_BEARER_TOKEN`

### Step 6: Update 2 Files (1 minute)

**File 1:** `docs/index.html` (lines ~339-342)
```javascript
const PUBLISH_PASSWORD = 'CHANGE_ME';  // Same as Vercel
const VERCEL_APP_URL = 'https://nba-predictor-xyz123.vercel.app';  // Your Vercel URL
```

**File 2:** `src/email_reporter.py` (line ~286)
```python
<a href='https://YOUR_USERNAME.github.io/nba_predictor/'
```

Push changes:
```bash
git add docs/index.html src/email_reporter.py
git commit -m "Configure Vercel URL and password"
git push
```

## Done! ✅

Share with your friend:

```
Hey! To publish NBA threads:

1. Visit: https://YOUR_USERNAME.github.io/nba_predictor/
2. Click "Publier le thread"
3. Enter password: CHANGE_ME
4. Wait 1-2 minutes!

You'll also get the link in the daily email.
```

## Daily Workflow

```
11:00 AM - Daily script runs automatically
   ├─ Makes predictions
   ├─ Exports to JSON
   └─ Sends email

Your friend:
   ├─ Opens link from email
   ├─ Clicks "Publier le thread"
   ├─ Enters password
   └─ Done! Thread posts in 1-2 minutes
```

## Changing Password

1. Vercel: Settings → Environment Variables → Edit `PUBLISH_PASSWORD`
2. Update `PUBLISH_PASSWORD` in `docs/index.html`
3. Commit & push
4. Tell your friend

## Need More Help?

- **Detailed guide**: `VERCEL_SETUP.md`
- **Troubleshooting**: `VERCEL_SETUP.md` (bottom section)

---

**Total cost**: $0 (free tier)
**Maintenance**: None
**Friend needs**: Just a password!
