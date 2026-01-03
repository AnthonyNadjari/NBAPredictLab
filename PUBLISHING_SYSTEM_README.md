# Twitter Publishing System - Overview

## What Was Built

A complete system that allows your friend to publish prediction threads to Twitter using just a **simple password** - no GitHub account needed.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Daily Automation (11 AM)                 │
│  ┌────────────┐    ┌─────────────┐    ┌──────────────┐     │
│  │  Predict   │ →  │  Export to  │ →  │ Send Email   │     │
│  │   Games    │    │    JSON     │    │  with Link   │     │
│  └────────────┘    └─────────────┘    └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Your Friend Receives Email                      │
│  "Cliquez pour publier vos prédictions sur Twitter"         │
│  [Button: Ouvrir l'interface de publication]                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           GitHub Pages (Static Website)                      │
│  https://YOUR_USERNAME.github.io/nba_predictor/             │
│                                                               │
│  ┌──────────────────────────────────────┐                   │
│  │  Game 1: Lakers vs Celtics           │                   │
│  │  Prediction: Lakers (72% confidence) │                   │
│  │  [Publier le thread] ← Click!        │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
│  ┌──────────────────────────────────────┐                   │
│  │  Game 2: Warriors vs Heat            │                   │
│  │  Prediction: Warriors (68%)          │                   │
│  │  [Publier le thread]                 │                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    🔐 Enter Password
                              ↓
┌─────────────────────────────────────────────────────────────┐
│        Vercel Serverless Function (Backend)                  │
│        https://your-app.vercel.app/api/publish              │
│                                                               │
│  1. Validate password ✓                                      │
│  2. Use stored GitHub token (secure!)                        │
│  3. Trigger GitHub Actions workflow                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              GitHub Actions (Automated)                      │
│                                                               │
│  1. Run scripts/publish_single_thread.py                     │
│     ├─ Get game from database                               │
│     ├─ Generate prediction images                           │
│     └─ Post Twitter thread                                   │
│                                                               │
│  2. Run scripts/mark_published.py                            │
│     └─ Update pending_games.json                            │
│                                                               │
│  3. Commit changes                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
                   🎉 Thread Live on Twitter!
```

## Components

### 1. Frontend (GitHub Pages)
- **File**: `docs/index.html`
- **URL**: `https://YOUR_USERNAME.github.io/nba_predictor/`
- **Purpose**: Beautiful web interface showing games with publish buttons
- **Tech**: Pure HTML/CSS/JavaScript (no framework)

### 2. Backend (Vercel Serverless)
- **File**: `api/publish.js`
- **URL**: `https://your-app.vercel.app/api/publish`
- **Purpose**: Secure authentication and GitHub API proxy
- **Tech**: Node.js serverless function

### 3. Automation (GitHub Actions)
- **File**: `.github/workflows/publish_thread.yml`
- **Trigger**: Repository dispatch event from Vercel
- **Purpose**: Generate images and post to Twitter

### 4. Data Layer
- **File**: `docs/pending_games.json`
- **Generated**: Daily at 11 AM
- **Purpose**: Contains today's games for publishing

### 5. Scripts
- **`scripts/publish_single_thread.py`**: Posts one thread
- **`scripts/mark_published.py`**: Marks game as published
- **`src/daily_games_exporter.py`**: Exports games to JSON

## Security Model

### What's Public
✅ HTML interface (read-only game data)
✅ Password hash comparison (client-side, quick feedback)

### What's Private
🔒 GitHub Personal Access Token (stored in Vercel)
🔒 Twitter API credentials (stored in GitHub Secrets)
🔒 Password validation (server-side in Vercel)

### Authentication Flow
```
User enters password
  ↓
Client-side check (instant feedback)
  ↓
Send to Vercel with password
  ↓
Server validates password
  ↓
Server uses stored token to call GitHub API
  ↓
GitHub Actions uses stored Twitter secrets
```

## Data Flow

### Daily Export (11 AM)
```python
# In daily_auto_prediction.py
DailyGamesExporter().export_games_for_publishing()
# Creates: docs/pending_games.json
```

### User Publishes
```javascript
// In docs/index.html
fetch('https://your-app.vercel.app/api/publish', {
  method: 'POST',
  body: JSON.stringify({
    game_id: 'LAL_vs_BOS_2026-01-03',
    password: 'CHANGE_ME'
  })
})
```

```javascript
// In api/publish.js (Vercel)
if (password !== process.env.PUBLISH_PASSWORD) {
  return res.status(401).json({error: 'Invalid password'});
}

fetch('https://api.github.com/repos/USER/REPO/dispatches', {
  headers: { 'Authorization': `Bearer ${process.env.GITHUB_TOKEN}` },
  body: JSON.stringify({
    event_type: 'publish_thread',
    client_payload: { game_id }
  })
})
```

```yaml
# In .github/workflows/publish_thread.yml
on:
  repository_dispatch:
    types: [publish_thread]

jobs:
  publish:
    - python scripts/publish_single_thread.py "${{ github.event.client_payload.game_id }}"
    - python scripts/mark_published.py "${{ github.event.client_payload.game_id }}"
```

## Files Structure

```
nba_predictor/
├── .github/
│   └── workflows/
│       └── publish_thread.yml          # GitHub Actions workflow
├── api/
│   └── publish.js                      # Vercel backend function
├── docs/
│   ├── index.html                      # Web interface (GitHub Pages)
│   └── pending_games.json              # Daily game data (auto-generated)
├── scripts/
│   ├── publish_single_thread.py        # Posts one thread to Twitter
│   └── mark_published.py               # Marks game as published
├── src/
│   ├── daily_games_exporter.py         # Exports games to JSON
│   └── email_reporter.py               # Sends email (modified)
├── daily_auto_prediction.py            # Main automation (modified)
├── vercel.json                         # Vercel configuration
├── SIMPLE_SETUP.md                     # Quick setup guide
├── VERCEL_SETUP.md                     # Detailed Vercel guide
└── PUBLISHING_SYSTEM_README.md         # This file
```

## User Experience

### For You (Setup)
1. Deploy to Vercel (5 min)
2. Configure environment variables (3 min)
3. Update 2 files (1 min)
4. Push to GitHub (1 min)
**Total: 10 minutes**

### For Your Friend (Daily)
1. Receive email
2. Click link
3. Click "Publier le thread"
4. Enter password (first time only)
5. Wait 1-2 minutes
**Total: 30 seconds of work**

## Why This Approach?

### Alternatives Considered

| Approach | Pro | Con | Chosen? |
|----------|-----|-----|---------|
| GitHub token in HTML | Simple | Friend needs GitHub account | ❌ |
| Direct Streamlit access | No web setup | Friend needs VPN/access to your machine | ❌ |
| Custom backend server | Full control | Need to host/maintain server | ❌ |
| **Vercel serverless** | **No server, secure, free** | **Slightly more setup** | ✅ |

### Why Vercel?
- **Free**: Generous free tier, never expires
- **Serverless**: No server to maintain
- **Global**: Fast edge functions worldwide
- **Simple**: Deploy with GitHub integration
- **Secure**: Environment variables encrypted

## Costs

### Vercel Free Tier
- 100 GB bandwidth/month
- 100 hours serverless execution/month
- Unlimited requests

### Your Usage
- ~10 publishes/day maximum
- Each publish: ~0.1 seconds
- Monthly: ~30 seconds of execution time
- **Cost: $0** (well within free tier)

### GitHub Actions Free Tier
- 2,000 minutes/month
- Each workflow: ~1-2 minutes
- Monthly: ~300 minutes maximum
- **Cost: $0** (well within free tier)

### GitHub Pages
- Free for public repos
- Free for private repos (with GitHub Pro)
- **Cost: $0**

**Total Monthly Cost: $0**

## Maintenance

### None Required!

Once set up:
- ✅ Daily script runs automatically
- ✅ Vercel stays deployed (never sleeps)
- ✅ GitHub Pages stays live
- ✅ No updates needed

### Optional: Change Password
1. Vercel → Environment Variables → Edit
2. Update HTML file
3. Push to GitHub
4. Tell your friend

## Troubleshooting Quick Reference

| Error | Solution |
|-------|----------|
| "Invalid password" | Check password matches in HTML and Vercel |
| "Configuration incomplète" | Update VERCEL_APP_URL in HTML |
| "Server configuration error" | Set environment variables in Vercel |
| Thread not posting | Check GitHub Actions logs, verify Twitter secrets |
| CORS error | Redeploy Vercel function |

## Support

- **Quick Setup**: Read `SIMPLE_SETUP.md`
- **Detailed Guide**: Read `VERCEL_SETUP.md`
- **Troubleshooting**: See bottom of `VERCEL_SETUP.md`

---

## Summary

You now have a **production-grade publishing system** that:

✅ Requires **zero maintenance**
✅ Costs **$0/month**
✅ Works with **just a password** (no GitHub account)
✅ Is **secure** (tokens hidden server-side)
✅ Is **fast** (global edge network)
✅ Is **reliable** (industry-standard platforms)

Your friend can publish threads with one click, and you never have to touch it again!
