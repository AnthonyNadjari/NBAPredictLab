# NBAVision Engine — Complete Guide

Everything you need to know in one page.

---

## What this is

A bot that runs on X (Twitter). It searches for NBA-related tweets, generates smart replies (via Groq LLM or templates), and posts them automatically. It runs via GitHub Actions on **your own PC** (self-hosted runner).

---

## Why it runs on your PC (not GitHub's servers)

This is the most important thing to understand.

When you log into x.com in your browser, X records **your IP address** alongside your session cookies. If those same cookies are suddenly used from a completely different IP (like a GitHub datacenter in Virginia), X sees that as suspicious and **invalidates the session immediately**. That's why the bot kept failing on GitHub-hosted runners.

A **self-hosted runner** means GitHub Actions sends the job to your PC. The code runs on your machine, with your home IP — the same IP where you exported the cookies. X sees no IP change, so the session stays valid much longer.

**When you trigger a run manually on GitHub**, the job is still executed on your PC. GitHub is just the trigger — it tells your runner "start this job now" and your PC does the actual work. The cookies come from the `TWITTER_COOKIES_JSON` GitHub secret, which gets injected as an environment variable into the job running on your machine.

---

## How cookies work (and why they expire)

1. You log into x.com in Chrome on your PC. X gives your browser cookies (`auth_token`, `ct0`, etc.).
2. You export those cookies and paste them into the GitHub secret `TWITTER_COOKIES_JSON`.
3. When the bot runs, it loads those cookies into a headless Chrome browser on your PC and pretends to be you.
4. After each successful run, the bot **saves refreshed cookies** to a file called `state.json` on your PC. The next run uses those fresh cookies instead of the original snapshot.

**Why cookies stop working:**
- X rotates cookies server-side. If the bot hasn't run in a while, the saved cookies go stale.
- If you log out of X in your browser, all sessions (including the bot's) get invalidated.
- If X detects automation patterns, it can force-expire the session.

**When you need to re-export:** If the bot fails with "session invalid", export fresh cookies and update the GitHub secret. This should be rare if the bot runs regularly (daily runs keep cookies fresh).

---

## One-time setup

### 1. Install the self-hosted runner on your PC

If you already did this, skip to step 2. You can check: go to **https://github.com/AnthonyNadjari/NBAVisionEngine/settings/actions/runners** — if you see a runner listed, it's already installed.

**Install from scratch:**

1. Open PowerShell and run:

```powershell
mkdir C:\actions-runner
cd C:\actions-runner
```

2. Go to **https://github.com/AnthonyNadjari/NBAVisionEngine/settings/actions/runners/new?arch=x64&os=win**
3. Follow the commands GitHub shows you (download zip, extract, run `.\config.cmd`)
4. When asked for the repo URL: `https://github.com/AnthonyNadjari/NBAVisionEngine`
5. When asked for the token: paste the token from the GitHub page
6. Accept defaults for runner name, labels, and work folder

**Install as Windows service (recommended — starts automatically):**

```powershell
cd C:\actions-runner
.\svc.cmd install
.\svc.cmd start
```

This makes the runner start automatically when Windows boots. You never need to think about it again.

**Or run manually (if you prefer):**

```powershell
cd C:\actions-runner
.\run.cmd
```

Leave the window open. The runner stops when you close it.

### 2. Make the runner start on boot (alternative to service)

If you didn't install as a service, you can add a startup shortcut:

1. Press **Windows + R**, type `shell:startup`, Enter
2. Right-click → **New** → **Shortcut**
3. Target: `C:\Windows\System32\cmd.exe /c "cd /d C:\actions-runner && run.cmd"`
4. Name it **GitHub Runner**

### 3. Set up the cookie secret

1. Install the **Cookie-Editor** extension in Chrome: https://chromewebstore.google.com/detail/cookie-editor/hlkenndednhfkekhgpnkiidkbajdkebn
2. Go to **x.com** and make sure you're logged in
3. Click Cookie-Editor icon → **Export** → **JSON** (copies to clipboard)
4. Go to **https://github.com/AnthonyNadjari/NBAVisionEngine/settings/secrets/actions**
5. Click **New repository secret** (or update existing)
6. Name: `TWITTER_COOKIES_JSON`
7. Value: paste the JSON array you just copied
8. Save

### 4. Optional secrets

Set these in the same place (GitHub repo → Settings → Secrets):

| Secret | What it does |
|--------|-------------|
| `LLM_API_KEY` | Groq API key for AI-generated replies. Without it, the bot uses template replies. Get one at https://console.groq.com |
| `DISCORD_WEBHOOK_URL` | Discord webhook URL. The bot will notify you when auth fails or a session finishes. Create one in your Discord server settings → Integrations → Webhooks. |

---

## How to run

### Manual run (recommended for testing)

1. Make sure the runner is running on your PC (service or `.\run.cmd`)
2. Go to **https://github.com/AnthonyNadjari/NBAVisionEngine/actions**
3. Click **NBAVision Engine** → **Run workflow**
4. Choose **Dry run = true** to test without posting, or **false** for a real run
5. Click **Run workflow**

The job gets picked up by your PC within seconds. Click on the run to see live logs.

### Automatic schedule

The bot runs automatically at the times you set in the dashboard (default: **11:00 AM UTC+1** and **4:00 PM UTC+1**). You can add, remove, or toggle time slots directly from the dashboard's **Schedule** card — just edit the times and click **Save**. A lightweight scheduler workflow runs every 15 minutes and triggers the main engine when the current UTC+1 time is within **20 minutes** of a slot (so delayed cron runs still match).

**Important:** All schedule times are in **UTC+1**.

**If scheduled runs don't fire:** (1) Check the **Actions** tab for "Scheduler" runs at :00 and :30 — do they run? Open a Scheduler run and look at the "Check schedule and trigger" step: it logs "API response HTTP status: 204" on success, or a different status (e.g. 403) and a warning if the trigger failed. (2) If you see 403 or the main workflow never triggers, go to **Settings → Actions → General → Workflow permissions** and set to **Read and write** for the default `GITHUB_TOKEN`. (3) The 15‑minute window allows for delayed cron. (4) Your PC and runner must be on at the scheduled times.

### Check if the runner is active

Go to **https://github.com/AnthonyNadjari/NBAVisionEngine/settings/actions/runners**

- **Green dot (Idle)**: Runner is connected and waiting for jobs. Good.
- **Yellow dot**: Runner is busy (a job is running).
- **Gray/offline**: Runner is not connected. Start it on your PC.

---

## What happens during a run

1. **Cookie check** — Validates that critical cookies aren't expired before starting
2. **Auth** — Opens headless Chrome with stealth patches, loads cookies, navigates to x.com, checks if logged in
3. **Scrape** — Picks 20 random NBA keywords, searches X, scrolls 5 times per search for more tweets
4. **Filter** — Removes tweets older than 2h, low-engagement tweets, URL-only tweets
5. **Score** — Ranks remaining tweets by engagement velocity + freshness + text quality, keeps top 40
6. **Reply** — For each top tweet: generates a reply via LLM (or template), validates it, pastes and posts it
7. **Repeat** — Sleeps ~30s, runs the next cycle. Continues until 60 replies or time runs out.
8. **Save** — Writes session log, saves refreshed cookies for next run, notifies Discord

---

## GitHub Pages dashboard

The dashboard at **https://anthonynadjari.github.io/NBAVisionEngine/** shows:

- **Live run status** — whether a run is active, queued, completed, or failed
- **Step-by-step progress** during a live run
- **Last run summary** — how many tweets were scraped, filtered, replied to, and the actual reply texts

**To make the dashboard work**, you need a GitHub Personal Access Token:

1. Go to **https://github.com/settings/tokens?type=beta**
2. Click **Generate new token**
3. Name: `NBAVision Dashboard`
4. Repository access: select **NBAVisionEngine** only
5. Permissions: **Actions: Read and write**, **Contents: Read and write**
6. Generate and copy the token (starts with `github_pat_...`)
7. Open the dashboard URL with `#` followed by your token:
   `https://anthonynadjari.github.io/NBAVisionEngine/#github_pat_YOUR_TOKEN_HERE`
8. The token is stored in your browser's session storage (cleared when you close the tab). You only need to do this once per browser session.

**Trigger runs from a bookmark (no redirect):** You can bookmark the dashboard with your token in the URL so you don't have to paste the token each time. After opening the dashboard once with `#YOUR_TOKEN` (or `?t=YOUR_TOKEN`), use **Copy bookmark link** on the page to copy the full URL, then paste it into a new tab and add that tab to your bookmarks. When you open the bookmark later, the dashboard loads with your token and you can click **Trigger run** or **Trigger dry run** immediately. Keep the bookmark private — the token is in the URL and in browser history. Use a fine-grained token with minimal scope (Actions: read+write, Contents: read) and rotate it if the bookmark is ever exposed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **"Session invalid"** | Re-export cookies from x.com (same PC as runner) and update the `TWITTER_COOKIES_JSON` secret |
| **Run stuck in "Queued"** | Runner isn't connected. Start `.\run.cmd` on your PC or check the service |
| **"Critical cookies expired"** | Same as session invalid — re-export cookies |
| **Screenshots in artifacts** | When something fails, check the **error-screenshots** artifact in the failed run for a screenshot of what the browser saw |
| **Runner offline** | On your PC: `cd C:\actions-runner` then `.\run.cmd`, or restart the service: `.\svc.cmd start` |
| **Want to test without posting** | Run manually with **Dry run = true** |

---

## Files overview

| File | Purpose |
|------|---------|
| `main.py` | Entry point — auth then run engine |
| `auth.py` | Cookie loading, stealth browser launch, session validation |
| `engine.py` | Main loop — scrape, filter, score, reply, post |
| `scraper.py` | Search X for keywords, extract tweets |
| `filter_tweets.py` | Remove tweets that don't meet criteria |
| `scorer.py` | Rank tweets by engagement + freshness |
| `llm_client.py` | Generate replies via Groq LLM or templates |
| `reply_validator.py` | Check replies aren't repetitive |
| `poster.py` | Navigate to tweet and post reply |
| `notify.py` | Discord webhook notifications |
| `session_log.py` | Write JSON session logs |
| `config.py` | All configuration and credentials |
| `.github/workflows/nbavision.yml` | GitHub Actions workflow (manual trigger only) |
| `.github/workflows/scheduler.yml` | Lightweight scheduler — checks every 30 min, triggers engine |
| `docs/index.html` | GitHub Pages dashboard with schedule editor |
| `docs/schedule.json` | Schedule config — editable from dashboard |
