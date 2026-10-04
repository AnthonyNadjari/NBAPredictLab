# NBAVision: reply bot for @NBAPredictLab

Finds NBA tweets with traction (keyword searches + a watchlist of big accounts),
drafts a reply with an LLM grounded in verified facts from this repo's data
(team records, last results, streaks, next game and win probability), validates
it, and posts it through a real Chrome profile on the home PC.

Merged from the former `NBAVisionEngine` repo (history kept under `vision/`).

## How it runs
- `.github/workflows/vision.yml` on the **self-hosted runner** (home PC): free minutes,
  same IP and browser profile as the manual login.
- Ticks every 15 min; `tools/schedule_gate.py` reads `docs/vision/schedule.json`
  (Paris time, editable from the control panel, tab "Compte X").
- After each session: `docs/vision/runs.json` (summary + replies) and
  `docs/vision/stats.json` (follower trend) are pushed for the control panel.

## Logging the bot in (once, on the runner PC)
```bash
python vision/tools/connect_x.py                  # opens Chrome, log in by hand
python vision/tools/connect_x.py --update-secret  # also refresh the backup cookie secret
```
The profile lives in `%LOCALAPPDATA%\NBAVision\profile` (outside the git checkout,
which `actions/checkout` wipes). Before this fix the refreshed session was saved
inside the checkout and deleted on every run, so each run fell back to cookies
exported in March until X invalidated them (May 2026).

## Settings
Repo variables: `LLM_MODEL` (Groq model for writing), `BOT_PROFILE_USERNAME`.
Secrets: `LLM_API_KEY` (Groq), `TWITTER_COOKIES_JSON` (backup cookies), `DISCORD_WEBHOOK_URL` (optional).
Targeting knobs in `config.py`: `SEARCH_MIN_FAVES`, `WATCHLIST_ACCOUNTS`, `MAX_REPLIES_PER_AUTHOR`.

Full original guide: `GUIDE.md`.
