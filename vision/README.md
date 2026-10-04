# NBAVision: reply bot for @NBAPredictLab

Finds NBA tweets with traction (keyword searches + a watchlist of big accounts),
drafts a reply with an LLM grounded in verified facts from this repo's data
(team records, last results, streaks, next game and win probability), validates
it, and posts it through a headless Chrome logged in to X.

Merged from the former `NBAVisionEngine` repo (history kept under `vision/`).

## How it runs
- `.github/workflows/vision.yml` on GitHub-hosted runners (free: the repo is public).
  Nothing runs on a home PC.
- Ticks every 15 min; `tools/schedule_gate.py` reads `docs/vision/schedule.json`
  (Paris time, editable in the control panel, tab "Compte X"). A slot runs once
  (tracked in `docs/vision/runs.json`), up to 45 min late.
- The X session is kept between runs as an AES-256 encrypted file in the Actions
  cache (key: secret `X_STATE_KEY`); every session refreshes it.
- After each session: `docs/vision/runs.json` (summary + replies with the original
  tweet) and `docs/vision/stats.json` (follower trend) are pushed for the panel.

## Connecting the bot to X
Control panel → Compte X → "Connecter le bot de réponses à X": paste a Cookie-Editor
JSON export (or `auth_token=...; ct0=...`) from a browser logged in as @NBAPredictLab.
The value is sealed with the repo's public key and stored as the
`TWITTER_COOKIES_JSON` secret. Then run "Essai à blanc".
Same panel for the Groq key (`LLM_API_KEY`): without it no session runs (the bot
never posts canned replies).

`tools/connect_x.py` is an optional alternative (interactive login on a desktop).

## Checking reply quality before going live
```bash
LLM_API_KEY=<groq key> python vision/tools/reply_lab.py          # sample tweets, nothing posted
LLM_API_KEY=<groq key> python vision/tools/reply_lab.py my.txt   # your own tweets, "@author: text" per line
```
Prints the facts given to the model, its decision, and whether the validator would
let it through. `LLM_BASE_URL` points it at any OpenAI-compatible endpoint.

## Settings
Repo variables: `LLM_MODEL` (Groq model for writing), `BOT_PROFILE_USERNAME`.
Secrets: `LLM_API_KEY`, `TWITTER_COOKIES_JSON`, `X_STATE_KEY`, `DISCORD_WEBHOOK_URL` (optional).
Targeting knobs in `config.py`: `SEARCH_MIN_FAVES`, `WATCHLIST_ACCOUNTS`, `MAX_REPLIES_PER_AUTHOR`.

Original guide: `GUIDE.md`.
