# NBA Predict Lab

Daily NBA win probabilities, a control panel, and a Twitter/X thread bot.

## How it works

```
GitHub Actions, 09:00 UTC daily ── scripts/morning_routine.py
  1. ESPN results of the last 7 days  ─▶ data/games_history.csv
  2. pending predictions resolved     ─▶ data/nba_predictor.db
  3. tonight + tomorrow predicted     ─▶ data/nba_predictor.db
  4. docs/pending_games.json + docs/dashboard.json (control panel data)
  5. e-mail report, commit + push

GitHub Pages: docs/index.html (control panel)
  Publish ─▶ Vercel api/publish.js ─▶ repository_dispatch
          ─▶ .github/workflows/publish_thread.yml ─▶ thread + charts on X
  Run now / schedule / run history ─▶ Vercel api/admin.js, api/update-schedule.js
```

ESPN's public JSON endpoints are used for schedule, scores, box scores and odds:
they answer from GitHub Actions runners, where `cdn.nba.com` returns 403 and
`stats.nba.com` times out. No API key needed.

## The model (src/engine)

- Published probability: the betting market's de-vigged consensus (ESPN odds).
- Fallback when no odds yet: a logistic model on Elo (margin-of-victory),
  shrunk season point differential, recent form, rest, back-to-backs.
- Training and serving share one feature code path; upcoming games are added
  to the history without a result, so features can't leak.
- `confidence` = probability of the picked side.

Walk-forward backtest (train on earlier seasons, predict the next), 2022-23 → 2025-26:

| | accuracy | Brier |
|---|---|---|
| Market consensus (published) | 68.6 % | 0.203 |
| Stats model (fallback) | 65.5 % | 0.214 |
| Previous engine, live 2025-26 | 61.6 % | 0.232 |

A blend gave the stats model ~0 weight on top of the market, and betting the
model's disagreements with the market lost ~7 % per bet: there is no edge to
sell, only honest probabilities. Research scripts: `research/`.

## Commands

```bash
python scripts/morning_routine.py --no-push      # full daily run locally (no git push)
python scripts/morning_routine.py --skip-email   # same without the e-mail
python scripts/dry_run_thread.py 2026-10-20 0    # build a real thread + charts, no posting
python scripts/train_v2.py                       # backtest + retrain models/v2_model.json
python -m pytest tests -q
```

Windows: `run_daily_prediction.bat` runs the routine without pushing.

## Configuration

GitHub secrets: `EMAIL_ADDRESS`, `EMAIL_APP_PASSWORD`, `TWITTER_API_KEY`,
`TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_SECRET`, `TWITTER_BEARER_TOKEN`.

Vercel environment: `PUBLISH_PASSWORD`, `GITHUB_TOKEN` (fine-grained PAT on this repo:
Contents, Actions and Workflows read/write), `GITHUB_REPO=AnthonyNadjari/NBAPredictLab`.

## Legacy

`app.py` (Streamlit) and `daily_auto_prediction.py` predictions use the previous
engine; the thread formatter in `daily_auto_prediction.py` is still used by the bot.
