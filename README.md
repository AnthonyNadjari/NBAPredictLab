# NBA Match Outcome Predictor & Auto-Poster

A production-grade NBA game prediction tool with **automatic Twitter posting** and **prediction tracking**.

## Features

✅ **Predictions** - ML-powered predictions (68-72% accuracy)
✅ **Auto-Post to Twitter** - Daily thread with charts
✅ **Result Tracking** - Automatically updates predictions with actual results
✅ **Performance Analytics** - Track accuracy over time

## Quick Start

### 1. Manual App (Streamlit)
```bash
streamlit run app.py
```

### 2. Daily Automation (Updates + Twitter)
```bash
run_daily_prediction.bat
```

This will:
- Update previous pending predictions with actual results
- Generate today's predictions
- Post best prediction to Twitter as 8-tweet thread

### 3. View Predictions
Run the Streamlit app and go to **Performance** tab

## Daily Automation Details

**What it does:**
1. Fetches game results from NBA API (last 7 days)
2. Updates pending predictions with actual results (correct/wrong)
3. Fetches today's NBA games
4. Generates predictions for all games
5. Selects best prediction (confidence + odds > 1.3)
6. Posts to Twitter with 7 chart images

**Options:**
```bash
python daily_auto_prediction.py --lookback-days 14            # Update last 14 days
python daily_auto_prediction.py --dry-run                     # Test mode (no Twitter)
python daily_auto_prediction.py --skip-prediction-check       # Skip updating old predictions
python daily_auto_prediction.py --verbose                     # Show debug logs
```

## Troubleshooting

### Twitter Rate Limit (429 Error)

**Problem:** Nothing posted to Twitter

**Cause:** Hit Twitter's rate limit (~50 tweets/24h, threads count as 8 tweets)

**Solution:** Wait 15-30 minutes and run again

**Check status:** Look at `logs/daily_predictions_*.log` for the exact reset time in the error message

### Database Issues

```bash
python scripts/init_database.py
```

## Files

**Main Scripts:**
- `app.py` - Streamlit app (predictions, portfolio, analytics)
- `daily_auto_prediction.py` - Automation script
- `run_daily_prediction.bat` - Daily automation runner

**Batch Files:**
- `run_daily_prediction.bat` - Run daily automation
- `quick_start.bat` - Run Streamlit app
- `setup.bat` - Initial setup

**Logs:**
- `logs/daily_predictions_*.log` - Detailed workflow logs
- `logs/scheduler.log` - Batch file output

**Database:**
- `data/nba_predictor.db` - Contains predictions, bets, game data

## Scheduling (Windows Task Scheduler)

Use Windows Task Scheduler to set up automatic daily runs of `run_daily_prediction.bat`.

