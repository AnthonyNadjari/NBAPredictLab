# Daily NBA Prediction Automation - Scheduling Guide

This guide explains how to schedule the `daily_auto_prediction.py` script to run automatically every day on different platforms.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Windows Task Scheduler (Recommended for Windows)](#windows-task-scheduler)
3. [Linux/Mac Cron Jobs](#linuxmac-cron-jobs)
4. [Cloud Solutions](#cloud-solutions)
5. [Testing Your Setup](#testing-your-setup)
6. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Prerequisites

Before scheduling, ensure:

1. **Environment Variables Configured**: Your `.env` file contains valid Twitter API credentials:
   ```env
   TW_API_KEY=your_api_key_here
   TW_API_SECRET=your_api_secret_here
   TW_ACCESS_TOKEN=your_access_token_here
   TW_ACCESS_SECRET=your_access_secret_here
   TW_DRY_RUN=false
   ```

2. **Dependencies Installed**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model Trained**: Ensure models exist in the `models/` directory

4. **Script Tested**: Run manually first to verify it works:
   ```bash
   python daily_auto_prediction.py --dry-run --verbose
   ```

---

## Windows Task Scheduler

**Best for:** Windows users (most reliable native solution)

### Step-by-Step Setup

#### 1. Create a Batch Script Wrapper

Create `run_daily_prediction.bat` in your project directory:

```batch
@echo off
REM Daily NBA Prediction Automation Runner
REM This script activates the virtual environment and runs the prediction script

REM Set the project directory (UPDATE THIS PATH!)
set PROJECT_DIR=C:\Users\nadja\OneDrive\Bureau\code\nba_predictor

REM Change to project directory
cd /d "%PROJECT_DIR%"

REM Activate virtual environment (if using one)
REM Uncomment and update the path if you use a virtual environment:
REM call venv\Scripts\activate.bat

REM Run the prediction script
python daily_auto_prediction.py >> logs\scheduler.log 2>&1

REM Exit
exit /b %ERRORLEVEL%
```

**Important:** Update `PROJECT_DIR` to your actual project path.

#### 2. Open Task Scheduler

- Press `Win + R`, type `taskschd.msc`, and press Enter
- Or search for "Task Scheduler" in the Start menu

#### 3. Create a New Task

1. Click **"Create Task"** (not "Create Basic Task") in the right panel

2. **General Tab**:
   - Name: `NBA Daily Prediction`
   - Description: `Automated NBA prediction posting to Twitter`
   - Select **"Run whether user is logged on or not"**
   - Check **"Run with highest privileges"** (if needed for file access)
   - Configure for: **Windows 10** (or your version)

3. **Triggers Tab**:
   - Click **"New..."**
   - Begin the task: **On a schedule**
   - Settings: **Daily**
   - Start: Select your preferred time (e.g., `10:00 AM`)
   - Recur every: **1 day**
   - Advanced settings:
     - Check **"Enabled"**
   - Click **OK**

4. **Actions Tab**:
   - Click **"New..."**
   - Action: **Start a program**
   - Program/script: `C:\Windows\System32\cmd.exe`
   - Add arguments: `/c "C:\Users\nadja\OneDrive\Bureau\code\nba_predictor\run_daily_prediction.bat"`
   - Start in: `C:\Users\nadja\OneDrive\Bureau\code\nba_predictor`
   - Click **OK**

5. **Conditions Tab**:
   - Uncheck **"Start the task only if the computer is on AC power"** (if running on laptop)
   - Check **"Wake the computer to run this task"** (optional)

6. **Settings Tab**:
   - Check **"Allow task to be run on demand"**
   - Check **"Run task as soon as possible after a scheduled start is missed"**
   - If the task fails, restart every: **10 minutes** (optional)
   - Attempt to restart up to: **3 times** (optional)
   - Stop the task if it runs longer than: **1 hour**

7. Click **OK** and enter your Windows password if prompted

#### 4. Test the Task

- Right-click the task in Task Scheduler
- Click **"Run"**
- Check the "Last Run Result" column (should be `0x0` for success)
- Verify logs in `logs/` directory

---

## Linux/Mac Cron Jobs

**Best for:** Linux/Mac users or headless servers

### Step-by-Step Setup

#### 1. Create a Shell Script Wrapper

Create `run_daily_prediction.sh` in your project directory:

```bash
#!/bin/bash
# Daily NBA Prediction Automation Runner

# Set project directory (UPDATE THIS PATH!)
PROJECT_DIR="/home/yourusername/nba_predictor"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment (if using one)
# Uncomment and update the path if you use a virtual environment:
# source venv/bin/activate

# Run the prediction script
/usr/bin/python3 daily_auto_prediction.py >> logs/scheduler.log 2>&1

# Exit with script's exit code
exit $?
```

**Important:**
- Update `PROJECT_DIR` to your actual project path
- Update `/usr/bin/python3` to your Python path (find with `which python3`)

#### 2. Make Script Executable

```bash
chmod +x run_daily_prediction.sh
```

#### 3. Configure Cron

Open crontab editor:

```bash
crontab -e
```

Add the following line (runs daily at 10:00 AM):

```cron
# NBA Daily Prediction - Runs at 10:00 AM every day
0 10 * * * /home/yourusername/nba_predictor/run_daily_prediction.sh
```

**Cron Time Format:**
```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday = 0)
│ │ │ │ │
* * * * * command to execute
```

**Examples:**
- `0 10 * * *` - 10:00 AM daily
- `30 14 * * *` - 2:30 PM daily
- `0 9 * * 1-5` - 9:00 AM weekdays only
- `0 */6 * * *` - Every 6 hours

#### 4. Verify Cron Configuration

List your cron jobs:

```bash
crontab -l
```

#### 5. Test the Script

Run manually first:

```bash
./run_daily_prediction.sh
```

Check logs:

```bash
tail -f logs/scheduler.log
```

---

## Cloud Solutions

For reliable 24/7 operation without keeping your computer running.

### Option 1: GitHub Actions (Free, Recommended for Beginners)

**Pros:** Free for public repos, easy setup, reliable
**Cons:** Public repo required (or paid for private), limited to 2000 minutes/month free

Create `.github/workflows/daily-prediction.yml`:

```yaml
name: Daily NBA Prediction

on:
  schedule:
    # Runs at 10:00 AM UTC daily (adjust timezone as needed)
    - cron: '0 10 * * *'
  workflow_dispatch:  # Allows manual triggering

jobs:
  predict-and-post:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run prediction script
      env:
        TW_API_KEY: ${{ secrets.TW_API_KEY }}
        TW_API_SECRET: ${{ secrets.TW_API_SECRET }}
        TW_ACCESS_TOKEN: ${{ secrets.TW_ACCESS_TOKEN }}
        TW_ACCESS_SECRET: ${{ secrets.TW_ACCESS_SECRET }}
        TW_DRY_RUN: false
      run: |
        python daily_auto_prediction.py

    - name: Upload logs
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: prediction-logs
        path: logs/
```

**Setup:**
1. Create the workflow file in your repo
2. Go to repo Settings → Secrets → Actions
3. Add your Twitter credentials as secrets
4. Enable Actions in your repo settings

### Option 2: AWS Lambda (Serverless)

**Pros:** Highly scalable, cheap, reliable
**Cons:** Requires AWS account, more complex setup

**Estimated Cost:** ~$0.20/month (Free tier covers this)

**Setup Overview:**
1. Package your code + dependencies as a Lambda deployment package
2. Create Lambda function with Python 3.9+ runtime
3. Add environment variables for Twitter credentials
4. Create EventBridge rule to trigger daily
5. Configure 5-minute timeout (predictions can be slow)

**Deployment Script:** (create `lambda_package.sh`)

```bash
#!/bin/bash
# Package for AWS Lambda deployment

# Create deployment package
mkdir -p lambda_package
pip install -r requirements.txt -t lambda_package/
cp -r src/ lambda_package/
cp -r models/ lambda_package/
cp -r data/ lambda_package/
cp daily_auto_prediction.py lambda_package/

# Create lambda handler
cat > lambda_package/lambda_handler.py << 'EOF'
import sys
from daily_auto_prediction import DailyPredictionAutomation

def lambda_handler(event, context):
    automation = DailyPredictionAutomation(
        db_path='data/nba_predictor.db',
        model_dir='models',
        log_dir='/tmp/logs'  # Lambda writable directory
    )
    success = automation.run()
    return {'statusCode': 200 if success else 500}
EOF

# Zip package
cd lambda_package
zip -r ../lambda_function.zip .
cd ..

echo "Lambda package created: lambda_function.zip"
```

### Option 3: Heroku (Platform as a Service)

**Pros:** Easy deployment, free tier available
**Cons:** Free tier sleeps after 30 mins inactivity

**Setup:**
1. Create `Procfile`:
   ```
   worker: python daily_auto_prediction.py
   ```

2. Create `runtime.txt`:
   ```
   python-3.9.16
   ```

3. Use Heroku Scheduler add-on (free):
   ```bash
   heroku addons:create scheduler:standard
   heroku addons:open scheduler
   ```

4. Add daily job: `python daily_auto_prediction.py`

### Option 4: Raspberry Pi / Home Server

**Pros:** Full control, no cloud costs, runs 24/7
**Cons:** Requires hardware, electricity costs, maintenance

- Use Linux cron setup (see above)
- Keep Pi connected to power and internet
- Ideal for always-on home automation

---

## Recommended Timing

Consider NBA schedule when choosing run time:

| Time (EST) | Why |
|------------|-----|
| **10:00 AM** | ✓ Morning predictions before games start (most games are evening) |
| **12:00 PM** | ✓ Lunch time, good engagement |
| **2:00 PM** | ✓ Afternoon, games usually start 7-8 PM EST |
| **6:00 AM** | ⚠ Very early, low engagement |
| **8:00 PM** | ✗ Too late, games already started |

**Recommendation:** Run at **10:00 AM EST** (15:00 UTC) for best results.

---

## Testing Your Setup

### 1. Dry Run Test

Test without posting to Twitter:

```bash
python daily_auto_prediction.py --dry-run --verbose
```

Expected output:
```
2025-12-21 10:00:00 - DailyPredictionBot - INFO - Initializing...
2025-12-21 10:00:05 - DailyPredictionBot - INFO - ✓ Predictor model loaded
2025-12-21 10:00:06 - DailyPredictionBot - INFO - Found 5 game(s)
...
2025-12-21 10:02:30 - DailyPredictionBot - INFO - ✓ WORKFLOW COMPLETED SUCCESSFULLY
```

### 2. Specific Date Test

Test with a known date (e.g., Christmas Day):

```bash
python daily_auto_prediction.py --dry-run --date 2025-12-25
```

### 3. Live Test (Careful!)

Test actual posting (this WILL post to Twitter):

```bash
python daily_auto_prediction.py --verbose
```

### 4. Scheduler Test

**Windows:** Right-click task → "Run"
**Linux/Mac:**
```bash
./run_daily_prediction.sh
```

---

## Monitoring & Troubleshooting

### Check Logs

All logs are saved to `logs/` directory:

- **`daily_predictions_YYYYMM.log`** - Full debug logs (rotates monthly)
- **`posted_predictions.jsonl`** - JSON log of all posted predictions
- **`scheduler.log`** - Scheduler execution logs

**View latest logs:**

```bash
# Windows (PowerShell)
Get-Content logs\daily_predictions_202512.log -Tail 50

# Linux/Mac
tail -f logs/daily_predictions_202512.log
```

### Common Issues

#### 1. Twitter API Authentication Failed

**Symptoms:** `✗ Twitter authentication failed`

**Solutions:**
- Verify `.env` file exists with correct credentials
- Check Twitter API keys are active in developer portal
- Ensure you have "Read and Write" permissions
- Try regenerating access tokens

#### 2. No Games Found

**Symptoms:** `No NBA games scheduled for YYYY-MM-DD`

**Solutions:**
- Check the date (NBA season: October - June)
- Verify internet connection (script needs to fetch schedule)
- Check NBA API status

#### 3. Model Loading Failed

**Symptoms:** `✗ Predictor model loaded failed`

**Solutions:**
- Verify `models/` directory exists with trained models
- Check database file exists at `data/nba_predictor.db`
- Retrain model if needed: `python scripts/train_model.py`

#### 4. Script Runs But Doesn't Post

**Symptoms:** Script completes but no Twitter post

**Solutions:**
- Check `TW_DRY_RUN` is set to `false` in `.env`
- Verify odds threshold (only posts if odds > 1.3)
- Check confidence - might not meet criteria
- Review logs for "No predictions meet criteria"

#### 5. Permission Denied (Linux/Mac)

**Symptoms:** `Permission denied` when running script

**Solutions:**
```bash
chmod +x run_daily_prediction.sh
chmod +x daily_auto_prediction.py
```

### Email Notifications (Optional)

Add email alerts for failures by modifying the wrapper script:

**Linux/Mac** (`run_daily_prediction.sh`):
```bash
#!/bin/bash
PROJECT_DIR="/home/yourusername/nba_predictor"
cd "$PROJECT_DIR" || exit 1

python3 daily_auto_prediction.py >> logs/scheduler.log 2>&1

# Send email if failed
if [ $? -ne 0 ]; then
    echo "NBA prediction script failed. Check logs." | mail -s "NBA Bot Failed" your@email.com
fi
```

**Windows** (using PowerShell and SMTP):
```powershell
# Add to run_daily_prediction.bat
if errorlevel 1 (
    powershell -Command "Send-MailMessage -From 'bot@example.com' -To 'you@example.com' -Subject 'NBA Bot Failed' -Body 'Check logs' -SmtpServer 'smtp.gmail.com'"
)
```

---

## Summary

### Quick Recommendations by Platform

| Platform | Recommended Method | Difficulty | Cost |
|----------|-------------------|------------|------|
| **Windows** | Task Scheduler | Easy | Free |
| **Linux/Mac** | Cron | Easy | Free |
| **Cloud (Free)** | GitHub Actions | Medium | Free |
| **Cloud (Reliable)** | AWS Lambda | Hard | ~$0.20/mo |
| **Home Server** | Cron on Raspberry Pi | Medium | One-time hardware |

### Best Overall Setup

For most users:
1. **Development/Testing:** Run manually with `--dry-run`
2. **Local Automation:** Windows Task Scheduler or Linux Cron
3. **24/7 Reliability:** GitHub Actions (if repo is public) or AWS Lambda

---

## Next Steps

1. ✅ Test script with `--dry-run` flag
2. ✅ Configure environment variables (`.env`)
3. ✅ Choose scheduling method
4. ✅ Set up scheduled task
5. ✅ Test scheduled task runs correctly
6. ✅ Monitor for first few days
7. ✅ Review posted predictions in logs

---

## Support

If you encounter issues:

1. Check logs in `logs/` directory
2. Run with `--verbose` flag for detailed output
3. Test with `--dry-run` to isolate issues
4. Verify all prerequisites are met
5. Check NBA game schedule (no games = nothing to post)

Good luck with your automated NBA predictions! 🏀
