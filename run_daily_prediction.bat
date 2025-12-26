@echo off
REM ============================================================================
REM Daily NBA Prediction Automation Runner
REM ============================================================================
REM This script activates the virtual environment and runs the prediction script
REM Suitable for Windows Task Scheduler
REM ============================================================================

REM Set the project directory (UPDATE THIS PATH IF NEEDED!)
set PROJECT_DIR=C:\Users\nadja\OneDrive\Bureau\code\nba_predictor

REM Change to project directory
cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to change to project directory: %PROJECT_DIR%
    exit /b 1
)

REM Activate virtual environment (if using one)
REM Uncomment the following lines if you use a virtual environment:
REM if exist venv\Scripts\activate.bat (
REM     call venv\Scripts\activate.bat
REM ) else (
REM     echo WARNING: Virtual environment not found at venv\Scripts\activate.bat
REM )

REM Create logs directory if it doesn't exist
if not exist logs mkdir logs

REM Run the prediction script
echo Running NBA prediction automation at %date% %time%
python daily_auto_prediction.py >> logs\scheduler.log 2>&1

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

REM Log result
if %EXIT_CODE% EQU 0 (
    echo [%date% %time%] SUCCESS: Prediction script completed successfully >> logs\scheduler.log
) else (
    echo [%date% %time%] ERROR: Prediction script failed with exit code %EXIT_CODE% >> logs\scheduler.log
)

REM Exit with the script's exit code
exit /b %EXIT_CODE%
