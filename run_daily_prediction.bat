@echo off
REM ============================================================================
REM Daily NBA Prediction Automation Runner
REM ============================================================================
REM This script runs the complete morning routine:
REM   1. Refresh game data from NBA API (get recent scores)
REM   2. Update prediction results (verify yesterday's predictions)
REM   3. Fetch today's predictions
REM   4. Send email report
REM
REM Suitable for Windows Task Scheduler or manual execution
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

REM Run the morning routine script
echo ============================================================
echo NBA Predictor - Morning Routine
echo Started at %date% %time%
echo ============================================================
echo.
echo This script will:
echo   1. Refresh game data from NBA API (get yesterday's scores)
echo   2. Update prediction results (verify yesterday's predictions)
echo   3. Send email report
echo.
echo Note: Predictions should already be generated via Streamlit.
echo       Use --with-predictions flag to also generate predictions.
echo.

python scripts/morning_routine.py >> logs\scheduler.log 2>&1

REM Capture exit code
set EXIT_CODE=%ERRORLEVEL%

REM Log result
if %EXIT_CODE% EQU 0 (
    echo [%date% %time%] SUCCESS: Morning routine completed successfully >> logs\scheduler.log
    echo.
    echo ============================================================
    echo SUCCESS: Morning routine completed!
    echo ============================================================
) else (
    echo [%date% %time%] ERROR: Morning routine failed with exit code %EXIT_CODE% >> logs\scheduler.log
    echo.
    echo ============================================================
    echo WARNING: Morning routine completed with errors
    echo Check logs\scheduler.log for details
    echo ============================================================
)

REM Exit with the script's exit code
exit /b %EXIT_CODE%
