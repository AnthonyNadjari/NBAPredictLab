@echo off
cd /d "%~dp0"

echo ========================================
echo NBA Predictor - Starting Application
echo ========================================
echo.

echo Checking Python packages...
python -c "import streamlit, tweepy, plotly, kaleido; print('All packages OK')" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: Some packages may be missing!
    echo Installing/upgrading required packages...
    echo.
    pip install --upgrade tweepy plotly kaleido pillow python-dotenv tenacity
    echo.
)

echo.
echo Starting Streamlit application...
echo.
echo The app will open in your default browser.
echo Press Ctrl+C in this window to stop the server.
echo.

streamlit run app.py

pause
