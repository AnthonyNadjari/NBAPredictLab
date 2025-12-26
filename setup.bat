@echo off
echo ========================================
echo NBA Predictor - Dependency Installation
echo ========================================
echo.

echo Installing required packages...
echo This may take a few minutes...
echo.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.

echo Verifying installation...
python -c "import xgboost; import lightgbm; import sklearn; import pandas; import numpy; import streamlit; print('All packages installed successfully!')"

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may not have installed correctly.
    echo Please check the error messages above.
    pause
    exit /b 1
) else (
    echo.
    echo All dependencies are installed!
    echo You can now run the application with: run.bat
    pause
)







