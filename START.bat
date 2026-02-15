@echo off
REM 🚀 EV Market Forecasting Dashboard - Windows Startup Script
REM Compatible with Windows CMD

echo.
echo ================================================================================
echo   ^!BOLD! 🚀 EV MARKET FORECASTING DASHBOARD - STARTUP
echo      Kaggle Grandmaster Edition
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo    Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

echo ✓ Python detected
python --version

echo.
echo ================================================================================
echo   STEP 1: Installing Dependencies
echo ================================================================================
echo.

echo 📥 Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ❌ Failed to upgrade pip
    pause
    exit /b 1
)

echo.
echo 📥 Installing packages from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    echo    Try: python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!

echo.
echo ================================================================================
echo   STEP 2: Verifying Packages
echo ================================================================================
echo.

python -c "import streamlit; print('✓ Streamlit')" 2>nul || echo ❌ Streamlit not found
python -c "import prophet; print('✓ Prophet')" 2>nul || echo ⚠️  Prophet not found
python -c "import tensorflow; print('✓ TensorFlow')" 2>nul || echo ⚠️  TensorFlow not found
python -c "import plotly; print('✓ Plotly')" 2>nul || echo ⚠️  Plotly not found

echo.
echo ================================================================================
echo   STEP 3: Launching Dashboard
echo ================================================================================
echo.

echo 🚀 Starting EV Market Forecasting Dashboard...
echo.
echo 📊 Dashboard URL: http://localhost:8501
echo.
echo 💡 Tips:
echo    • First load trains all models (2-3 minutes) - be patient!
echo    • Use the sidebar to navigate different sections
echo    • Hover on charts for detailed data points
echo    • Press Ctrl+C to stop the application
echo.
echo ================================================================================
echo.

python -m streamlit run ev_market_app.py

if errorlevel 1 (
    echo.
    echo ❌ Application launch failed
    pause
    exit /b 1
)

pause
