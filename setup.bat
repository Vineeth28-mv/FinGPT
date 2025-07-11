@echo off
echo Financial RAG System Setup
echo ========================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python is installed.
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Set environment variables
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set CURL_CA_BUNDLE=
set REQUESTS_CA_BUNDLE=
set PYTHONHTTPSVERIFY=0

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create data directory if it doesn't exist
if not exist data (
    echo Creating data directory...
    mkdir data
    echo.
    echo IMPORTANT: Please add your PDF files to the 'data' directory
    echo.
)

echo.
echo Setup complete!
echo.
echo To run the application:
echo   1. Double-click run_app.bat
echo   2. Or run: streamlit run app.py
echo.
echo Don't forget to add your PDF files to the 'data' directory!
echo.

pause 