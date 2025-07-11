@echo off
echo Financial RAG System - Model Downloader
echo =====================================
echo.

REM Set environment variables
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set CURL_CA_BUNDLE=
set REQUESTS_CA_BUNDLE=
set PYTHONHTTPSVERIFY=0

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the model downloader
echo.
echo This will download AI models for offline use.
echo This may take 10-30 minutes depending on your internet speed.
echo.
echo Press Ctrl+C to cancel, or
pause

python download_models.py

echo.
echo Model download process completed.
pause 