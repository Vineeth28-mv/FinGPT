@echo off
echo Starting Financial RAG System...
echo.

REM Set environment variables to handle SSL and model download issues
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set CURL_CA_BUNDLE=
set REQUESTS_CA_BUNDLE=
set PYTHONHTTPSVERIFY=0
set HF_HUB_OFFLINE=0
set TRANSFORMERS_OFFLINE=0

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt --upgrade

REM Run the application
echo.
echo Starting Streamlit application...
echo Open your browser to http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py

pause 