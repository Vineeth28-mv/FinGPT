#!/bin/bash

echo "Starting Financial RAG System..."
echo

# Set environment variables to handle SSL and model download issues
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export PYTHONHTTPSVERIFY=0

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --upgrade

# Run the application
echo
echo "Starting Streamlit application..."
echo "Open your browser to http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application"
echo

streamlit run app.py 