#!/bin/bash

echo "Financial RAG System Setup"
echo "========================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python is installed."
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set environment variables
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
export PYTHONHTTPSVERIFY=0

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir data
    echo
    echo "IMPORTANT: Please add your PDF files to the 'data' directory"
    echo
fi

# Make shell scripts executable
chmod +x run_app.sh
chmod +x setup.sh

echo
echo "Setup complete!"
echo
echo "To run the application:"
echo "  1. Run: ./run_app.sh"
echo "  2. Or run: streamlit run app.py"
echo
echo "Don't forget to add your PDF files to the 'data' directory!"
echo 