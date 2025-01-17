#!/bin/bash

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Check if the webserver.py exists
if [ ! -f "webserver.py" ]; then
    echo "Error: webserver.py not found"
    exit 1
fi

# Check if .env file exists, create if it doesn't
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating one..."
    echo "ANTHROPIC_API_KEY=" > .env
    echo "Please add your Anthropic API key to the .env file"
fi

# Create cache directory if it doesn't exist
mkdir -p .cache

# Create output directory if it doesn't exist
mkdir -p output

echo "Starting RSS Summarizer web interface..."
python webserver.py
