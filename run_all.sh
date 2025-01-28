#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change to script's directory
cd "$(dirname "$0")"

# Load environment variables securely
if [ -f .env ]; then
    # Ensure .env file has restricted permissions
    chmod 600 .env
    
    # Source .env, but only export explicitly needed variables
    export ANTHROPIC_API_KEY=$(grep '^ANTHROPIC_API_KEY=' .env | cut -d '=' -f2-)
    export RSS_FEEDS_URL=$(grep '^RSS_FEEDS_URL=' .env | cut -d '=' -f2-)
fi

# Ensure required environment variables are set
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "Error: ANTHROPIC_API_KEY is not set. Please create a .env file from .env.template."
    exit 1
fi

# Create necessary directories
mkdir -p logs output .cache

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to run RSS reader
run_rss_reader() {
    echo "Starting RSS reader..."
    {
        echo "=== RSS Reader Run: $TIMESTAMP ==="
        python rss_reader.py
    } 2>&1 | tee "logs/rss_reader_${TIMESTAMP}.log"

    # Clean up old logs and HTML files (older than 7 days)
    find logs -type f -name "*.log" -mtime +7 -delete
    find . -type f -name "rss_summaries_*.html" -mtime +7 -delete
}

# Function to run web server
run_web_server() {
    echo "Starting web server..."
    python webserver.py
}

# Run RSS reader in the background
run_rss_reader &
RSS_PID=$!

# Run web server in the background
run_web_server &
WEB_PID=$!

# Function to handle script termination
cleanup() {
    echo "Shutting down services..."
    kill $RSS_PID 2>/dev/null || true
    kill $WEB_PID 2>/dev/null || true
    exit 0
}

# Set up trap for cleanup on script termination
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait $RSS_PID $WEB_PID
