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

# Create directories if they don't exist
mkdir -p logs output

# Generate timestamp for log and output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run RSS reader with logging
{
    echo "=== RSS Reader Run: $TIMESTAMP ==="
    python rss_reader.py
} 2>&1 | tee "logs/rss_reader_${TIMESTAMP}.log"

# Clean up old logs and HTML files (older than 7 days)
find logs -type f -name "*.log" -mtime +7 -delete
find . -type f -name "rss_summaries_*.html" -mtime +7 -delete

echo "RSS reader completed. Check logs and output for details."
