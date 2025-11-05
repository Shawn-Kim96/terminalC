#!/bin/bash
#
# Daily CoinDesk News Scraper
#
# This script runs the CoinDesk scraper daily to accumulate news data over time.
# Add this to crontab to run automatically.
#
# Example crontab entry (runs daily at 9 AM):
# 0 9 * * * /path/to/terminalC/src/data_scripts/web/run_daily_scraper.sh
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Change to script directory
cd "$SCRIPT_DIR"

# Set up log file with date
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/coindesk_scraper_$(date +%Y%m%d).log"

echo "========================================" >> "$LOG_FILE"
echo "CoinDesk Scraper Run: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Activate virtual environment if it exists
if [ -d "$PROJECT_DIR/.venv" ]; then
    echo "Activating virtual environment..." >> "$LOG_FILE"
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Run the scraper
echo "Running scraper..." >> "$LOG_FILE"
python coindesk_scraper.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Scraper completed successfully" >> "$LOG_FILE"
else
    echo "✗ Scraper failed with exit code: $EXIT_CODE" >> "$LOG_FILE"
fi

echo "Finished at: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Optional: Ingest to DuckDB after scraping
# Uncomment the lines below if you want to automatically ingest to database
# echo "Ingesting to DuckDB..." >> "$LOG_FILE"
# python "$PROJECT_DIR/src/database/ingest_to_duckdb.py" >> "$LOG_FILE" 2>&1

exit $EXIT_CODE
