#!/bin/bash
#
# Setup Cron Job for CoinDesk Scraper
#
# This script helps you set up a cron job for daily news scraping.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRAPER_SCRIPT="$SCRIPT_DIR/run_daily_scraper.sh"

echo "=========================================="
echo "CoinDesk Scraper - Cron Job Setup"
echo "=========================================="
echo ""

# Check if script exists and is executable
if [ ! -f "$SCRAPER_SCRIPT" ]; then
    echo "Error: Scraper script not found at $SCRAPER_SCRIPT"
    exit 1
fi

if [ ! -x "$SCRAPER_SCRIPT" ]; then
    echo "Making scraper script executable..."
    chmod +x "$SCRAPER_SCRIPT"
fi

# Test run
echo "Testing scraper script..."
echo "(This will run once to make sure everything works)"
echo ""
"$SCRAPER_SCRIPT"

if [ $? -ne 0 ]; then
    echo ""
    echo "Warning: Test run failed!"
    echo "Please check the logs and fix any issues before setting up cron."
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Choose when to run the scraper:"
echo "1) Daily at 9:00 AM"
echo "2) Daily at 6:00 AM"
echo "3) Twice daily (9:00 AM and 6:00 PM)"
echo "4) Every 6 hours"
echo "5) Custom time"
echo "6) Cancel"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        CRON_SCHEDULE="0 9 * * *"
        DESCRIPTION="daily at 9:00 AM"
        ;;
    2)
        CRON_SCHEDULE="0 6 * * *"
        DESCRIPTION="daily at 6:00 AM"
        ;;
    3)
        CRON_SCHEDULE="0 9,18 * * *"
        DESCRIPTION="twice daily (9:00 AM and 6:00 PM)"
        ;;
    4)
        CRON_SCHEDULE="0 */6 * * *"
        DESCRIPTION="every 6 hours"
        ;;
    5)
        echo ""
        echo "Enter cron schedule (format: minute hour day month weekday)"
        echo "Example: 0 9 * * * for daily at 9 AM"
        read -p "Schedule: " CRON_SCHEDULE
        DESCRIPTION="custom schedule: $CRON_SCHEDULE"
        ;;
    6)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

# Create cron job entry
CRON_JOB="$CRON_SCHEDULE $SCRAPER_SCRIPT"

echo ""
echo "The following cron job will be added:"
echo "$CRON_JOB"
echo ""
echo "This will run the scraper $DESCRIPTION"
echo ""
read -p "Proceed? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null | grep -v "$SCRAPER_SCRIPT"; echo "$CRON_JOB") | crontab -

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Cron job added successfully!"
    echo ""
    echo "Current crontab:"
    crontab -l | grep "$SCRAPER_SCRIPT"
    echo ""
    echo "Logs will be saved to:"
    echo "  $(cd "$SCRIPT_DIR/../../.." && pwd)/logs/coindesk_scraper_YYYYMMDD.log"
    echo ""
    echo "To view logs:"
    echo "  cat $(cd "$SCRIPT_DIR/../../.." && pwd)/logs/coindesk_scraper_\$(date +%Y%m%d).log"
    echo ""
    echo "To remove the cron job, run:"
    echo "  $SCRIPT_DIR/remove_cron.sh"
else
    echo ""
    echo "✗ Failed to add cron job."
    echo "You may need to add it manually:"
    echo "  crontab -e"
    echo "Then add this line:"
    echo "  $CRON_JOB"
fi
