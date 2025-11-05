#!/bin/bash
#
# Remove Cron Job for CoinDesk Scraper
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRAPER_SCRIPT="$SCRIPT_DIR/run_daily_scraper.sh"

echo "=========================================="
echo "CoinDesk Scraper - Remove Cron Job"
echo "=========================================="
echo ""

# Check if cron job exists
if ! crontab -l 2>/dev/null | grep -q "$SCRAPER_SCRIPT"; then
    echo "No cron job found for the scraper."
    exit 0
fi

echo "Current cron job(s):"
crontab -l | grep "$SCRAPER_SCRIPT"
echo ""

read -p "Remove this cron job? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Remove from crontab
crontab -l | grep -v "$SCRAPER_SCRIPT" | crontab -

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Cron job removed successfully!"
else
    echo ""
    echo "✗ Failed to remove cron job."
    echo "You may need to remove it manually:"
    echo "  crontab -e"
fi
