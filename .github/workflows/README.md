# GitHub Actions Automation Guide

This directory contains GitHub Actions workflows for automated tasks.

## 📁 Files

- **coindesk_scraper.yml** - Automated CoinDesk news scraping workflow

## 🚀 Setup Instructions

### 1. Push Code to GitHub

Push the workflow files to your GitHub repository:

```bash
cd /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC

# Initialize Git (if not already done)
git init
git add .
git commit -m "feat: add GitHub Actions workflow for news scraper"

# Push to GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/terminalC.git
git branch -M main
git push -u origin main
```

### 2. Enable GitHub Actions

1. Go to your GitHub repository page
2. Navigate to **Settings** → **Actions** → **General**
3. In the **Actions permissions** section:
   - ✅ Select "Allow all actions and reusable workflows"
4. In the **Workflow permissions** section:
   - ✅ Select "Read and write permissions"
   - ✅ Check "Allow GitHub Actions to create and approve pull requests"

### 3. Verify Automatic Execution

The workflow will run:
- **Automatically**: Daily at UTC 0:00 (9:00 AM KST / 4:00 PM PST)
- **Manually**: You can trigger it from GitHub web interface

#### Manual Trigger
1. Go to GitHub repository → **Actions** tab
2. Select **CoinDesk News Scraper** from the left sidebar
3. Click **Run workflow** button in the top right
4. Confirm by clicking **Run workflow**

### 4. Check Execution Logs

1. Go to GitHub repository → **Actions** tab
2. Click on a workflow run to see details
3. View logs for each step

## ⚙️ Configuration

### Change Execution Time

Edit the `cron` section in `.github/workflows/coindesk_scraper.yml`:

```yaml
schedule:
  # Current: Daily at UTC 0:00 (9 AM KST)
  - cron: '0 0 * * *'

  # Other examples:
  # Daily at UTC 6:00 (3 PM KST)
  # - cron: '0 6 * * *'

  # Daily at UTC 12:00 (9 PM KST)
  # - cron: '0 12 * * *'

  # Twice daily (UTC 0:00, 12:00)
  # - cron: '0 0,12 * * *'

  # Every 6 hours
  # - cron: '0 */6 * * *'
```

### Change Execution Frequency

To run more frequently, add multiple cron schedules:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
  - cron: '0 */4 * * *'  # Every 4 hours
  - cron: '0 */2 * * *'  # Every 2 hours
```

⚠️ **Note**: GitHub Actions free tier has usage limits (2,000 minutes/month).

## 📊 Data Storage Methods

The workflow saves data in two ways:

### Method 1: Git Commit (Automatic)
- Automatically commits scraped data to the repository
- Stored in `data/raw_data/news/` directory
- History is tracked
- **Downside**: Increases repository size

### Method 2: Artifacts (Backup)
- Stored as GitHub Actions artifacts for 30 days
- Can be downloaded from Actions tab
- Doesn't affect repository size

## 🔍 Troubleshooting

### Q: Workflow isn't running
**A**:
1. Verify GitHub Actions is enabled
2. Check workflow file is in `.github/workflows/` directory
3. Confirm file extension is `.yml` or `.yaml`
4. Ensure it's pushed to main/master branch

### Q: Commit permission error
**A**:
1. Go to Settings → Actions → General
2. Select "Read and write permissions" under "Workflow permissions"

### Q: Data isn't being committed
**A**:
1. Check if `data/` is in `.gitignore`
2. If yes, make sure these lines are in `.gitignore`:
   ```
   # Allow news data
   !data/raw_data/
   !data/raw_data/news/
   ```

### Q: Exceeded free usage quota
**A**:
- Reduce execution frequency (1-2 times per day recommended)
- Or upgrade to GitHub Pro (3,000 minutes/month)

## 📈 Expected Resource Usage

- **Execution time**: ~1-2 minutes per run
- **Daily once**: ~30-60 minutes/month
- **Daily twice**: ~60-120 minutes/month
- **GitHub free quota**: 2,000 minutes/month

## 🎯 Benefits

✅ **Fully Automated**: Runs even when your computer is off
✅ **Free**: GitHub free tier is sufficient
✅ **Reliable**: Runs 24/7 on GitHub servers
✅ **Logged**: All execution history is available
✅ **Version Control**: Data history tracked with Git
✅ **Backup**: Automatic backup with Artifacts

## 📅 Data Accumulation Timeline

Running daily:
- **1 day**: ~25-50 articles
- **1 week**: ~175-350 articles
- **1 month**: ~750-1,500 articles
- **1 year**: ~9,000-18,000 articles

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Cron Expression Generator](https://crontab.guru/)
- [Check GitHub Actions Usage](https://github.com/settings/billing)

## 🔄 Alternative: Combine with Local Execution

You can use both GitHub Actions and local cron:
- **GitHub Actions**: Primary execution (daily)
- **Local Cron**: Additional runs (when computer is on)

Both methods automatically deduplicate data, so it's safe to use together.

## 🎬 Quick Start

1. Push code to GitHub
2. Enable Actions with write permissions
3. Wait for next scheduled run (UTC 0:00) or trigger manually
4. Check Actions tab for logs
5. Data will be automatically committed to `data/raw_data/news/`

That's it! Your scraper will run automatically every day. 🎉
