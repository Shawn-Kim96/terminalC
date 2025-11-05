# CoinDesk News Scraper

This directory contains tools for scraping and analyzing CoinDesk cryptocurrency news articles using the CoinDesk API.

## Files

- **coindesk_scraper.py** - Main scraper that fetches news articles and saves them as CSV/pickle files
- **query_coindesk_db.py** - Utility for querying and analyzing the DuckDB database
- **coindesk_api_desc.md** - API documentation reference
- **requirements.txt** - Python dependencies

## Prerequisites

Install required dependencies:

```bash
cd src/data_scripts/web
pip install -r requirements.txt
```

Or install manually:
```bash
pip install requests pandas duckdb
```

## Workflow

### Step 1: Scrape News Articles

**Important Note:** This scraper uses CoinDesk's RSS feed, which only contains the most recent ~20-50 articles. To build historical data over 1 year, you need to run this script regularly (daily recommended).

Fetch recent CoinDesk news from RSS feed:

```bash
python coindesk_scraper.py
```

This will save files to `data/raw_data/news/`:
- `coindesk_articles_YYYYMMDD_HHMMSS.csv` - Timestamped articles data
- `coindesk_articles_YYYYMMDD_HHMMSS.pkl` - Timestamped pickle format
- `coindesk_articles_latest.csv` - Latest articles (always updated)
- `coindesk_articles_latest.pkl` - Latest articles in pickle format

### Step 1.5: Set Up Daily Automated Scraping (Recommended)

To accumulate 1 year of historical data, set up a cron job to run daily:

**Easy Setup (Interactive):**
```bash
./setup_cron.sh
```

**Manual Setup:**
See [CRON_SETUP.md](CRON_SETUP.md) for detailed instructions.

**Files:**
- `run_daily_scraper.sh` - Daily scraper script
- `setup_cron.sh` - Interactive cron job setup
- `remove_cron.sh` - Remove cron job
- `CRON_SETUP.md` - Detailed setup guide

**Options:**

- `--days N` - Only include articles from last N days (default: all from RSS feed)
- `--output PATH` - Custom output directory
- `--no-merge` - Don't merge with existing data (default: merge to avoid duplicates)
- `--verbose` or `-v` - Enable debug logging

**Examples:**

```bash
# Fetch all articles from RSS feed (~20-50 articles)
python coindesk_scraper.py

# Filter to only articles from last 7 days
python coindesk_scraper.py --days 7

# Don't merge with existing data
python coindesk_scraper.py --no-merge

# Custom output location with verbose logging
python coindesk_scraper.py --output /path/to/output --verbose
```

### Step 2: Ingest into DuckDB

Load the CSV/pickle data into a DuckDB database:

```bash
python ../../database/ingest_to_duckdb.py
```

Or from project root:
```bash
python src/database/ingest_to_duckdb.py
```

**Options:**

- `--input PATH` - Input directory with CSV/pickle files
- `--db PATH` - Database path (default: `data/raw_data/news/coindesk_news.duckdb`)
- `--mode MODE` - 'replace' (default) or 'append'
- `--verbose` or `-v` - Enable debug logging

**Examples:**

```bash
# Append new data to existing database
python ../../database/ingest_to_duckdb.py --mode append

# Use custom database location
python ../../database/ingest_to_duckdb.py --db /path/to/custom.duckdb
```

### Step 3: Query the Database

View database statistics:

```bash
python query_coindesk_db.py --stats
```

Search for articles:

```bash
python query_coindesk_db.py --search "bitcoin" --limit 20
python query_coindesk_db.py --search "ethereum regulation"
```

Get recent articles:

```bash
python query_coindesk_db.py --recent --days 7 --limit 15
```

Export to CSV:

```bash
python query_coindesk_db.py --export coindesk_news.csv
```

Run custom SQL query:

```bash
python query_coindesk_db.py --query "SELECT COUNT(*) FROM articles WHERE sentiment = 'POSITIVE'"
```

## Data Flow

```
1. CoinDesk API
   ↓
2. coindesk_scraper.py → CSV/Pickle files (data/raw_data/news/)
   ↓
3. ingest_to_duckdb.py → DuckDB database (data/raw_data/news/coindesk_news.duckdb)
   ↓
4. query_coindesk_db.py → Analysis & Queries
```

## Database Schema

### Articles Table

| Column | Type | Description |
|--------|------|-------------|
| article_id | VARCHAR | Unique article identifier (PRIMARY KEY) |
| guid | VARCHAR | Article GUID (UNIQUE) |
| source | VARCHAR | News source name |
| title | VARCHAR | Article title |
| body | TEXT | Full article content |
| url | VARCHAR | Article URL |
| published_at | TIMESTAMP | Publication date/time |
| created_at | TIMESTAMP | Creation date/time |
| updated_at | TIMESTAMP | Last update date/time |
| sentiment | VARCHAR | Article sentiment (POSITIVE/NEGATIVE/NEUTRAL) |
| categories | VARCHAR | Article categories (comma-separated) |
| tags | VARCHAR | Article tags (comma-separated) |
| author | VARCHAR | Article author |
| image_url | VARCHAR | Featured image URL |
| fetched_at | TIMESTAMP | When the article was scraped |

### Sources Table

| Column | Type | Description |
|--------|------|-------------|
| source_id | VARCHAR | Source identifier (PRIMARY KEY) |
| source_name | VARCHAR | Source display name |
| source_url | VARCHAR | Source website URL |
| fetched_at | TIMESTAMP | When the source was fetched |

### Categories Table

| Column | Type | Description |
|--------|------|-------------|
| category_id | VARCHAR | Category identifier (PRIMARY KEY) |
| category_name | VARCHAR | Category display name |
| fetched_at | TIMESTAMP | When the category was fetched |

## Example Queries

### Count articles by sentiment
```sql
SELECT sentiment, COUNT(*) as count
FROM articles
GROUP BY sentiment
ORDER BY count DESC;
```

### Get articles from last month
```sql
SELECT title, published_at, sentiment
FROM articles
WHERE published_at >= CURRENT_DATE - INTERVAL '1 month'
ORDER BY published_at DESC;
```

### Find most active authors
```sql
SELECT author, COUNT(*) as article_count
FROM articles
WHERE author IS NOT NULL
GROUP BY author
ORDER BY article_count DESC
LIMIT 10;
```

### Articles mentioning specific cryptocurrencies
```sql
SELECT title, published_at, url
FROM articles
WHERE body ILIKE '%ethereum%' OR title ILIKE '%ethereum%'
ORDER BY published_at DESC
LIMIT 20;
```

## API Information

The scraper uses the CoinDesk News API v1:

- **Base URL:** https://api.coindesk.com/news/v1
- **Endpoints:**
  - `/article/list` - List recent articles
  - `/source/list` - List news sources
  - `/category/list` - List news categories

See [coindesk_api_desc.md](coindesk_api_desc.md) for full API documentation.

## Notes

### RSS Feed Limitations
- **RSS feeds only contain ~20-50 recent articles** (not 1 year of data)
- To accumulate historical data, **run the scraper daily** (use cron job)
- Articles are automatically merged to avoid duplicates
- Expected data accumulation:
  - 1 day: ~25-50 articles
  - 1 week: ~175-350 articles
  - 1 month: ~750-1,500 articles
  - 1 year: ~9,000-18,000 articles

### Data Management
- CSV files are saved with timestamps for versioning
- Latest files are always updated for easy access
- Automatic deduplication when merging with existing data
- The ingestion script supports both 'replace' and 'append' modes
- Default data location: `data/raw_data/news/`
- Default database location: `data/raw_data/news/coindesk_news.duckdb`
- Logs stored in: `logs/coindesk_scraper_YYYYMMDD.log`

## Directory Structure

```
terminalC/
├── src/
│   ├── data_scripts/
│   │   └── web/
│   │       ├── coindesk_scraper.py      # Step 1: Scrape API → CSV/Pickle
│   │       ├── query_coindesk_db.py     # Step 3: Query database
│   │       ├── coindesk_api_desc.md     # API documentation
│   │       ├── requirements.txt          # Dependencies
│   │       └── README.md                 # This file
│   └── database/
│       └── ingest_to_duckdb.py          # Step 2: CSV → DuckDB
└── data/
    └── raw_data/
        └── news/
            ├── coindesk_articles_YYYYMMDD_HHMMSS.csv
            ├── coindesk_articles_YYYYMMDD_HHMMSS.pkl
            ├── coindesk_articles_latest.csv
            ├── coindesk_articles_latest.pkl
            ├── coindesk_sources.csv
            ├── coindesk_categories.csv
            └── coindesk_news.duckdb
```

## Troubleshooting

**No articles fetched:**
- The API may require authentication or the endpoint format may have changed
- Try running with `--verbose` flag to see detailed error messages
- Check if the API endpoints in the code match the current CoinDesk API

**Database errors:**
- Ensure DuckDB is installed: `pip install duckdb`
- Check that you have write permissions to the data directory
- Make sure you've run the scraper first before ingesting to database

**Import errors:**
- Install all dependencies: `pip install -r requirements.txt`
- Or manually: `pip install requests pandas duckdb`

**File not found errors:**
- Verify the data files exist in `data/raw_data/news/`
- Run the scraper first: `python coindesk_scraper.py`
- Then ingest to database: `python ../../database/ingest_to_duckdb.py`
