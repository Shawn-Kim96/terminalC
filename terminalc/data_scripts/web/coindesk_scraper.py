#!/usr/bin/env python3
"""
CoinDesk News Scraper using RSS Feed

Fetches cryptocurrency news from CoinDesk's RSS feed and saves to CSV/Pickle files.
Note: RSS feeds typically contain only the most recent ~20-50 articles.
For historical data, run this script regularly to accumulate data over time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from html import unescape
import os
import sys
import re
import pandas as pd
import requests
from time import sleep
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser

# CoinDesk RSS Feed URL
RSS_FEED_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/xml",
}
LOGGER = logging.getLogger("coindesk_scraper")

PROJECT_NAME = 'terminalC'
PROJECT_PATH = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'raw_data', 'news')

print(f"Project path: {PROJECT_PATH}")
print(f"Data path: {DATA_PATH}")
sys.path.append(PROJECT_PATH)


def strip_html(html: str) -> str:
    """Remove HTML tags from text."""
    if not html:
        return ""
    text = unescape(html)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def fetch_rss_feed(url: str = RSS_FEED_URL) -> str:
    """Fetch RSS feed content."""
    try:
        LOGGER.info("Fetching RSS feed from: %s", url)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        LOGGER.error("Failed to fetch RSS feed: %s", e)
        raise


def parse_rss_feed(xml_content: str, cutoff: dt.datetime = None) -> list[dict]:
    """
    Parse RSS feed XML and extract article information.

    Args:
        xml_content: RSS feed XML content
        cutoff: Optional datetime cutoff (articles older than this will be excluded)

    Returns:
        List of article dictionaries
    """
    articles = []

    try:
        root = ET.fromstring(xml_content)

        # Find all item elements
        items = root.findall('.//item')
        LOGGER.info("Found %s items in RSS feed", len(items))

        for item in items:
            try:
                # Extract basic fields
                title = item.find('title')
                link = item.find('link')
                guid = item.find('guid')
                pub_date = item.find('pubDate')
                description = item.find('description')
                creator = item.find('.//{http://purl.org/dc/elements/1.1/}creator')

                # Parse publication date
                if pub_date is not None and pub_date.text:
                    published_at = date_parser.parse(pub_date.text)

                    # Check cutoff
                    if cutoff and published_at < cutoff:
                        LOGGER.debug("Skipping article older than cutoff: %s", title.text if title is not None else "Unknown")
                        continue
                else:
                    published_at = None

                # Extract categories/tags
                categories = []
                tags = []
                for category in item.findall('category'):
                    cat_text = category.text
                    domain = category.get('domain', '')

                    if 'coindesk.com' in domain and cat_text:
                        categories.append(cat_text)
                    elif domain == 'tag' and cat_text:
                        tags.append(cat_text)

                # Extract media content (image)
                media_content = item.find('.//{http://search.yahoo.com/mrss/}content')
                image_url = media_content.get('url') if media_content is not None else None

                article = {
                    'article_id': guid.text if guid is not None else None,
                    'guid': guid.text if guid is not None else None,
                    'source': 'CoinDesk',
                    'title': strip_html(title.text) if title is not None else '',
                    'body': strip_html(description.text) if description is not None else '',
                    'excerpt': strip_html(description.text) if description is not None else '',
                    'url': link.text if link is not None else '',
                    'published_at': published_at,
                    'created_at': published_at,
                    'updated_at': None,
                    'author': creator.text if creator is not None else '',
                    'categories': ','.join(categories),
                    'category_names': ','.join(categories),
                    'tags': ','.join(tags),
                    'tag_names': ','.join(tags),
                    'sentiment': None,  # Not provided in RSS feed
                    'image_url': image_url,
                }

                articles.append(article)

            except Exception as e:
                LOGGER.warning("Error parsing RSS item: %s", e)
                continue

        LOGGER.info("Successfully parsed %s articles", len(articles))
        return articles

    except ET.ParseError as e:
        LOGGER.error("Failed to parse RSS XML: %s", e)
        raise


def merge_with_existing(new_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Merge new articles with existing data, avoiding duplicates.

    Args:
        new_df: New articles DataFrame
        output_dir: Directory containing existing data files

    Returns:
        Merged DataFrame
    """
    latest_pkl = os.path.join(output_dir, "coindesk_articles_latest.pkl")

    if os.path.exists(latest_pkl):
        try:
            LOGGER.info("Loading existing data for merge...")
            existing_df = pd.read_pickle(latest_pkl)

            # Combine and remove duplicates based on guid/article_id
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Remove duplicates (prefer newer entries)
            combined_df = combined_df.drop_duplicates(
                subset=['article_id'],
                keep='last'
            )

            LOGGER.info("Merged: %s existing + %s new = %s total (%s unique)",
                       len(existing_df), len(new_df),
                       len(existing_df) + len(new_df), len(combined_df))

            return combined_df

        except Exception as e:
            LOGGER.warning("Could not load existing data: %s. Using new data only.", e)
            return new_df
    else:
        LOGGER.info("No existing data found. Using new data only.")
        return new_df


def convert_to_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Convert article list to pandas DataFrame."""
    if not articles:
        LOGGER.warning("No articles to convert")
        return pd.DataFrame()

    df = pd.DataFrame(articles)

    # Sort by publication date (newest first)
    if 'published_at' in df.columns:
        df = df.sort_values('published_at', ascending=False)

    LOGGER.info("Converted %s articles to DataFrame", len(df))
    return df


def save_to_files(
    articles_df: pd.DataFrame,
    output_dir: str,
    timestamp: str = None
) -> list[str]:
    """Save DataFrame to CSV and pickle files."""
    os.makedirs(output_dir, exist_ok=True)

    if timestamp is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = []

    if not articles_df.empty:
        # Save as CSV
        csv_path = os.path.join(output_dir, f"coindesk_articles_{timestamp}.csv")
        articles_df.to_csv(csv_path, index=False, encoding='utf-8')
        LOGGER.info("Saved %s articles to CSV: %s", len(articles_df), csv_path)
        saved_files.append(csv_path)

        # Save as pickle for better data type preservation
        pickle_path = os.path.join(output_dir, f"coindesk_articles_{timestamp}.pkl")
        articles_df.to_pickle(pickle_path)
        LOGGER.info("Saved articles to pickle: %s", pickle_path)
        saved_files.append(pickle_path)

        # Also save as latest version for easy access
        latest_csv = os.path.join(output_dir, "coindesk_articles_latest.csv")
        latest_pkl = os.path.join(output_dir, "coindesk_articles_latest.pkl")
        articles_df.to_csv(latest_csv, index=False, encoding='utf-8')
        articles_df.to_pickle(latest_pkl)
        LOGGER.info("Updated latest versions")
    else:
        LOGGER.warning("No articles to save")

    return saved_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape recent CoinDesk articles from RSS feed and save to CSV/Pickle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch from RSS feed (typically ~20-50 most recent articles)
  python coindesk_scraper.py

  # Filter to only articles from last 7 days
  python coindesk_scraper.py --days 7

  # Don't merge with existing data
  python coindesk_scraper.py --no-merge

  # Specify custom output directory
  python coindesk_scraper.py --output /path/to/output

Note: RSS feeds only contain recent articles. To build historical data,
run this script regularly (e.g., daily via cron job).
        """
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only include articles from last N days (default: include all from RSS feed).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DATA_PATH,
        help=f"Output directory for CSV/pickle files (default: {DATA_PATH}).",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge with existing data (default: merge to avoid duplicates).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the CoinDesk scraper."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    LOGGER.info("=" * 60)
    LOGGER.info("CoinDesk News Scraper (RSS Feed)")
    LOGGER.info("=" * 60)
    LOGGER.info("Configuration:")
    LOGGER.info("  Output directory: %s", args.output)
    LOGGER.info("  Days filter: %s", args.days or "None (all from feed)")
    LOGGER.info("  Merge mode: %s", "disabled" if args.no_merge else "enabled")
    LOGGER.info("=" * 60)

    # Calculate cutoff date if specified
    cutoff = None
    if args.days:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)
        LOGGER.info("Filtering articles published after: %s", cutoff.isoformat())

    try:
        # Fetch RSS feed
        LOGGER.info("\n[1/3] Fetching RSS feed...")
        xml_content = fetch_rss_feed()

        # Parse RSS feed
        LOGGER.info("\n[2/3] Parsing RSS feed...")
        articles = parse_rss_feed(xml_content, cutoff)

        if not articles:
            LOGGER.warning("No articles found in RSS feed matching criteria.")
            return

        # Convert to DataFrame
        articles_df = convert_to_dataframe(articles)

        # Merge with existing data if enabled
        if not args.no_merge:
            articles_df = merge_with_existing(articles_df, args.output)

        # Save to files
        LOGGER.info("\n[3/3] Saving to files...")
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = save_to_files(articles_df, args.output, timestamp)

        LOGGER.info("\n" + "=" * 60)
        LOGGER.info("SUCCESS! Data saved to: %s", args.output)
        LOGGER.info("=" * 60)
        LOGGER.info("\nSaved files:")
        for file in saved_files:
            LOGGER.info("  - %s", file)

        # Show summary
        LOGGER.info("\nSummary:")
        LOGGER.info("  Total articles in dataset: %s", len(articles_df))
        if not articles_df.empty and 'published_at' in articles_df.columns:
            valid_dates = articles_df['published_at'].dropna()
            if not valid_dates.empty:
                LOGGER.info("  Date range: %s to %s",
                           valid_dates.min(),
                           valid_dates.max())

        LOGGER.info("\nTo ingest into DuckDB database, run:")
        LOGGER.info("  python src/database/ingest_to_duckdb.py")

        LOGGER.info("\nNote: RSS feeds only show recent articles (~20-50).")
        LOGGER.info("For historical data, run this script regularly (e.g., daily cron job).")

    except KeyboardInterrupt:
        LOGGER.info("\nInterrupted by user. Exiting...")
    except Exception as e:
        LOGGER.error("\nFATAL ERROR: %s", e, exc_info=args.verbose)
        raise


if __name__ == "__main__":
    main()
