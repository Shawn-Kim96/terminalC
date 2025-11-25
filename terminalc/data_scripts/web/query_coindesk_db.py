#!/usr/bin/env python3
"""
Utility script to query and analyze the CoinDesk news database.

This script provides easy access to the scraped news data stored in DuckDB.
"""
from __future__ import annotations

import argparse
import os
import sys
import duckdb
import pandas as pd
from pathlib import Path

PROJECT_NAME = 'terminalC'
PROJECT_PATH = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'database')
DEFAULT_DB_PATH = os.path.join(DATA_PATH, 'market.duckdb')


def print_stats(db_path: str) -> None:
    """Print database statistics."""
    conn = duckdb.connect(db_path, read_only=True)

    print("=" * 70)
    print("CoinDesk News Database Statistics")
    print("=" * 70)

    # Total articles
    total = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
    print(f"\nTotal Articles: {total:,}")

    if total == 0:
        print("\nNo articles found in database.")
        conn.close()
        return

    # Date range
    date_range = conn.execute("""
        SELECT
            MIN(published_at) as earliest,
            MAX(published_at) as latest
        FROM news_articles
    """).fetchone()
    print(f"Date Range: {date_range[0]} to {date_range[1]}")

    # Articles by source
    print("\nTop 10 Sources:")
    sources = conn.execute("""
        SELECT source, COUNT(*) as count
        FROM news_articles
        WHERE source IS NOT NULL
        GROUP BY source
        ORDER BY count DESC
        LIMIT 10
    """).fetchdf()
    print(sources.to_string(index=False))

    # Sentiment distribution
    print("\nSentiment Distribution:")
    sentiment = conn.execute("""
        SELECT
            sentiment,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
        FROM news_articles
        GROUP BY sentiment
        ORDER BY count DESC
    """).fetchdf()
    print(sentiment.to_string(index=False))

    # Articles per month
    print("\nArticles per Month (Last 12 months):")
    monthly = conn.execute("""
        SELECT
            strftime(published_at, '%Y-%m') as month,
            COUNT(*) as count
        FROM news_articles
        WHERE published_at >= CURRENT_DATE - INTERVAL '12 months'
        GROUP BY month
        ORDER BY month DESC
        LIMIT 12
    """).fetchdf()
    print(monthly.to_string(index=False))

    conn.close()
    print("\n" + "=" * 70)


def search_articles(db_path: str, keyword: str, limit: int = 10) -> None:
    """Search articles by keyword."""
    conn = duckdb.connect(db_path, read_only=True)

    print(f"\nSearching for: '{keyword}'")
    print("=" * 70)

    results = conn.execute("""
        SELECT
            title,
            source,
            published_at,
            url
        FROM news_articles
        WHERE
            title ILIKE ? OR
            body ILIKE ?
        ORDER BY published_at DESC
        LIMIT ?
    """, [f'%{keyword}%', f'%{keyword}%', limit]).fetchdf()

    if len(results) == 0:
        print("No articles found.")
    else:
        for idx, row in results.iterrows():
            print(f"\n[{idx + 1}] {row['title']}")
            print(f"    Source: {row['source']} | Published: {row['published_at']}")
            print(f"    URL: {row['url']}")

    conn.close()
    print("\n" + "=" * 70)


def export_to_csv(db_path: str, output_path: str) -> None:
    """Export all articles to CSV."""
    conn = duckdb.connect(db_path, read_only=True)

    print(f"Exporting data to: {output_path}")

    df = conn.execute("""
        SELECT
            article_id,
            title,
            source,
            author,
            published_at,
            sentiment,
            url,
            body
        FROM news_articles
        ORDER BY published_at DESC
    """).fetchdf()

    df.to_csv(output_path, index=False)
    print(f"Exported {len(df):,} articles to {output_path}")

    conn.close()


def get_recent(db_path: str, days: int = 7, limit: int = 20) -> None:
    """Get recent articles."""
    conn = duckdb.connect(db_path, read_only=True)

    print(f"\nMost Recent Articles (Last {days} days, limit {limit}):")
    print("=" * 70)

    results = conn.execute("""
        SELECT
            title,
            source,
            published_at,
            sentiment,
            url
        FROM news_articles
        WHERE published_at >= CURRENT_DATE - INTERVAL ? DAY
        ORDER BY published_at DESC
        LIMIT ?
    """, [days, limit]).fetchdf()

    if len(results) == 0:
        print(f"No articles found in the last {days} days.")
    else:
        for idx, row in results.iterrows():
            sentiment_tag = f"[{row['sentiment']}]" if pd.notna(row['sentiment']) else ""
            print(f"\n[{idx + 1}] {row['title']} {sentiment_tag}")
            print(f"    Source: {row['source']} | Published: {row['published_at']}")
            print(f"    URL: {row['url']}")

    conn.close()
    print("\n" + "=" * 70)


def run_custom_query(db_path: str, query: str) -> None:
    """Run a custom SQL query."""
    conn = duckdb.connect(db_path, read_only=True)

    print("\nExecuting query:")
    print(query)
    print("=" * 70)

    try:
        result = conn.execute(query).fetchdf()
        print(result.to_string())
    except Exception as e:
        print(f"Error executing query: {e}")

    conn.close()
    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query and analyze CoinDesk news database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show database statistics
  python query_coindesk_db.py --stats

  # Search for articles about Bitcoin
  python query_coindesk_db.py --search "bitcoin" --limit 20

  # Get recent articles from last 7 days
  python query_coindesk_db.py --recent --days 7

  # Export all data to CSV
  python query_coindesk_db.py --export output.csv

  # Run custom SQL query
  python query_coindesk_db.py --query "SELECT COUNT(*) FROM news_articles WHERE sentiment = 'POSITIVE'"
        """
    )

    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to DuckDB database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search articles by keyword",
    )
    parser.add_argument(
        "--recent",
        action="store_true",
        help="Show recent articles",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days for recent articles (default: 7)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of results (default: 10)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export all articles to CSV file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run custom SQL query",
    )

    args = parser.parse_args()

    # Check if database exists
    if not os.path.exists(args.db):
        print(f"Error: Database not found at {args.db}")
        print("Run coindesk_scraper.py first to create the database.")
        sys.exit(1)

    # Execute requested action
    if args.stats:
        print_stats(args.db)
    elif args.search:
        search_articles(args.db, args.search, args.limit)
    elif args.recent:
        get_recent(args.db, args.days, args.limit)
    elif args.export:
        export_to_csv(args.db, args.export)
    elif args.query:
        run_custom_query(args.db, args.query)
    else:
        # Default: show stats
        print_stats(args.db)


if __name__ == "__main__":
    main()
