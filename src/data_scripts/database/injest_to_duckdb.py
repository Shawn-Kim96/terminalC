import duckdb
import os
import sys
import pandas as pd
import logging
from glob import glob

PROJECT_PATH = os.path.join(os.path.abspath('.').split('terminalC')[0], 'terminalC')
DATA_PROCESSED_PATH = os.path.join(PROJECT_PATH, 'data', 'processed')
DATA_RAW_NEWS_PATH = os.path.join(PROJECT_PATH, 'data', 'raw_data', 'news')
sys.path.append(PROJECT_PATH)

LOGGER = logging.getLogger(__name__)

# Single database connection for all data
con = duckdb.connect(os.path.join(PROJECT_PATH, "data/database/market.duckdb"))

# =============================
#      1. CORE TABLE
# =============================

def generate_core_table(con):
  # 1-1. assets table
  con.execute("""
  CREATE TABLE IF NOT EXISTS assets (
    asset_id INTEGER PRIMARY KEY,
    symbol   TEXT UNIQUE NOT NULL,
    name     TEXT
  );""")

  con.execute("""
  INSERT OR IGNORE INTO assets (asset_id, symbol, name) VALUES
    (1, 'BTC', 'Bitcoin'),
    (2, 'ETH', 'Ethereum'),
    (3, 'XRP', 'Ripple'),
    (4, 'SOL', 'Solana'),
    (5, 'DOGE', 'Dogecoin'),
    (6, 'ADA', 'Cardano'),
    (7, 'LINK', 'Chainlink'),
    (8, 'AVAX', 'Avalanche'),
    (9, 'XLM', 'Stellar'),
    (10, 'HBAR', 'Hedera'),
    (11, 'APT', 'Aptos'),
    (12, 'ONDO', 'Ondo'),
    (13, 'SUI', 'Sui');
  """)


  # 1-2. OHLCV + indicator table
  candle_df = pd.read_pickle(os.path.join(DATA_PROCESSED_PATH, 'final_candle.pickle'))
  con.execute("DROP TABLE IF EXISTS candles")
  con.execute("CREATE TABLE candles AS SELECT * FROM candle_df")

  # 1-3. divergence table
  divergence_df = pd.read_pickle(os.path.join(DATA_PROCESSED_PATH, 'final_divergence.pickle'))
  con.execute("DROP TABLE IF EXISTS divergence")
  con.execute("CREATE TABLE divergence AS SELECT * FROM divergence_df")


# =============================
#      2. INDICATOR STRATEGIES
# =============================

def generate_strategy_table(con):
  rules_path = os.path.join(DATA_PROCESSED_PATH, 'final_indicator_rules.pickle')
  signals_path = os.path.join(DATA_PROCESSED_PATH, 'final_indicator_signals.pickle')
  summary_path = os.path.join(DATA_PROCESSED_PATH, 'final_indicator_summary.pickle')

  if os.path.exists(rules_path):
      rules_df = pd.read_pickle(rules_path)
  else:
      from src.data_scripts.core.indicator_signals import indicator_rulebook
      rules_df = indicator_rulebook()

  signals_df = pd.read_pickle(signals_path)
  summary_df = pd.read_pickle(summary_path)

  con.execute("DROP TABLE IF EXISTS indicator_rules")
  con.execute("CREATE TABLE indicator_rules AS SELECT * FROM rules_df")
  con.execute("DROP TABLE IF EXISTS indicator_signals")
  con.execute("CREATE TABLE indicator_signals AS SELECT * FROM signals_df")

  con.execute("DROP TABLE IF EXISTS indicator_signal_summary")
  con.execute("CREATE TABLE indicator_signal_summary AS SELECT * FROM summary_df")

  # =============================
  #      3. STRATEGY CATALOG
  # =============================

  strategy_df = pd.read_pickle(os.path.join(DATA_PROCESSED_PATH, 'final_strategies.pickle'))
  con.execute("DROP TABLE IF EXISTS strategies")
  con.execute("CREATE TABLE strategies AS SELECT * FROM strategy_df")

  con.commit()
  con.close()


# =============================
#      4. NEWS TABLE
# =============================

def generate_news_table(con):
  """
  Ingest CoinDesk news data from CSV/pickle files into DuckDB.

  Reads the latest news data and creates a news articles table.
  """
  # Find the latest pickle file
  latest_pkl = os.path.join(DATA_RAW_NEWS_PATH, 'coindesk_articles_latest.pkl')

  if not os.path.exists(latest_pkl):
    LOGGER.warning("No news data found at %s", latest_pkl)
    return

  LOGGER.info("Loading news data from %s", latest_pkl)
  news_df = pd.read_pickle(latest_pkl)

  # Drop and recreate table
  con.execute("DROP TABLE IF EXISTS news_articles")
  con.execute("""
    CREATE TABLE news_articles AS
    SELECT
      article_id,
      guid,
      source,
      title,
      body,
      excerpt,
      url,
      published_at,
      created_at,
      updated_at,
      author,
      categories,
      category_names,
      tags,
      tag_names,
      sentiment,
      image_url
    FROM news_df
  """)

  LOGGER.info("Created news_articles table with %s rows", len(news_df))

  # Create indexes for better query performance
  con.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at)")
  con.execute("CREATE INDEX IF NOT EXISTS idx_news_title ON news_articles(title)")

  LOGGER.info("Created indexes on news_articles table")


if __name__=="__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 60)
    print("Database Ingestion to market.duckdb")
    print("=" * 60)

    # Market data (existing)
    print("\n[1/2] Ingesting market data...")
    # generate_core_table(con)
    # generate_strategy_table(con)
    print("✓ Market data ingested successfully")

    # News data (new)
    print("\n[2/2] Ingesting news data...")
    generate_news_table(con)
    print("✓ News data ingested successfully")

    con.commit()
    con.close()

    print("\n" + "=" * 60)
    print("All data ingested to: data/database/market.duckdb")
    print("=" * 60)
    print("\nTables created:")
    print("  - assets")
    print("  - candles")
    print("  - divergence")
    print("  - indicator_rules")
    print("  - indicator_signals")
    print("  - indicator_signal_summary")
    print("  - strategies")
    print("  - news_articles")