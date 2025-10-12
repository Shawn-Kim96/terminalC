import duckdb
import os
import sys
import pandas as pd

PROJECT_PATH = os.path.join(os.path.abspath('.').split('terminalC')[0], 'terminalC')
DATA_PROCESSED_PATH = os.path.join(PROJECT_PATH, 'data', 'processed')
sys.path.append(PROJECT_PATH)

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

if __name__=="__main__":
    generate_core_table()
    generate_strategy_table()