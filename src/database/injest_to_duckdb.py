import duckdb
import json
from pathlib import Path
import os
import sys
import pandas as pd
import re


PROJECT_PATH = os.path.join(os.path.abspath('.').split('terminalC')[0], 'terminalC')
DATA_RAW_PATH = os.path.join(PROJECT_PATH, 'data', 'raw_data')
DATA_PROCESSED_PATH = os.path.join(PROJECT_PATH, 'data', 'processed')
sys.path.append(PROJECT_PATH)

con = duckdb.connect(os.path.join(PROJECT_PATH, "data/database/market.duckdb"))

# 1. CORE
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
# con.execute("""
# CREATE TABLE IF NOT EXISTS candles (
#   coin TEXT, timeframe TEXT, ts TIMESTAMP,
#   open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
#   rsi DOUBLE, ema_12 DOUBLE, ema_26 DOUBLE, macd DOUBLE, macd_signal DOUBLE, macd_hist DOUBLE,
#   bb_middle DOUBLE, bb_upper DOUBLE, bb_lower DOUBLE, willr DOUBLE, atr DOUBLE, adx DOUBLE, cci DOUBLE, 
#   peak_high_high BOOLEAN, peak_high_close BOOLEAN, peak_low_low BOOLEAN, peak_low_close BOOLEAN, ts_int BIGINT
# );
            
# CREATE INDEX IF NOT EXISTS coin_timeframe_ts ON candles(coin, timeframe, ts);
# """)
# from src.data_process.data_preprocess import *

# DATE_FROM = "2024-10-01T00:00:00Z"

# csv_files = []
# for root, dirs, files in os.walk(os.path.join(DATA_PATH, "240815")):
#     for file in files:
#         if file.lower().endswith(".csv"):
#             abs_path = os.path.join(root, file)
#             csv_files.append(abs_path)


# for filepath in csv_files:
#     coin, _, timeframe = filepath.split('/')[-1].split('.csv')[0].split('_')

#     df = pd.read_csv(filepath)    
#     df['ts'] = pd.to_datetime(df['datetime'], utc=True)

#     df = calculate_rsi(df)
#     df = calculate_technical_indicators(df)
#     df = preprocess_peaks_strict(df)
#     df = df[df['ts'] >= pd.Timestamp(DATE_FROM, tz='UTC')]

#     if 'timestamp' in df.columns and pd.api.types.is_integer_dtype(df['timestamp']):
#         df['ts_int'] = df['timestamp'].astype('int64')
#     else:
#         df['ts_int'] = pd.NA
    

#     df['coin'] = coin
#     df['timeframe'] = timeframe

#     df = df.sort_values(['coin', 'timeframe', 'ts'])

#     cols = ["coin","timeframe","ts","open","high","low","close","volume",
#             "rsi","ema_12","ema_26","macd","macd_signal","macd_hist",
#             "bb_middle","bb_upper","bb_lower","willr","atr","adx","cci","peak_high_high", "peak_high_close", "peak_low_low", "peak_low_close", "ts_int"]
    
#     for c in cols:
#         if c not in df.columns:
#             df[c] = pd.NA
#     df = df[cols]

#     con.register("staging_df", df)
#     con.execute("""
#         CREATE TEMP TABLE staging AS
#         SELECT * FROM staging_df;

#         -- 2) delete duplicates (coin, timeframe, ts 기준 최신 1건)
#         CREATE TEMP TABLE staging_dedup AS
#         SELECT *
#         FROM (
#         SELECT s.*,
#                 ROW_NUMBER() OVER (PARTITION BY coin, timeframe, ts ORDER BY ts DESC) AS rn
#         FROM staging s
#         )
#         WHERE rn = 1;

#         -- 3) upsert
#         MERGE INTO candles AS tgt
#         USING staging_dedup AS src
#         ON  tgt.coin = src.coin
#         AND tgt.timeframe = src.timeframe
#         AND tgt.ts = src.ts
#         WHEN MATCHED THEN UPDATE SET
#         open  = src.open,
#         high  = src.high,
#         low   = src.low,
#         close = src.close,
#         volume = src.volume,
#         rsi   = src.rsi,
#         ema_12 = src.ema_12,
#         ema_26 = src.ema_26,
#         macd = src.macd,
#         macd_signal = src.macd_signal,
#         macd_hist   = src.macd_hist,
#         bb_middle = src.bb_middle,
#         bb_upper  = src.bb_upper,
#         bb_lower  = src.bb_lower,
#         willr = src.willr,
#         atr   = src.atr,
#         adx   = src.adx,
#         cci   = src.cci,
#         peak_high_high  = src.peak_high_high,
#         peak_high_close = src.peak_high_close,
#         peak_low_low    = src.peak_low_low,
#         peak_low_close  = src.peak_low_close,
#         ts_int = src.ts_int

#         WHEN NOT MATCHED THEN INSERT (
#         coin, timeframe, ts, open, high, low, close, volume,
#         rsi, ema_12, ema_26, macd, macd_signal, macd_hist,
#         bb_middle, bb_upper, bb_lower, willr, atr, adx, cci, ts_int,
#         peak_high_high, peak_high_close, peak_low_low, peak_low_close
#         ) VALUES (
#         src.coin, src.timeframe, src.ts, src.open, src.high, src.low, src.close, src.volume,
#         src.rsi, src.ema_12, src.ema_26, src.macd, src.macd_signal, src.macd_hist,
#         src.bb_middle, src.bb_upper, src.bb_lower, src.willr, src.atr, src.adx, src.cci, src.ts_int,
#         src.peak_high_high, src.peak_high_close, src.peak_low_low, src.peak_low_close
#         );

#         DROP TABLE staging_dedup;
#     """)

#     con.unregister("staging_df")
#     con.execute("DROP TABLE staging;")


con.commit()
con.close()

