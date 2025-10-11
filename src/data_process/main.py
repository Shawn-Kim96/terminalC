# main.py (발췌)
import os
import sys
import logging
import pandas as pd
from tqdm import tqdm

PROJECT_NAME = 'terminalC'
PROJECT_PATH = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'raw_data')
DATE_FROM = "2024-10-01T00:00:00Z"
print(PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from src.utils.timing_utils import timeit, timer
from src.data_process.technical_analysis.divergence import DivergenceDetector
from src.data_process.data_preprocess import *

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)


asset_id_hashmap = {
    'btc': 1, 'eth': 2, 'xrp': 3, 'sol': 4, 'doge': 5, 'ada': 6, 'link': 7,
    'avax': 8, 'xlm': 9, 'hbar': 10, 'apt': 11, 'ondo': 12, 'sui': 13
}

divergence_detector = DivergenceDetector()

@timeit("process_one_file")  # <-- for-loop마다 한 번씩 타이밍 로그가 찍힘
def process_one_file(filepath: str) -> pd.DataFrame:
    coin, _, timeframe = filepath.split('/')[-1].split('.csv')[0].split('_')
    asset_id = asset_id_hashmap[coin.lower()]

    print('='*70)
    print(f"Preprocessing {coin} {timeframe} data")
    print('='*70)

    
    df = pd.read_csv(filepath)
    df['ts'] = pd.to_datetime(df['datetime'], utc=True)

    with timer(f"indicators {coin}_{timeframe}"):
        df = calculate_rsi(df)
        df = calculate_technical_indicators(df)
        df = preprocess_peaks_strict(df)
        df = df[df['ts'] >= pd.Timestamp(DATE_FROM, tz='UTC')]

        if 'timestamp' in df.columns and pd.api.types.is_integer_dtype(df['timestamp']):
            df['ts_int'] = df['timestamp'].astype('int64')
        else:
            df['ts_int'] = pd.NA
        df['coin'] = coin
        df['timeframe'] = timeframe
        df = df.sort_values(['coin', 'timeframe', 'ts'])

        cols = ["coin","timeframe","ts","open","high","low","close","volume",
                "rsi","ema_12","ema_26","macd","macd_signal","macd_hist",
                "bb_middle","bb_upper","bb_lower","willr","atr","adx","cci",
                "peak_high_high", "peak_high_close", "peak_low_low", "peak_low_close", "ts_int"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[cols]

        candle_df_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'candle', f'candle_{coin}_{timeframe}.pickle')
        os.makedirs(os.path.dirname(candle_df_path), exist_ok=True)
        df.to_pickle(candle_df_path)

    with timer(f"divergence {coin}_{timeframe}"):
        divergence_df = divergence_detector.find_divergences(
            df, asset_id, timeframe, bullish_rsi_threshold=35, bearish_rsi_threshold=65
        )

        divergence_df_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'divergence', f'divergence_{coin}_{timeframe}.pickle')
        os.makedirs(os.path.dirname(divergence_df_path), exist_ok=True)
        divergence_df.to_pickle(divergence_df_path)

    return df, divergence_df

@timeit("main")
def main():
    # 1) read raw data
    csv_files = []
    for root, dirs, files in os.walk(os.path.join(DATA_PATH, "240815")):
        for file in files:
            if file.lower().endswith(".csv"):
                abs_path = os.path.join(root, file)
                csv_files.append(abs_path)

    total_candle_df = pd.DataFrame()
    total_divergence_df = pd.DataFrame()


    # 2) preprocess + divergence
    for filepath in tqdm(csv_files, desc="processing files"):
        candle_df, divergence_df = process_one_file(filepath)
        candle_df['asset_id'] = candle_df['coin'].apply(lambda x: asset_id_hashmap[x.lower()])
        candle_df = candle_df[['asset_id'] + [c for c in candle_df.columns if c != 'asset_id']]
        total_candle_df = pd.concat([total_candle_df, candle_df], ignore_index=True)
        total_divergence_df = pd.concat([total_divergence_df, divergence_df], ignore_index=True)

    # 3) save final
    with timer("save_final_divergence"):
        candle_final_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'final_candle.pickle')
        os.makedirs(os.path.dirname(candle_final_path), exist_ok=True)
        total_candle_df.to_pickle(candle_final_path)

        divergence_final_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'final_divergence.pickle')
        os.makedirs(os.path.dirname(divergence_final_path), exist_ok=True)
        total_divergence_df.to_pickle(divergence_final_path)

if __name__ == "__main__":
    main()
