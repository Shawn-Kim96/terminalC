# main.py (발췌)
import os
import sys
import logging
import argparse
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

PROJECT_NAME = 'terminalC'
PROJECT_PATH = os.path.join(os.path.abspath('.').split(PROJECT_NAME)[0], PROJECT_NAME)
DATA_PATH = os.path.join(PROJECT_PATH, 'data', 'raw_data')
DATE_FROM = "2024-10-01T00:00:00Z"
print(PROJECT_PATH)
sys.path.append(PROJECT_PATH)

from src.utils.timing_utils import timeit, timer
from src.data_scripts.core.technical_analysis.divergence import DivergenceDetector
from src.data_scripts.core.technical_indicators import add_technical_indicators, preprocess_peaks_strict
from src.data_scripts.core.indicator_signals import (
    generate_indicator_signals,
    summarize_indicator_signals,
)
from src.data_scripts.strategy import build_strategy_catalog
from src.data_scripts.core.data_fetcher import (
    DataFetcher,
    _find_latest_dataset_dir,
    _infer_start_time_from_data,
)
import src.data_scripts.database.injest_to_duckdb as injest_db

if getattr(injest_db, "con", None) is not None:
    try:
        injest_db.con.close()
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)


asset_id_hashmap = {
    'btc': 1, 'eth': 2, 'xrp': 3, 'sol': 4, 'doge': 5, 'ada': 6, 'link': 7,
    'avax': 8, 'xlm': 9, 'hbar': 10, 'apt': 11, 'ondo': 12, 'sui': 13
}

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "LINK/USDT",
    "AVAX/USDT",
    "XLM/USDT",
    "HBAR/USDT",
    "APT/USDT",
    "ONDO/USDT",
    "SUI/USDT",
]

TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']

divergence_detector = DivergenceDetector()

@timeit("process_one_file")
def process_one_file(filepath: str) -> pd.DataFrame:
    coin, _, timeframe = filepath.split('/')[-1].split('.csv')[0].split('_')
    asset_id = asset_id_hashmap[coin.lower()]

    print('='*70)
    print(f"Preprocessing {coin} {timeframe} data")
    print('='*70)

    
    df = pd.read_csv(filepath)
    df['ts'] = pd.to_datetime(df['datetime'], utc=True)

    with timer(f"indicators {coin}_{timeframe}"):
        df = add_technical_indicators(df)
        df = preprocess_peaks_strict(df)
        df = df[df['ts'] >= pd.Timestamp(DATE_FROM, tz='UTC')]

        if 'timestamp' in df.columns and pd.api.types.is_integer_dtype(df['timestamp']):
            df['ts_int'] = df['timestamp'].astype('int64')
        else:
            df['ts_int'] = pd.NA
        df['coin'] = coin
        df['timeframe'] = timeframe
        df = df.sort_values(['coin', 'timeframe', 'ts'])

        cols = [
            "coin","timeframe","ts","open","high","low","close","volume",
            "rsi","ema_12","ema_26","macd","macd_signal","macd_hist",
            "bb_middle","bb_upper","bb_lower","willr","atr","atr_14",
            "plus_di_14","minus_di_14","adx","adx_14","cci","cci_14",
            "stoch_k_9","stoch_d_9_6","stoch_rsi_14","ultimate_osc","roc_12",
            "ema_13","bull_power_13","bear_power_13","bull_bear_power_13",
            "highs_lows_14",
            "sma_5","ema_5","sma_10","ema_10","sma_20","ema_20",
            "sma_50","ema_50","sma_100","ema_100","sma_200","ema_200",
            "peak_high_high", "peak_high_close", "peak_low_low", "peak_low_close", "ts_int"
        ]
        for column in cols:
            if column not in df.columns:
                df[column] = pd.NA
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


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="terminalC data orchestration entry point."
    )
    parser.add_argument(
        "--fetch-candle-data",
        dest="fetch_candle_data",
        action="store_true",
        help="Fetch new OHLCV data and rebuild derived candle artifacts.",
    )
    parser.add_argument(
        "--update-candle-database",
        dest="update_candle_databse",
        action="store_true",
        help="Ingest the latest processed candle data into DuckDB.",
    )
    parser.add_argument(
        "--fetch-news-data",
        dest="fetch_news_data",
        action="store_true",
        help="Run the CoinDesk RSS scraper to pull the latest news dataset.",
    )
    parser.add_argument(
        "--update-news-data",
        dest="update_news_data",
        action="store_true",
        help="Ingest the latest scraped news dataset into DuckDB.",
    )
    return parser.parse_args()


def resolve_candle_dataset_root() -> Path:
    base_path = Path(DATA_PATH)
    dataset_root = _find_latest_dataset_dir(base_path)
    if dataset_root is None:
        raise FileNotFoundError(
            f"No candle dataset directory found under {base_path}. "
            "Run data_fetcher.py with --start-date to bootstrap the dataset."
        )
    return dataset_root


def fetch_candle_data(dataset_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    logging.info("Fetching candle data into %s", dataset_root)
    start_time = _infer_start_time_from_data(dataset_root, SYMBOLS, TIMEFRAMES)
    if start_time is None:
        raise RuntimeError(
            f"Unable to infer start time from data in {dataset_root}. "
            "Ensure at least one timeframe CSV exists or provide a manual start date via data_fetcher.py."
        )
    logging.info("Next fetch window starts at %s UTC", start_time)
    data_fetcher = DataFetcher()
    for symbol in SYMBOLS:
        symbol_dir = dataset_root / symbol.split("/")[0].lower()
        data_fetcher.fetch_and_save_all_data(
            symbol,
            TIMEFRAMES,
            since=start_time,
            output_dir=str(symbol_dir)
        )


def process_candle_dataset(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_root = dataset_root.resolve()
    csv_files = []
    for root, _, files in os.walk(str(dataset_root)):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        logging.warning("No CSV files found under %s. Skipping candle processing.", dataset_root)
        return pd.DataFrame(), pd.DataFrame()

    total_candle_df = pd.DataFrame()
    total_divergence_df = pd.DataFrame()

    for filepath in tqdm(sorted(csv_files), desc="processing files"):
        candle_df, divergence_df = process_one_file(filepath)
        candle_df['asset_id'] = candle_df['coin'].apply(lambda x: asset_id_hashmap[x.lower()])
        candle_df = candle_df[['asset_id'] + [c for c in candle_df.columns if c != 'asset_id']]
        total_candle_df = pd.concat([total_candle_df, candle_df], ignore_index=True)
        total_divergence_df = pd.concat([total_divergence_df, divergence_df], ignore_index=True)

    with timer("save_final_artifacts"):
        candle_final_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'final_candle.pickle')
        os.makedirs(os.path.dirname(candle_final_path), exist_ok=True)
        total_candle_df.to_pickle(candle_final_path)

        divergence_final_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'final_divergence.pickle')
        os.makedirs(os.path.dirname(divergence_final_path), exist_ok=True)
        total_divergence_df.to_pickle(divergence_final_path)

        indicator_signals_df = generate_indicator_signals(candle_df=total_candle_df)
        indicator_summary_df = summarize_indicator_signals(indicator_signals_df)

        signals_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'final_indicator_signals.pickle')
        summary_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'final_indicator_summary.pickle')
        indicator_signals_df.to_pickle(signals_path)
        indicator_summary_df.to_pickle(summary_path)

        strategy_df = build_strategy_catalog()
        strategy_path = os.path.join(PROJECT_PATH, 'data', 'processed', 'final_strategies.pickle')
        strategy_df.to_pickle(strategy_path)

    return total_candle_df, total_divergence_df


def run_coindesk_scraper():
    logging.info("Running CoinDesk RSS scraper...")
    import src.data_scripts.web.coindesk_scraper as coindesk_scraper

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        coindesk_scraper.main()
    finally:
        sys.argv = original_argv


def _connect_duckdb():
    database_dir = Path(PROJECT_PATH) / 'data' / 'database'
    database_dir.mkdir(parents=True, exist_ok=True)
    db_path = database_dir / 'market.duckdb'
    return duckdb.connect(str(db_path))


def update_candle_database():
    logging.info("Updating DuckDB candle tables...")
    con = _connect_duckdb()
    try:
        injest_db.generate_core_table(con)
        con.commit()
    finally:
        con.close()


def update_news_database():
    logging.info("Updating DuckDB news table...")
    con = _connect_duckdb()
    try:
        injest_db.generate_news_table(con)
        con.commit()
    finally:
        con.close()

@timeit("main")
def main():
    args = parse_cli_args()
    actions_requested = any([
        args.fetch_candle_data,
        args.update_candle_databse,
        args.fetch_news_data,
        args.update_news_data,
    ])

    if not actions_requested:
        logging.info("No actions requested. Use --help to see available options.")
        return

    if args.fetch_candle_data:
        try:
            dataset_root = resolve_candle_dataset_root()
            fetch_candle_data(dataset_root)
            process_candle_dataset(dataset_root)
        except Exception:
            logging.exception("Candle data workflow failed.")
            raise

    if args.update_candle_databse:
        update_candle_database()

    if args.fetch_news_data:
        run_coindesk_scraper()

    if args.update_news_data:
        update_news_database()


if __name__ == "__main__":
    main()
