import ccxt
import pandas as pd
import os
import logging
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Set up logging
load_dotenv(".env")

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


class DataFetcher:
    """
    Fetch historical OHLCV data for given symbol and timeframes from an exchange platform.
    """
    def __init__(self, exchange_name='binance'):
        self.exchange = getattr(ccxt, exchange_name)()

    def fetch_and_save_all_data(self, symbol, timeframes, output_dir='data', since=None):
        """
        Fetch all historical OHLCV data for the specified symbol and save it as CSV files.

        Args:
            symbol (str): Symbol to fetch data for (e.g., 'BTC/USDT').
            timeframes (list): List of timeframes to fetch (e.g., ['1m', '5m', '15m']).
            output_dir (str): Directory to save the CSV files.
            since (str): Starting date in ISO format (e.g., '2020-01-01T00:00:00Z').
        """
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

        for timeframe in timeframes:
            logging.info(f"Starting to fetch {timeframe} data for {symbol}.")
            start_time = time.time()
            try:
                # Fetch data
                df = self.fetch_ohlcv(symbol, timeframe, since=since)
                # Save to CSV
                file_path = os.path.join(output_dir, f'{symbol.replace("/", "_")}_{timeframe}.csv')

                # If file exists, concatenate with existing data
                if os.path.exists(file_path):
                    logging.info(f"File {file_path} exists. Concatenating with existing data.")
                    existing_df = pd.read_csv(file_path, index_col='datetime', parse_dates=['datetime'])
                    df = pd.concat([existing_df, df])
                    df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
                    df.sort_index(inplace=True)  # Ensure chronological order

                df.to_csv(file_path)
                logging.info(f"Successfully saved {timeframe} data to {file_path}.")
            except Exception as e:
                logging.error(f"Error fetching {timeframe} data for {symbol}: {str(e)}")
            end_time = time.time()
            logging.info(f"Fetching {timeframe} data took {end_time - start_time:.2f} seconds.")

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        """
        Fetch historical OHLCV data for the specified symbol and timeframe.

        Args:
            symbol (str): Symbol to fetch data for (e.g., 'BTC/USDT').
            timeframe (str): Timeframe to fetch (e.g., '1m', '5m').
            since (str): Starting date in ISO format (e.g., '2020-01-01T00:00:00Z').
            limit (int): Maximum number of candles to fetch in each request.

        Returns:
            pd.DataFrame: DataFrame containing OHLCV data.
        """
        all_data = []
        since = self.exchange.parse8601(since) if since else None
        pbar = tqdm(desc=f"Fetching {timeframe} data from starting time {since}", unit=" requests", position=0)

        while True:
            try:
                data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                if not data:
                    logging.info("No more data to fetch.")
                    break

                all_data.extend(data)
                since = data[-1][0] + 1  # Fetch data after the last timestamp

                # Update progress bar
                pbar.update(1)

                if len(data) < limit:
                    logging.info("Reached the end of available data.")
                    break

            except Exception as e:
                logging.error(f"Error fetching data: {str(e)}")
                break

        pbar.close()

        if all_data:
            logging.info(f"Successfully fetched {len(all_data)} rows of {timeframe} data for {symbol}.")
        else:
            logging.warning(f"No data fetched for {symbol} with timeframe {timeframe}.")

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = df[~df.index.duplicated(keep='first')]  # Remove duplicate indices
        return df

    @staticmethod
    def load_data(symbol, timeframe, input_dir='data'):
        """
        Load saved OHLCV data from CSV.

        Args:
            symbol (str): Symbol to load data for (e.g., 'BTC/USDT').
            timeframe (str): Timeframe to load data for (e.g., '1m', '5m').
            input_dir (str): Directory containing the CSV files.

        Returns:
            pd.DataFrame: DataFrame containing OHLCV data.
        """
        file_path = os.path.join(input_dir, f'{symbol.replace("/", "_")}_{timeframe}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist. Please fetch the data first.")
        logging.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path, index_col='datetime', parse_dates=['datetime'])

    

if __name__=="__main__":
    symbols = [
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

    timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
    for symbol in symbols:
        data_fetcher = DataFetcher()
        DATA_DIR = f'data/raw_data/240815/{symbol.split("/")[0].lower()}'
        data_fetcher.fetch_and_save_all_data(symbol, timeframes, since='2024-08-15T00:00:00Z', output_dir=DATA_DIR)

        for timeframe in timeframes:
            print(f"Loading {timeframe} data...")
            df = data_fetcher.load_data(symbol, timeframe, input_dir=DATA_DIR)
            print(df.head())