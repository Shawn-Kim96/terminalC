import ccxt
import pandas as pd
import os
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import argparse
from pathlib import Path
from typing import Optional


# Set up logging
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


class DataFetcher:
    """
    Fetch historical OHLCV data for given symbol and timeframes from an exchange platform.
    """
    def __init__(self, exchange_name='binance'):
        self.exchange = getattr(ccxt, exchange_name)()

    def _ensure_datetime(self, value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, pd.Timestamp):
            if value.tz is None:
                return value.tz_localize('UTC').to_pydatetime()
            return value.tz_convert('UTC').to_pydatetime()
        if isinstance(value, (int, float)):
            # Assume milliseconds since epoch to stay consistent with CCXT
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        if isinstance(value, str):
            try:
                return pd.Timestamp(value, tz='UTC').to_pydatetime()
            except ValueError as exc:
                logging.error(f"Unable to parse datetime string '{value}': {exc}")
                raise
        raise TypeError(f"Unsupported datetime type: {type(value)}")

    def _to_milliseconds(self, value):
        dt = self._ensure_datetime(value)
        return int(dt.timestamp() * 1000) if dt else None

    @staticmethod
    def _end_of_previous_day(reference=None):
        reference = reference or datetime.now(timezone.utc)
        reference = reference.astimezone(timezone.utc)
        midnight_today = reference.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight_today - timedelta(seconds=1)

    @staticmethod
    def _end_of_day(dt):
        dt = dt.astimezone(timezone.utc)
        return dt.replace(hour=23, minute=59, second=59, microsecond=0)

    @staticmethod
    def _start_of_next_day(dt):
        dt = dt.astimezone(timezone.utc)
        next_day = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return next_day

    def _timeframe_delta(self, timeframe):
        seconds = self.exchange.parse_timeframe(timeframe)
        return pd.to_timedelta(seconds, unit='s')

    @staticmethod
    def _ensure_utc_index(df):
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        return df

    def _load_existing_dataset(self, file_path):
        if not os.path.exists(file_path):
            return None, None
        existing_df = pd.read_csv(file_path, index_col='datetime', parse_dates=['datetime'])
        existing_df.index.name = 'datetime'
        existing_df = self._ensure_utc_index(existing_df)
        last_timestamp = existing_df.index.max()
        return existing_df, last_timestamp

    def _log_missing_intervals(self, df, timeframe, symbol, context):
        if df is None or df.empty or len(df.index) < 2:
            return
        freq = self._timeframe_delta(timeframe)
        tolerance = pd.Timedelta(seconds=1)
        diffs = df.index.to_series().diff().dropna()
        gaps = diffs[diffs > freq + tolerance]
        for idx, delta in gaps.items():
            missing = int(delta / freq) - 1
            gap_start = idx - delta + freq
            gap_end = idx - freq
            logging.warning(
                f"[{symbol} {timeframe}] Missing {missing} candles between {gap_start} and {gap_end} during {context}."
            )

    def fetch_and_save_all_data(self, symbol, timeframes, output_dir='data', since=None, until=None):
        """
        Fetch OHLCV data day-by-day and append it to disk without overlaps.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
            timeframes (list[str]): List of timeframes to fetch.
            output_dir (str): Directory where CSV files live.
            since (str | datetime): Starting datetime (inclusive).
            until (str | datetime | None): Optional end datetime. Defaults to the end of the previous UTC day.
        """
        if since is None:
            raise ValueError("A starting datetime ('since') must be provided.")

        os.makedirs(output_dir, exist_ok=True)
        start_dt = self._ensure_datetime(since)
        target_end = self._ensure_datetime(until) if until else self._end_of_previous_day()

        if start_dt > target_end:
            logging.info(f"Start time {start_dt} is after target end {target_end}. Nothing to fetch.")
            return

        for timeframe in timeframes:
            timeframe_delta = self._timeframe_delta(timeframe)
            file_path = os.path.join(output_dir, f'{symbol.replace("/", "_")}_{timeframe}.csv')
            existing_df, last_existing_ts = self._load_existing_dataset(file_path)

            effective_start = start_dt
            if last_existing_ts is not None:
                next_expected = (last_existing_ts + timeframe_delta).to_pydatetime()
                if next_expected > effective_start:
                    effective_start = next_expected

            if effective_start > target_end:
                logging.info(f"[{symbol} {timeframe}] Up-to-date through {last_existing_ts}.")
                continue

            logging.info(
                f"[{symbol} {timeframe}] Fetching from {effective_start} to {target_end} (daily chunks)."
            )
            chunk_frames = []
            day_cursor = effective_start

            while day_cursor <= target_end:
                day_end = min(self._end_of_day(day_cursor), target_end)
                chunk_df = self.fetch_ohlcv(symbol, timeframe, since=day_cursor, until=day_end)
                if not chunk_df.empty:
                    chunk_frames.append(chunk_df)
                else:
                    logging.info(f"[{symbol} {timeframe}] No data returned for {day_cursor.date()}.")
                day_cursor = self._start_of_next_day(day_cursor)

            if not chunk_frames:
                logging.info(f"[{symbol} {timeframe}] No new data fetched.")
                continue

            new_df = pd.concat(chunk_frames)
            new_df.index.name = 'datetime'
            new_df = self._ensure_utc_index(new_df)
            new_df = new_df[~new_df.index.duplicated(keep='first')]
            new_df.sort_index(inplace=True)

            if last_existing_ts is not None:
                overlap_mask = new_df.index <= last_existing_ts
                overlap_count = int(overlap_mask.sum())
                if overlap_count:
                    logging.info(
                        f"[{symbol} {timeframe}] Dropping {overlap_count} overlapping candles up to {last_existing_ts}."
                    )
                new_df = new_df[~overlap_mask]
                if not new_df.empty:
                    gap = new_df.index.min() - last_existing_ts
                    if gap > timeframe_delta + pd.Timedelta(seconds=1):
                        missing = int(gap / timeframe_delta) - 1
                        logging.warning(
                            f"[{symbol} {timeframe}] Missing {missing} candles between {last_existing_ts} and {new_df.index.min()}."
                        )

            if new_df.empty:
                logging.info(f"[{symbol} {timeframe}] Nothing new after removing overlaps.")
                continue

            if existing_df is not None:
                combined_df = pd.concat([existing_df, new_df])
            else:
                combined_df = new_df

            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df.sort_index(inplace=True)
            self._log_missing_intervals(combined_df, timeframe, symbol, context='merge')

            output_df = combined_df.copy()
            if output_df.index.tz is not None:
                output_df.index = output_df.index.tz_convert('UTC').tz_localize(None)
            output_df.index.name = 'datetime'

            output_df.to_csv(file_path)
            logging.info(f"[{symbol} {timeframe}] Saved {len(new_df)} new rows to {file_path}.")

    def fetch_ohlcv(self, symbol, timeframe, since=None, until=None, limit=1000):
        """
        Fetch historical OHLCV data for the specified symbol/timeframe bounded by [since, until].

        Args:
            symbol (str): Symbol to fetch data for (e.g., 'BTC/USDT').
            timeframe (str): Timeframe to fetch (e.g., '1m', '5m').
            since (datetime | str): Starting datetime (inclusive).
            until (datetime | str | None): Ending datetime (inclusive).
            limit (int): Maximum number of candles to fetch in each request.

        Returns:
            pd.DataFrame: DataFrame containing OHLCV data.
        """
        all_data = []
        since_display = self._ensure_datetime(since)
        until_display = self._ensure_datetime(until) if until is not None else None
        since_ms = self._to_milliseconds(since)
        until_ms = self._to_milliseconds(until)
        timeframe_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
        desc = f"{symbol} {timeframe} from {since_display} to {until_display or 'prev-day-end'}"
        pbar = tqdm(desc=f"Fetching {desc}", unit="request", position=0, leave=False)

        try:
            while True:
                if since_ms is not None and until_ms is not None and since_ms > until_ms:
                    break

                data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
                if not data:
                    logging.info(f"[{symbol} {timeframe}] No more data returned by exchange.")
                    break

                pbar.update(1)

                if until_ms is not None:
                    trimmed = [row for row in data if row[0] <= until_ms]
                else:
                    trimmed = data

                all_data.extend(trimmed)

                if until_ms is not None and trimmed and trimmed[-1][0] >= until_ms:
                    break

                if len(data) < limit:
                    logging.info(f"[{symbol} {timeframe}] Received partial batch; stopping.")
                    break

                since_ms = data[-1][0] + timeframe_ms
        except Exception as exc:
            logging.error(f"Error fetching {symbol} {timeframe}: {exc}")
        finally:
            pbar.close()

        if not all_data:
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            empty_df = pd.DataFrame(columns=columns)
            empty_df.index = pd.DatetimeIndex([], name='datetime', tz='UTC')
            return empty_df

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
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


def _find_latest_dataset_dir(base_path: Path) -> Optional[Path]:
    base_path = Path(base_path)
    if not base_path.exists():
        return None
    candidates = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('candle')],
        key=lambda path: path.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _infer_start_time_from_data(dataset_dir: Path, symbols, timeframes) -> Optional[datetime]:
    preferred_order = ['1d'] + [tf for tf in timeframes if tf != '1d']
    timestamps = []
    for timeframe in preferred_order:
        if timeframe not in timeframes:
            continue
        for symbol in symbols:
            symbol_folder = dataset_dir / symbol.split("/")[0].lower()
            file_path = symbol_folder / f'{symbol.replace("/", "_")}_{timeframe}.csv'
            if not file_path.exists():
                continue
            try:
                history = pd.read_csv(file_path, usecols=['datetime'])
            except Exception as exc:
                logging.warning(f"Failed to inspect {file_path}: {exc}")
                continue
            if history.empty or 'datetime' not in history.columns:
                continue
            ts = pd.to_datetime(history['datetime'], utc=True).max()
            if pd.isna(ts):
                continue
            timestamps.append(ts)
        if timestamps:
            break

    if not timestamps:
        return None

    min_last = min(timestamps)
    next_day = (min_last.floor('D') + pd.Timedelta(days=1)).to_pydatetime()
    return next_day.astimezone(timezone.utc)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch market data from exchanges.")
    parser.add_argument("--start-date", dest="start_date", type=str,
                        help="Override the inferred UTC start date (ISO 8601).")
    parser.add_argument("--dataset-root", dest="dataset_root", type=str,
                        help="Relative or absolute path to an existing candle dataset directory.")
    parser.add_argument("--exchange", dest="exchange", type=str, default="binance",
                        help="Exchange name supported by ccxt (default: binance).")
    args = parser.parse_args()

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

    base_data_path = Path("data/raw_data")
    dataset_root: Optional[Path]
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        if not dataset_root.is_absolute():
            dataset_root = Path.cwd() / dataset_root
    else:
        dataset_root = _find_latest_dataset_dir(base_data_path)

    manual_start: Optional[datetime] = None
    if args.start_date:
        try:
            manual_start = pd.Timestamp(args.start_date, tz='UTC').to_pydatetime()
        except ValueError as exc:
            parser.error(f"Invalid --start-date value '{args.start_date}': {exc}")

    if dataset_root is None:
        if manual_start is None:
            parser.error("No dataset directory found. Provide --start-date to bootstrap a new dataset.")
        date_token = manual_start.strftime('%y%m%d')
        dataset_root = base_data_path / f'candle'
    if not dataset_root.exists():
        dataset_root.mkdir(parents=True, exist_ok=True)

    if manual_start:
        start_time = manual_start
    else:
        inferred_start = _infer_start_time_from_data(dataset_root, symbols, timeframes)
        if inferred_start is None:
            parser.error(f"Unable to infer a start date from existing data under {dataset_root}. "
                         "Specify --start-date explicitly.")
        start_time = inferred_start

    logging.info(f"Next fetch window starts at {start_time} UTC.")

    data_fetcher = DataFetcher(exchange_name=args.exchange)
    for symbol in symbols:
        symbol_dir = dataset_root / symbol.split("/")[0].lower()
        data_fetcher.fetch_and_save_all_data(symbol, timeframes, since=start_time, output_dir=str(symbol_dir))

        for timeframe in timeframes:
            print(f"Loading {timeframe} data...")
            df = data_fetcher.load_data(symbol, timeframe, input_dir=str(symbol_dir))
            print(df.head())
