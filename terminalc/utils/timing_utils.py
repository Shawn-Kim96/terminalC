# timing_utils.py
import time
import logging
import functools
from contextlib import contextmanager

def _get_logger(logger):
    if logger is None:
        logger = logging.getLogger("timer")
        if not logger.handlers:
            h = logging.StreamHandler()
            f = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
            h.setFormatter(f)
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
    return logger

def timeit(name: str = None, logger=None, level=logging.INFO):
    """
    usage: @timeit("process_file") 혹은 @timeit()
    """
    logger = _get_logger(logger)

    def decorator(func):
        label = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                logger.log(level, f"{label} finished in {elapsed:.3f}s")
        return wrapper
    return decorator

@contextmanager
def timer(name: str, logger=None, level=logging.INFO):
    """
    usage: with timer("load file"): ...
    """
    logger = _get_logger(logger)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.log(level, f"{name} finished in {elapsed:.3f}s")
