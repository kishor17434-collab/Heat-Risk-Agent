"""
rate_limit.py
─────────────
Provides a simple decorator to rate limit API calls.
"""

import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def rate_limit(min_interval_seconds: float):
    """
    Decorator to ensure that a function is not called more frequently
    than once per `min_interval_seconds`.
    """
    def decorator(func):
        last_called = 0.0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_called
            now = time.time()
            elapsed = now - last_called
            
            if elapsed < min_interval_seconds:
                sleep_time = min_interval_seconds - elapsed
                logger.debug("Rate limiting: sleeping for %.2fs before calling %s", sleep_time, func.__name__)
                time.sleep(sleep_time)
                
            result = func(*args, **kwargs)
            last_called = time.time()
            return result
            
        return wrapper
    return decorator
