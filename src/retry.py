"""Простой retry-декоратор с exponential backoff и jitter.

Применяется к вызовам Binance API. Не-retryable исключения (валидация,
insufficient margin и т.п.) пробрасываются сразу.
"""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Tuple, Type

log = logging.getLogger(__name__)


def _is_retryable(exc: BaseException, non_retryable_codes: Tuple[str, ...]) -> bool:
    msg = str(exc)
    for code in non_retryable_codes:
        if code in msg:
            return False
    return True


def with_retry(
    max_attempts: int = 5,
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    retryable: Tuple[Type[BaseException], ...] = (Exception,),
    non_retryable_codes: Tuple[str, ...] = (
        # Hyperliquid: ошибки приходят как text в response payload.
        "insufficient margin",
        "insufficient balance",
        "Order has invalid",
        "User or API Wallet",
        "must be a multiple of",
        "below minimum size",
        "Reduce only order would increase position",
    ),
) -> Callable:
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable as exc:
                    if not _is_retryable(exc, non_retryable_codes):
                        raise
                    if attempt == max_attempts:
                        log.error(
                            "%s: giving up after %d attempts: %s",
                            fn.__name__, attempt, exc,
                        )
                        raise
                    sleep_for = min(max_delay, delay) * (0.7 + random.random() * 0.6)
                    log.warning(
                        "%s attempt %d/%d failed (%s); retrying in %.1fs",
                        fn.__name__, attempt, max_attempts, exc, sleep_for,
                    )
                    time.sleep(sleep_for)
                    delay = min(max_delay, delay * 2)
        return wrapper
    return deco
