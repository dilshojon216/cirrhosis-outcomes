"""Umumiy yordamchi funksiyalar: seed, logger, timer."""

from __future__ import annotations

import functools
import logging
import os
import random
import sys
import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np

F = TypeVar("F", bound=Callable)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def seed_everything(seed: int = 42) -> None:
    """Python, NumPy va hash seed'ini qat'iylashtiradi.

    Model kutubxonalari (LightGBM, XGBoost, CatBoost) seed'ni o'z
    parametrlari orqali oladi — ularga ``random_state=seed`` bering.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "cirrhosis", level: int = logging.INFO) -> logging.Logger:
    """Stdout'ga yozadigan logger. Qayta chaqirilganda handler dublikat bo'lmaydi."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def timer(func: F) -> F:
    """Funksiya bajarilish vaqtini log qiladigan dekorator."""
    logger = get_logger()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info("%s: %.2fs", func.__qualname__, time.perf_counter() - start)

    return wrapper  # type: ignore[return-value]
