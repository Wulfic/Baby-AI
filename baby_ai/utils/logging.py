"""
Logging and monitoring for Baby-AI.

Provides a unified logger with file + console output,
and anomaly tracking for reward signals and safety events.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

from baby_ai.config import LOG_DIR, ensure_dirs


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if log_file:
        ensure_dirs()
        fh = logging.FileHandler(LOG_DIR / log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class RewardMonitor:
    """
    Tracks running statistics on reward channels.
    Triggers alerts when anomalies (sudden spikes/drops) are detected.
    """

    def __init__(self, window: int = 1000, alert_threshold: float = 5.0):
        self._window = window
        self._alert_threshold = alert_threshold
        self._history: dict[str, deque] = {}
        self._logger = get_logger("RewardMonitor", log_file="rewards.log")

    def record(self, channel: str, value: float) -> None:
        if channel not in self._history:
            self._history[channel] = deque(maxlen=self._window)
        buf = self._history[channel]
        buf.append(value)

        if len(buf) >= 50:
            import numpy as np

            arr = np.array(buf)
            mean, std = arr.mean(), arr.std() + 1e-8
            z = abs(value - mean) / std
            if z > self._alert_threshold:
                self._logger.warning(
                    "Anomaly in '%s': value=%.4f  z=%.2f  (mean=%.4f std=%.4f)",
                    channel, value, z, mean, std,
                )

    def summary(self) -> dict:
        import numpy as np
        out = {}
        for ch, buf in self._history.items():
            arr = np.array(buf)
            out[ch] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(buf)}
        return out


class LatencyTracker:
    """Simple context-manager timer for measuring inference latency."""

    def __init__(self, name: str = "op"):
        self.name = name
        self._start: float = 0
        self.last_ms: float = 0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.last_ms = (time.perf_counter() - self._start) * 1000
        return False

    def check(self, target_ms: float) -> bool:
        """Return True if last measurement was within target."""
        return self.last_ms <= target_ms
