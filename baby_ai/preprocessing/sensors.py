"""
Sensor preprocessing pipeline.

Normalizes and timestamps raw sensor streams into
fixed-rate frames suitable for the sensor encoder.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch


class SensorPreprocessor:
    """
    Normalizes multi-channel sensor data into fixed-rate frames.

    Maintains running statistics for online normalization
    and interpolates irregular samples to a fixed frame rate.

    Args:
        max_channels: Maximum number of sensor channels.
        frame_rate: Target fixed frame rate (Hz).
        normalize: Apply online normalization.
        window_size: Window for running stats computation.
    """

    def __init__(
        self,
        max_channels: int = 16,
        frame_rate: int = 30,
        normalize: bool = True,
        window_size: int = 1000,
    ):
        self.max_channels = max_channels
        self.frame_rate = frame_rate
        self.normalize = normalize

        # Running stats
        self._running_mean = np.zeros(max_channels, dtype=np.float32)
        self._running_var = np.ones(max_channels, dtype=np.float32)
        self._count = 0
        self._alpha = 2.0 / (window_size + 1)

    def update_stats(self, values: np.ndarray) -> None:
        """Update running mean and variance with a new sample."""
        if self._count == 0:
            self._running_mean[:len(values)] = values
        else:
            delta = values - self._running_mean[:len(values)]
            self._running_mean[:len(values)] += self._alpha * delta
            self._running_var[:len(values)] = (
                (1 - self._alpha) * self._running_var[:len(values)]
                + self._alpha * delta ** 2
            )
        self._count += 1

    def process(self, values: np.ndarray) -> torch.Tensor:
        """
        Process a single sensor reading.

        Args:
            values: (C,) raw sensor values, C <= max_channels.

        Returns:
            (max_channels,) normalized tensor.
        """
        # Pad to max_channels
        padded = np.zeros(self.max_channels, dtype=np.float32)
        n = min(len(values), self.max_channels)
        padded[:n] = values[:n]

        if self.normalize:
            self.update_stats(padded[:n])
            std = np.sqrt(self._running_var[:n] + 1e-8)
            padded[:n] = (padded[:n] - self._running_mean[:n]) / std

        return torch.from_numpy(padded)

    def process_batch(self, batch: np.ndarray) -> torch.Tensor:
        """
        Process a batch of sensor readings.

        Args:
            batch: (B, C) raw sensor values.

        Returns:
            (B, max_channels) normalized tensor.
        """
        result = []
        for i in range(batch.shape[0]):
            result.append(self.process(batch[i]))
        return torch.stack(result)

    def dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        """Create dummy sensor input for testing."""
        return torch.randn(batch_size, self.max_channels)
