"""
Audio preprocessing pipeline.

Raw audio → mono → log-mel spectrogram → normalized tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class AudioPreprocessor:
    """
    Converts raw audio into log-mel spectrograms.

    Pipeline:
    1. Load/resample to target sample rate
    2. Convert to mono
    3. Compute log-mel spectrogram
    4. Normalize
    5. Slice into context windows

    Args:
        sample_rate: Target sample rate (Hz).
        n_mels: Number of mel frequency bins.
        hop_length: Hop length for STFT.
        win_length: Window length for STFT.
        context_sec: Context window duration in seconds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        hop_length: int = 160,
        win_length: int = 400,
        context_sec: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.context_sec = context_sec

        # Number of spectrogram frames per context window
        samples_per_window = int(sample_rate * context_sec)
        self.frames_per_window = samples_per_window // hop_length

        # Use torchaudio MelSpectrogram if available (GPU-friendly)
        if HAS_TORCHAUDIO:
            self._mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
        else:
            self._mel_transform = None

    def process_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process a raw waveform tensor.

        Args:
            waveform: (1, T) or (T,) mono audio tensor.

        Returns:
            (n_mels, T_frames) log-mel spectrogram tensor.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Mono downmix
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self._mel_transform is not None:
            mel = self._mel_transform(waveform)  # (1, n_mels, T)
            mel = mel.squeeze(0)
        else:
            # Fallback to librosa
            if not HAS_LIBROSA:
                raise ImportError("Either torchaudio or librosa required for audio processing.")
            wav_np = waveform.squeeze().numpy()
            mel_np = librosa.feature.melspectrogram(
                y=wav_np, sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                win_length=self.win_length,
            )
            mel = torch.from_numpy(mel_np).float()

        # Log scale (add small epsilon to avoid log(0))
        log_mel = torch.log(mel + 1e-9)

        # Normalize to zero mean, unit variance
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        return log_mel

    def process_file(self, path: Union[str, Path]) -> list[torch.Tensor]:
        """
        Load an audio file and produce context-window spectrograms.

        Returns:
            List of (n_mels, T_frames) tensors, one per window.
        """
        if HAS_TORCHAUDIO:
            waveform, sr = torchaudio.load(str(path))
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
        elif HAS_LIBROSA:
            wav_np, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
            waveform = torch.from_numpy(wav_np).unsqueeze(0)
        else:
            raise ImportError("torchaudio or librosa required.")

        log_mel = self.process_waveform(waveform)  # (n_mels, total_frames)

        # Slice into context windows
        windows = []
        for i in range(0, log_mel.shape[1], self.frames_per_window):
            window = log_mel[:, i:i + self.frames_per_window]
            if window.shape[1] == self.frames_per_window:
                windows.append(window)

        return windows

    def process_chunk(self, audio_data: np.ndarray) -> torch.Tensor:
        """
        Process a raw audio chunk (e.g., from microphone).

        Args:
            audio_data: (T,) float32 numpy array.

        Returns:
            (n_mels, T_frames) log-mel spectrogram.
        """
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        return self.process_waveform(waveform)

    def dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        """Create a dummy log-mel tensor for testing."""
        return torch.randn(batch_size, self.n_mels, self.frames_per_window)
