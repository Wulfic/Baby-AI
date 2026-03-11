"""
Live audio capture from the system audio output (WASAPI loopback).

Captures the mixed audio that includes Minecraft's game sounds and
converts it into a rolling mel-spectrogram buffer that the model can
consume each step.

Uses ``sounddevice`` with WASAPI loopback on Windows.  Falls back
gracefully to silent (zero) tensors if audio capture is unavailable
(missing device, Linux, permissions).

The capture runs on a background thread and maintains a circular
buffer of raw audio samples.  Each call to :meth:`get_mel_spectrogram`
returns the most recent ``context_sec`` seconds of audio as a
(1, n_mels, T) tensor ready for the AudioEncoder.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import torch

from baby_ai.utils.logging import get_logger

log = get_logger("audio_capture")

# Optional dependencies — graceful degradation if missing
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None  # type: ignore
    HAS_SOUNDDEVICE = False

try:
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
    T = None  # type: ignore
    HAS_TORCHAUDIO = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore
    HAS_LIBROSA = False


class AudioCapture:
    """
    Captures system audio via WASAPI loopback and produces mel spectrograms.

    Args:
        sample_rate: Target audio sample rate (Hz).
        n_mels: Number of mel frequency bins.
        hop_length: STFT hop length in samples.
        win_length: STFT window length in samples.
        context_sec: Seconds of audio context per observation.
        device_name: Substring to match in audio device names.
                     Defaults to None (= system default loopback).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        hop_length: int = 160,
        win_length: int = 400,
        context_sec: float = 1.0,
        device_name: Optional[str] = None,
    ):
        self._sample_rate = sample_rate
        self._n_mels = n_mels
        self._hop_length = hop_length
        self._win_length = win_length
        self._context_sec = context_sec
        self._device_name = device_name

        # Buffer holds raw mono float32 samples
        self._buffer_samples = int(sample_rate * context_sec * 2)  # 2x context for safety
        self._buffer = np.zeros(self._buffer_samples, dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()

        # Expected output shape: (n_mels, T_frames)
        self._context_samples = int(sample_rate * context_sec)
        self._expected_frames = self._context_samples // hop_length

        # Mel transform (torchaudio preferred, librosa fallback)
        self._mel_transform: Optional[object] = None
        if HAS_TORCHAUDIO:
            self._mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
            log.info("Using torchaudio MelSpectrogram for audio processing")

        self._stream: Optional[object] = None
        self._running = False
        self._available = False

        # Try to start capture
        self._init_capture()

    @property
    def available(self) -> bool:
        """True if audio capture is active and producing data."""
        return self._available

    def _init_capture(self) -> None:
        """Attempt to set up WASAPI loopback capture."""
        if not HAS_SOUNDDEVICE:
            log.warning(
                "sounddevice not installed — audio capture disabled. "
                "Install with: pip install sounddevice"
            )
            return

        if not (HAS_TORCHAUDIO or HAS_LIBROSA):
            log.warning(
                "Neither torchaudio nor librosa installed — audio capture disabled. "
                "Install with: pip install torchaudio"
            )
            return

        try:
            # Find a WASAPI loopback device
            device_id = self._find_loopback_device()
            if device_id is None:
                log.warning("No WASAPI loopback device found — audio capture disabled")
                return

            device_info = sd.query_devices(device_id)
            native_sr = int(device_info['default_samplerate'])
            max_channels = device_info['max_input_channels']
            log.info(
                "Audio device: '%s' (id=%d, sr=%d, ch=%d)",
                device_info['name'], device_id, native_sr, max_channels,
            )

            # We capture at native sample rate and resample later
            self._native_sr = native_sr
            self._resample_ratio = self._sample_rate / native_sr

            # Start the capture stream
            self._stream = sd.InputStream(
                device=device_id,
                samplerate=native_sr,
                channels=1,  # mono
                dtype='float32',
                blocksize=int(native_sr * 0.05),  # 50ms blocks
                callback=self._audio_callback,
                latency='low',
            )
            self._running = True
            self._stream.start()
            self._available = True
            log.info(
                "Audio capture started — %d Hz → %d Hz, %d mel bins, %.1fs context",
                native_sr, self._sample_rate, self._n_mels, self._context_sec,
            )
        except Exception as exc:
            log.warning("Failed to start audio capture: %s", exc)
            self._available = False

    def _find_loopback_device(self) -> Optional[int]:
        """Find a suitable WASAPI loopback device for system audio capture."""
        devices = sd.query_devices()

        # If user specified a device name, search for it
        if self._device_name:
            for i, d in enumerate(devices):
                if (self._device_name.lower() in d['name'].lower()
                        and d['max_input_channels'] > 0):
                    return i

        # Try to find a loopback device automatically
        # On Windows, WASAPI loopback devices contain "Loopback" or
        # are listed as input devices from output-capable hardware
        hostapis = sd.query_hostapis()
        wasapi_idx = None
        for i, api in enumerate(hostapis):
            if 'WASAPI' in api['name']:
                wasapi_idx = i
                break

        if wasapi_idx is None:
            # No WASAPI available — try default input device
            try:
                default = sd.default.device[0]  # default input
                if default is not None and default >= 0:
                    return int(default)
            except Exception:
                pass
            return None

        # Search WASAPI devices for a loopback-capable input
        for i, d in enumerate(devices):
            if d.get('hostapi') == wasapi_idx and d['max_input_channels'] > 0:
                name = d['name'].lower()
                # Prefer devices that are clearly loopback/stereo mix
                if 'loopback' in name or 'stereo mix' in name or 'what u hear' in name:
                    return i

        # Fall back to any WASAPI input device (speakers often expose
        # a loopback channel on Windows)
        for i, d in enumerate(devices):
            if d.get('hostapi') == wasapi_idx and d['max_input_channels'] > 0:
                return i

        return None

    def _audio_callback(self, indata: np.ndarray, frames: int,
                         time_info: object, status: object) -> None:
        """Called by sounddevice on each audio block (background thread)."""
        if status:
            log.debug("Audio callback status: %s", status)

        # indata is (frames, channels) — take first channel
        mono = indata[:, 0] if indata.ndim > 1 else indata

        # Simple resampling if native rate differs from target
        if abs(self._resample_ratio - 1.0) > 0.01:
            target_len = int(len(mono) * self._resample_ratio)
            if target_len > 0:
                indices = np.linspace(0, len(mono) - 1, target_len)
                mono = np.interp(indices, np.arange(len(mono)), mono).astype(np.float32)

        # Write to circular buffer
        with self._lock:
            n = len(mono)
            space = self._buffer_samples - self._write_pos
            if n <= space:
                self._buffer[self._write_pos:self._write_pos + n] = mono
                self._write_pos += n
            else:
                # Wrap around
                self._buffer[self._write_pos:] = mono[:space]
                remainder = n - space
                self._buffer[:remainder] = mono[space:]
                self._write_pos = remainder

    def get_mel_spectrogram(self) -> torch.Tensor:
        """
        Return the most recent audio context as a mel spectrogram tensor.

        Returns:
            (1, n_mels, T) float32 tensor.  Returns zeros if audio
            capture is not available.
        """
        if not self._available:
            return torch.zeros(1, self._n_mels, max(1, self._expected_frames))

        # Extract the most recent context_samples from the circular buffer
        with self._lock:
            n = self._context_samples
            if self._write_pos >= n:
                audio = self._buffer[self._write_pos - n:self._write_pos].copy()
            else:
                # Wrap: take from end + from beginning
                tail_len = n - self._write_pos
                audio = np.concatenate([
                    self._buffer[self._buffer_samples - tail_len:],
                    self._buffer[:self._write_pos],
                ])

        # Convert to mel spectrogram
        waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)

        if self._mel_transform is not None:
            # torchaudio path
            mel = self._mel_transform(waveform)  # (1, n_mels, T_frames)
            # Log scale
            mel = torch.log(mel.clamp(min=1e-10))
        elif HAS_LIBROSA:
            # librosa fallback
            mel_np = librosa.feature.melspectrogram(
                y=audio,
                sr=self._sample_rate,
                n_mels=self._n_mels,
                hop_length=self._hop_length,
                n_fft=self._win_length,
            )
            mel = torch.from_numpy(np.log(mel_np + 1e-10)).unsqueeze(0)
        else:
            return torch.zeros(1, self._n_mels, max(1, self._expected_frames))

        # Normalize to roughly zero-mean, unit-variance
        if mel.numel() > 0:
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        return mel.float()

    def stop(self) -> None:
        """Stop the audio capture stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._available = False
        log.info("Audio capture stopped")
