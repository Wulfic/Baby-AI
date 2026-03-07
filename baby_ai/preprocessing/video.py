"""
Video preprocessing pipeline.

Raw video → sampled frames → resized → normalized tensors.
Runs on CPU to offload GPU for model computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import cv2


class VideoPreprocessor:
    """
    Converts raw video streams into compact tensor batches.

    Pipeline:
    1. Sample at target FPS (3-6 fps)
    2. Resize to target resolution (160x160)
    3. Normalize to [0, 1]
    4. Optionally convert to grayscale
    5. Stack into (T, C, H, W) tensor clips

    Args:
        target_fps: Frames per second to sample.
        resolution: (H, W) target resolution.
        grayscale: Convert to single-channel grayscale.
        clip_length_sec: Length of each clip in seconds.
        normalize: Normalize pixel values to [0, 1].
    """

    def __init__(
        self,
        target_fps: int = 4,
        resolution: tuple[int, int] = (160, 160),
        grayscale: bool = False,
        clip_length_sec: float = 1.0,
        normalize: bool = True,
    ):
        self.target_fps = target_fps
        self.resolution = resolution  # (H, W)
        self.grayscale = grayscale
        self.clip_length_sec = clip_length_sec
        self.normalize = normalize
        self.frames_per_clip = int(target_fps * clip_length_sec)

    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Process a single frame.

        Args:
            frame: (H, W, C) uint8 BGR numpy array from OpenCV.

        Returns:
            (C, H, W) float32 tensor.
        """
        # Resize
        frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]))

        # Color conversion
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame[..., np.newaxis]  # (H, W, 1)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To tensor: (H, W, C) → (C, H, W)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()

        if self.normalize:
            tensor = tensor / 255.0

        return tensor

    def process_video_file(self, path: Union[str, Path]) -> list[torch.Tensor]:
        """
        Process a video file into a list of clip tensors.

        Args:
            path: Path to video file.

        Returns:
            List of (T, C, H, W) tensor clips.
        """
        path = str(path)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = max(1, int(source_fps / self.target_fps))

        all_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                all_frames.append(self.process_frame(frame))
            frame_idx += 1

        cap.release()

        # Split into clips
        clips = []
        for i in range(0, len(all_frames), self.frames_per_clip):
            clip_frames = all_frames[i:i + self.frames_per_clip]
            if len(clip_frames) == self.frames_per_clip:
                clips.append(torch.stack(clip_frames))

        return clips

    def process_camera_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Process a single live camera frame.

        Args:
            frame: Raw BGR frame from camera.

        Returns:
            (1, C, H, W) tensor ready for the vision encoder.
        """
        return self.process_frame(frame).unsqueeze(0)

    def dummy_input(self, batch_size: int = 1) -> torch.Tensor:
        """Create a dummy input tensor for testing/profiling."""
        channels = 1 if self.grayscale else 3
        return torch.randn(
            batch_size, channels,
            self.resolution[0], self.resolution[1],
        )
