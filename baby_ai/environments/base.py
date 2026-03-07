"""
Abstract base class for game/application environments.

All environments expose a step-based interface where:
1. The agent observes the current state (screen capture, etc.)
2. The agent selects an action (discrete index)
3. The environment executes the action and returns new observations

This decouples the learning system from any specific game or application.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import torch


class GameEnvironment(abc.ABC):
    """
    Abstract environment interface for Baby-AI.

    Subclasses implement environment-specific logic for capturing
    observations and executing actions. The interface mirrors
    OpenAI Gym semantics for familiarity.
    """

    @abc.abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """
        Reset the environment state and return initial observation.

        Returns:
            Observation dict with at least a "vision" key containing
            a (1, C, H, W) tensor suitable for the vision encoder.
        """
        ...

    @abc.abstractmethod
    def step(self, action_id: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Execute an action and return (observation, reward, done, info).

        Args:
            action_id: Index into the environment's discrete action space.

        Returns:
            observation: Dict with tensor inputs for the model.
            reward: Scalar extrinsic reward (0.0 if using intrinsic only).
            done: Whether the episode has ended.
            info: Auxiliary information (latency, action name, etc.).
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release all resources (input devices, capture handles, etc.)."""
        ...

    @property
    @abc.abstractmethod
    def action_space_size(self) -> int:
        """Number of discrete actions available."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable environment name."""
        ...

    def render(self) -> Optional[Any]:
        """Optional: return a renderable frame for visualization."""
        return None
