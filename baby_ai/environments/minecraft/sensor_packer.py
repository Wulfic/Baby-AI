"""
Game-state sensor packer for Minecraft.

Converts structured data from the Fabric mod bridge (position_update
and player_status events) into a normalized 32-channel float tensor
for the sensor encoder.

Channel layout (32 channels):
    [ 0] health           — normalised 0–1 (0=dead, 1=full 20hp)
    [ 1] food             — normalised 0–1 (0=starving, 1=full)
    [ 2] saturation       — normalised 0–1
    [ 3] armor            — normalised 0–1 (0=none, 1=full diamond)
    [ 4] air              — normalised 0–1 (1=full breath, 0=drowning)
    [ 5] xp_level         — log-scaled (log(1+level) / 5)
    [ 6] xp_progress      — raw 0–1
    [ 7] y_height         — normalised: (y - 0) / 320 (world height 0–320)
    [ 8] pitch            — normalised: pitch / 90 → [-1, 1]
    [ 9] yaw_sin          — sin(yaw_rad) for circular encoding
    [10] yaw_cos          — cos(yaw_rad) for circular encoding
    [11] on_ground        — 0 or 1
    [12] sky_light        — normalised 0–1 (0=dark, 1=full daylight)
    [13] velocity_x       — clipped to [-1, 1]
    [14] velocity_y       — clipped to [-1, 1] (falling = negative)
    [15] velocity_z       — clipped to [-1, 1]
    [16] is_sprinting     — 0 or 1
    [17] is_swimming      — 0 or 1
    [18] is_sneaking      — 0 or 1
    [19] is_on_fire       — 0 or 1
    [20] day_phase_sin    — sin(2π * day_time / 24000) for smooth day/night
    [21] day_phase_cos    — cos(2π * day_time / 24000)
    [22] is_raining       — 0 or 1
    [23] is_thundering    — 0 or 1
    [24] inventory_fill   — normalised 0–1 (used_slots / 36)
    [25] damage_flash     — 1.0 for 5 steps after taking damage, else 0
    [26] food_delta       — recent food change (clipped [-1, 1])
    [27] blocks_broken    — recent count (clipped, decayed)
    [28] items_picked     — recent count (clipped, decayed)
    [29] items_crafted    — recent count (clipped, decayed)
    [30] distance_from_home — normalised (dist / 200, clipped 0–1)
    [31] speed            — |velocity| normalised 0–1

Provides structured, normalised game state that the neural network
can learn temporal patterns from, complementing the raw vision stream.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from baby_ai.utils.logging import get_logger

log = get_logger("sensor_pack")

NUM_SENSOR_CHANNELS = 32


class SensorPacker:
    """
    Maintains a live snapshot of structured game state from mod events
    and packs it into a normalised 32-dim tensor each step.

    Call :meth:`update` each step with the latest mod events, then
    :meth:`pack` to get the tensor.
    """

    def __init__(self):
        # ── Latest status snapshot ──────────────────────────────
        self._health: float = 20.0
        self._max_health: float = 20.0
        self._food: int = 20
        self._saturation: float = 5.0
        self._armor: int = 0
        self._xp_level: int = 0
        self._xp_progress: float = 0.0
        self._air: int = 300
        self._max_air: int = 300
        self._is_sprinting: bool = False
        self._is_swimming: bool = False
        self._is_sneaking: bool = False
        self._is_on_fire: bool = False
        self._game_time: int = 0
        self._day_time: int = 0
        self._is_raining: bool = False
        self._is_thundering: bool = False
        self._inventory_used: int = 0
        self._velocity_x: float = 0.0
        self._velocity_y: float = 0.0
        self._velocity_z: float = 0.0

        # ── Position data (from position_update) ────────────────
        self._y: float = 64.0
        self._pitch: float = 0.0
        self._yaw: float = 0.0
        self._on_ground: bool = True
        self._sky_light: int = 15

        # ── Home reference ──────────────────────────────────────
        self._home_x: Optional[float] = None
        self._home_z: Optional[float] = None
        self._player_x: float = 0.0
        self._player_z: float = 0.0

        # ── Event-driven decaying counters ──────────────────────
        self._damage_flash: int = 0        # steps remaining
        self._food_delta: float = 0.0      # recent food change
        self._recent_blocks_broken: float = 0.0
        self._recent_items_picked: float = 0.0
        self._recent_items_crafted: float = 0.0

        # Decay rate per step for event counters
        self._decay = 0.9

        self._has_status: bool = False  # True after first player_status event

    def update(
        self,
        mod_events: List[Dict[str, Any]],
        home_x: Optional[float] = None,
        home_z: Optional[float] = None,
    ) -> None:
        """
        Process mod events from the current step and update internal state.

        Args:
            mod_events: List of event dicts from ModBridge.drain_events().
            home_x: Current home X coordinate (from env).
            home_z: Current home Z coordinate (from env).
        """
        self._home_x = home_x
        self._home_z = home_z

        # Decay event counters each step
        self._recent_blocks_broken *= self._decay
        self._recent_items_picked *= self._decay
        self._recent_items_crafted *= self._decay
        self._food_delta *= self._decay
        if self._damage_flash > 0:
            self._damage_flash -= 1

        for ev in mod_events:
            evt = ev.get("event", "")

            if evt == "player_status":
                self._has_status = True
                self._health = ev.get("health", self._health)
                self._max_health = ev.get("max_health", self._max_health)
                self._food = ev.get("food", self._food)
                self._saturation = ev.get("saturation", self._saturation)
                self._armor = ev.get("armor", self._armor)
                self._xp_level = ev.get("xp_level", self._xp_level)
                self._xp_progress = ev.get("xp_progress", self._xp_progress)
                self._air = ev.get("air", self._air)
                self._max_air = ev.get("max_air", self._max_air)
                self._is_sprinting = ev.get("is_sprinting", self._is_sprinting)
                self._is_swimming = ev.get("is_swimming", self._is_swimming)
                self._is_sneaking = ev.get("is_sneaking", self._is_sneaking)
                self._is_on_fire = ev.get("is_on_fire", self._is_on_fire)
                self._game_time = ev.get("game_time", self._game_time)
                self._day_time = ev.get("day_time", self._day_time)
                self._is_raining = ev.get("is_raining", self._is_raining)
                self._is_thundering = ev.get("is_thundering", self._is_thundering)
                self._inventory_used = ev.get("inventory_used_slots", self._inventory_used)
                self._velocity_x = ev.get("velocity_x", self._velocity_x)
                self._velocity_y = ev.get("velocity_y", self._velocity_y)
                self._velocity_z = ev.get("velocity_z", self._velocity_z)

            elif evt == "position_update":
                self._y = ev.get("y", self._y)
                self._pitch = ev.get("pitch", self._pitch)
                self._yaw = ev.get("yaw", self._yaw)
                self._on_ground = ev.get("on_ground", self._on_ground)
                self._sky_light = ev.get("light", self._sky_light)
                self._player_x = ev.get("x", self._player_x)
                self._player_z = ev.get("z", self._player_z)

            elif evt == "health_changed":
                delta = ev.get("delta", 0.0)
                if delta < 0:
                    self._damage_flash = 5  # flash for 5 steps

            elif evt == "food_changed":
                self._food_delta += ev.get("delta", 0.0)

            elif evt == "block_broken":
                self._recent_blocks_broken += 1.0

            elif evt == "item_picked_up":
                self._recent_items_picked += ev.get("count", 1)

            elif evt == "item_crafted":
                self._recent_items_crafted += ev.get("count", 1)

    def pack(self) -> torch.Tensor:
        """
        Pack current state into a normalised (1, 32) float tensor.

        All channels are scaled to roughly [0, 1] or [-1, 1] ranges
        to make learning easier.  The sensor encoder's running
        normalisation further adapts to the actual data distribution.
        """
        s = np.zeros(NUM_SENSOR_CHANNELS, dtype=np.float32)

        # Vitals
        max_hp = max(self._max_health, 1.0)
        s[0] = self._health / max_hp
        s[1] = self._food / 20.0
        s[2] = self._saturation / 20.0
        s[3] = self._armor / 20.0
        max_air = max(self._max_air, 1)
        s[4] = self._air / max_air

        # XP (log-scaled since levels grow exponentially)
        s[5] = math.log(1.0 + self._xp_level) / 5.0
        s[6] = self._xp_progress

        # Spatial
        s[7] = np.clip(self._y / 320.0, 0.0, 1.0)
        s[8] = self._pitch / 90.0  # [-1, 1]

        # Yaw circular encoding (avoids discontinuity at ±180°)
        yaw_rad = math.radians(self._yaw)
        s[9] = math.sin(yaw_rad)
        s[10] = math.cos(yaw_rad)

        s[11] = 1.0 if self._on_ground else 0.0
        s[12] = self._sky_light / 15.0

        # Velocity (clipped, typical MC walking speed ~0.1 blocks/tick)
        s[13] = np.clip(self._velocity_x * 5.0, -1.0, 1.0)
        s[14] = np.clip(self._velocity_y * 5.0, -1.0, 1.0)
        s[15] = np.clip(self._velocity_z * 5.0, -1.0, 1.0)

        # Movement flags
        s[16] = 1.0 if self._is_sprinting else 0.0
        s[17] = 1.0 if self._is_swimming else 0.0
        s[18] = 1.0 if self._is_sneaking else 0.0
        s[19] = 1.0 if self._is_on_fire else 0.0

        # Day/night cycle (smooth circular)
        day_phase = 2.0 * math.pi * self._day_time / 24000.0
        s[20] = math.sin(day_phase)
        s[21] = math.cos(day_phase)

        # Weather
        s[22] = 1.0 if self._is_raining else 0.0
        s[23] = 1.0 if self._is_thundering else 0.0

        # Inventory fullness
        s[24] = self._inventory_used / 36.0

        # Event-driven signals
        s[25] = 1.0 if self._damage_flash > 0 else 0.0
        s[26] = np.clip(self._food_delta / 5.0, -1.0, 1.0)
        s[27] = np.clip(self._recent_blocks_broken / 3.0, 0.0, 1.0)
        s[28] = np.clip(self._recent_items_picked / 5.0, 0.0, 1.0)
        s[29] = np.clip(self._recent_items_crafted / 3.0, 0.0, 1.0)

        # Distance from home (normalised)
        if self._home_x is not None and self._home_z is not None:
            dx = self._player_x - self._home_x
            dz = self._player_z - self._home_z
            dist = math.sqrt(dx * dx + dz * dz)
            s[30] = np.clip(dist / 200.0, 0.0, 1.0)
        else:
            s[30] = 0.5  # neutral when home unknown

        # Speed magnitude
        speed = math.sqrt(
            self._velocity_x ** 2 +
            self._velocity_y ** 2 +
            self._velocity_z ** 2
        )
        s[31] = np.clip(speed * 5.0, 0.0, 1.0)

        return torch.from_numpy(s).unsqueeze(0)  # (1, 32)

    @property
    def has_data(self) -> bool:
        """True after at least one player_status event has been received."""
        return self._has_status
