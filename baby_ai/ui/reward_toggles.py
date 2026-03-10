"""
Thread-safe reward channel toggle state.

Houses the shared state that the tkinter control panel writes to
(UI thread) and the main training loop reads from (main thread).
All access is guarded by a threading.Lock so there are no races.

Channel groups and training-phase presets make it easy to ramp up
reward complexity as the agent's skills develop, reducing the
conflicting-gradient problem that causes training instability.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Channel definitions ─────────────────────────────────────────
# Each channel has a display name, its key in the reward_breakdown
# dict, the group it belongs to, and whether it's enabled by default.

@dataclass(frozen=True)
class ChannelInfo:
    key: str            # key in reward_breakdown / compose() kwarg
    label: str          # human-readable name for UI
    group: str          # group header in the control panel
    default: bool       # enabled on first launch?
    is_penalty: bool = False  # swaps sign in compose()


# Ordered list — UI renders in this order.
CHANNELS: List[ChannelInfo] = [
    # ── Intrinsic ───────────────────────────────────────────────
    ChannelInfo("intrinsic",          "Intrinsic (curiosity)", "Intrinsic",    True),

    # ── Exploration ─────────────────────────────────────────────
    ChannelInfo("visual_change",      "Visual Change",       "Exploration",   True),
    ChannelInfo("exploration",        "Exploration",          "Exploration",   True),
    ChannelInfo("action_diversity",   "Action Diversity",     "Exploration",   True),
    ChannelInfo("movement",           "Movement",             "Exploration",   True),
    ChannelInfo("new_chunk",          "New Chunk",            "Exploration",   True),

    # ── Interaction ─────────────────────────────────────────────
    ChannelInfo("interaction",        "Interaction",          "Interaction",   False),
    ChannelInfo("block_break",        "Block Break",          "Interaction",   False),
    ChannelInfo("item_pickup",        "Item Pickup",          "Interaction",   False),
    ChannelInfo("item_drop_penalty",  "Item Drop Penalty",    "Interaction",   False, is_penalty=True),

    # ── Creation ────────────────────────────────────────────────
    ChannelInfo("block_place",        "Block Place",          "Creation",      False),
    ChannelInfo("crafting",           "Crafting",             "Creation",      False),
    ChannelInfo("building_streak",    "Building Streak",      "Creation",      False),
    ChannelInfo("creative_sequence",  "Creative Sequence",    "Creation",      False),

    # ── Survival ────────────────────────────────────────────────
    ChannelInfo("damage_taken",       "Damage Taken",         "Survival",      False, is_penalty=True),
    ChannelInfo("healing",            "Healing",              "Survival",      False),
    ChannelInfo("food_reward",        "Food Reward",          "Survival",      False),
    ChannelInfo("xp_reward",          "XP Reward",            "Survival",      False),

    # ── Penalties / Shaping ─────────────────────────────────────
    ChannelInfo("death_penalty",      "Death Penalty",        "Penalties",     False, is_penalty=True),
    ChannelInfo("idle_penalty",       "Idle Penalty",         "Penalties",     True,  is_penalty=True),
    ChannelInfo("stagnation_penalty", "Stagnation Penalty",   "Penalties",     False, is_penalty=True),
    ChannelInfo("hotbar_spam_penalty","Hotbar Spam Penalty",  "Penalties",     True,  is_penalty=True),
    ChannelInfo("height_penalty",     "Height Penalty",       "Penalties",     False, is_penalty=True),
    ChannelInfo("pitch_penalty",      "Pitch Penalty",        "Penalties",     True,  is_penalty=True),
    ChannelInfo("home_proximity",     "Home Proximity",       "Penalties",     False),
]

CHANNEL_MAP: Dict[str, ChannelInfo] = {ch.key: ch for ch in CHANNELS}

# Unique group names, preserving order of first appearance.
GROUPS: List[str] = list(dict.fromkeys(ch.group for ch in CHANNELS))


# ── Training-phase presets ──────────────────────────────────────
# Each preset specifies which *groups* are enabled.  Individual
# channel overrides are preserved when switching presets.

PHASE_PRESETS: Dict[str, Tuple[str, List[str]]] = {
    "phase_1": (
        "Phase 1 — Explore",
        ["Intrinsic", "Exploration", "Penalties"],
    ),
    "phase_2": (
        "Phase 2 — Interact",
        ["Intrinsic", "Exploration", "Interaction", "Penalties"],
    ),
    "phase_3": (
        "Phase 3 — Create",
        ["Intrinsic", "Exploration", "Interaction", "Creation", "Survival", "Penalties"],
    ),
    "phase_4": (
        "Phase 4 — All",
        GROUPS,  # everything
    ),
}


class RewardToggleState:
    """
    Thread-safe container for reward channel enable/disable state.

    The tkinter UI calls :meth:`set_enabled` and :meth:`apply_preset`;
    the training loop calls :meth:`filter_channels` every step.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Start with each channel's default state.
        self._enabled: Dict[str, bool] = {
            ch.key: ch.default for ch in CHANNELS
        }
        self._active_preset: str = "phase_1"
        # Apply phase 1 defaults on construction.
        self._apply_preset_unlocked("phase_1")

    # ── Mutators (called from UI thread) ────────────────────────

    def set_enabled(self, key: str, enabled: bool) -> None:
        """Toggle a single channel on/off."""
        with self._lock:
            if key in self._enabled:
                self._enabled[key] = enabled
                self._active_preset = "custom"

    def apply_preset(self, preset_id: str) -> None:
        """Activate a training-phase preset."""
        with self._lock:
            self._apply_preset_unlocked(preset_id)

    def _apply_preset_unlocked(self, preset_id: str) -> None:
        if preset_id not in PHASE_PRESETS:
            return
        _, enabled_groups = PHASE_PRESETS[preset_id]
        for ch in CHANNELS:
            self._enabled[ch.key] = ch.group in enabled_groups
        self._active_preset = preset_id

    # ── Readers (called from training thread) ───────────────────

    def is_enabled(self, key: str) -> bool:
        with self._lock:
            return self._enabled.get(key, False)

    @property
    def active_preset(self) -> str:
        with self._lock:
            return self._active_preset

    def snapshot(self) -> Dict[str, bool]:
        """Return a copy of current states (for UI refresh)."""
        with self._lock:
            return dict(self._enabled)

    def filter_channels(self, breakdown: Dict[str, float]) -> Dict[str, float]:
        """
        Return a new dict with disabled channels zeroed out.

        The *original* dict is not mutated so diagnostic logging
        still sees the raw values.

        Always preserves ``survival`` and ``extrinsic`` — these
        are very low-magnitude base signals that should never be
        turned off.
        """
        with self._lock:
            enabled = dict(self._enabled)

        filtered = {}
        for key, value in breakdown.items():
            if key in ("survival", "extrinsic", "total"):
                # Always pass through un-filtered.
                filtered[key] = value
            elif enabled.get(key, True):
                filtered[key] = value
            else:
                filtered[key] = 0.0
        return filtered
