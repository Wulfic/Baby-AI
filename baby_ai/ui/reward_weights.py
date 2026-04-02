"""
Thread-safe reward weight multipliers.

Houses the shared state that the tkinter control panel writes to
(UI thread) and the environment's ``_compute_reward`` reads from
(main thread).  All access is guarded by a ``threading.Lock``.

Each weight is a float multiplier applied to its reward channel
when computing the combined total reward.  A weight of 0.0
effectively disables the channel; negative weights invert it.

The default values match the hardcoded weights that were previously
in ``MinecraftEnv._compute_reward``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class WeightInfo:
    """Description of a single reward weight slider.

    When ``parent`` is set the entry is a *sub-weight* — an internal
    multiplier that lives underneath a top-level channel weight.
    Sub-weights are displayed indented in the UI and can be
    expanded / collapsed.
    """
    key: str            # unique key; sub-weights use "parent.child" style
    label: str          # human-readable label for the UI
    group: str          # group header in the weights tab
    default: float      # default weight multiplier
    min_val: float      # slider minimum
    max_val: float      # slider maximum
    step: float         # slider step size
    is_penalty: bool = False   # True = subtracted from total
    parent: str | None = None  # key of parent weight (None = top-level)


# Ordered list — UI renders in this order.
# Sub-weights MUST appear immediately after their parent.
REWARD_WEIGHTS: List[WeightInfo] = [
    # ── Baseline ────────────────────────────────────────────────
    WeightInfo("intrinsic",         "Intrinsic (JEPA)",  "Baseline",     0.1,    0.0, 10.0,   0.1),
    WeightInfo("survival",          "Survival",           "Baseline",     1.0,    0.0,  5.0,   0.1),
    WeightInfo("visual_change",     "Visual Change",      "Baseline",     0.1,    0.0,  5.0,   0.1),

    # ── Exploration ─────────────────────────────────────────────
    WeightInfo("action_diversity",  "Action Diversity",   "Exploration",  0.5,    0.0, 10.0,   0.1),

    WeightInfo("interaction",       "Interaction",        "Exploration",  0.8,    0.0, 10.0,   0.1),
    # Sub-weights: internal multipliers within the interaction channel
    WeightInfo("int_impact",        "Impact Bonus",       "Exploration",  0.5,    0.0,  3.0,   0.05, parent="interaction"),
    WeightInfo("int_sustained",     "Sustained Mining",   "Exploration",  0.2,    0.0,  2.0,   0.05, parent="interaction"),

    WeightInfo("exploration",       "Exploration",        "Exploration",  0.8,    0.0, 10.0,   0.1),

    WeightInfo("movement",          "Movement",           "Exploration",  0.3,    0.0, 10.0,   0.1),
    # Sub-weights: internal multipliers within the movement channel
    WeightInfo("mv_forward",        "Forward (W)",        "Exploration",  3.0,    0.0, 10.0,   0.1, parent="movement"),
    WeightInfo("mv_backward",       "Backward (S)",       "Exploration",  1.0,    0.0, 10.0,   0.1, parent="movement"),
    WeightInfo("mv_strafe",         "Strafe (A/D)",       "Exploration",  0.4,    0.0,  5.0,   0.1, parent="movement"),
    WeightInfo("mv_look",           "Camera Look",        "Exploration",  0.02,   0.0,  0.5,   0.01, parent="movement"),
    WeightInfo("mv_jump",           "Jump (Space)",       "Exploration",  1.5,    0.0,  5.0,   0.1, parent="movement"),
    WeightInfo("mv_sprint",         "Sprint (Ctrl)",      "Exploration",  1.5,    0.0,  5.0,   0.1, parent="movement"),
    WeightInfo("new_chunk",          "New Chunk",          "Exploration",  1.0,    0.0, 10.0,   0.1),
    # ── Resource Gathering ──────────────────────────────────────
    WeightInfo("block_break",       "Block Break",        "Resources",    4.0,    0.0, 30.0,   0.5),
    WeightInfo("item_pickup",       "Item Pickup",        "Resources",    6.0,    0.0, 30.0,   0.5),

    # ── Creation ────────────────────────────────────────────────
    WeightInfo("block_place",       "Block Place",        "Creation",     4.0,    0.0, 30.0,   0.5),
    WeightInfo("crafting",          "Crafting",           "Creation",    25.0,    0.0, 50.0,   1.0),
    WeightInfo("building_streak",   "Building Streak",    "Creation",     3.0,    0.0, 20.0,   0.5),
    WeightInfo("creative_sequence", "Creative Sequence",  "Creation",     6.0,    0.0, 30.0,   0.5),

    # ── Penalties ───────────────────────────────────────────────
    WeightInfo("death_penalty",      "Death Penalty",      "Penalties",    5.0,   0.0, 20.0,  0.5, is_penalty=True),
    WeightInfo("idle_penalty",       "Idle Penalty",       "Penalties",    2.0,   0.0, 10.0,  0.1, is_penalty=True),
    WeightInfo("stagnation_penalty", "Stagnation Penalty", "Penalties",    3.0,   0.0, 10.0,  0.1, is_penalty=True),
    WeightInfo("item_drop_penalty",  "Item Drop Penalty",  "Penalties",    3.0,   0.0, 10.0,  0.1, is_penalty=True),
    WeightInfo("damage_taken",       "Damage Taken",       "Penalties",    1.5,   0.0, 10.0,  0.1, is_penalty=True),
    WeightInfo("hotbar_spam_penalty","Hotbar Spam Penalty","Penalties",    2.0,   0.0, 10.0,  0.1, is_penalty=True),
    WeightInfo("inventory_spam_penalty","Inventory Spam","Penalties",   2.0,   0.0, 10.0,  0.1, is_penalty=True),

    WeightInfo("height_penalty",     "Height Penalty",     "Penalties",    2.5,   0.0, 10.0,  0.1, is_penalty=True),
    # Sub-weights: components of the height penalty
    WeightInfo("height_underground", "Underground",        "Penalties",    1.0,   0.0,  5.0,  0.1, parent="height_penalty"),
    WeightInfo("height_fall",        "Fall Damage",        "Penalties",    1.0,   0.0,  5.0,  0.1, parent="height_penalty"),
    WeightInfo("height_darkness",    "Darkness",           "Penalties",    1.0,   0.0,  5.0,  0.1, parent="height_penalty"),

    WeightInfo("pitch_penalty",      "Pitch Penalty",      "Penalties",    3.0,   0.0, 10.0,  0.1, is_penalty=True),

    # ── Survival / Sustain ──────────────────────────────────────
    WeightInfo("healing",           "Healing",            "Sustain",      1.0,   0.0, 10.0,  0.1),
    WeightInfo("food_reward",       "Food Reward",        "Sustain",      0.8,   0.0, 10.0,  0.1),
    WeightInfo("xp_reward",         "XP Reward",          "Sustain",      0.1,   0.0,  5.0,  0.1),
    WeightInfo("home_proximity",    "Home Proximity",     "Sustain",      1.5,  -5.0, 10.0,  0.1),
    # ── Combat ───────────────────────────────────────────────────
    WeightInfo("entity_hit",        "Entity Hit",         "Combat",      3.0,   0.0, 20.0,  0.5),
    WeightInfo("mob_killed",        "Mob Killed",         "Combat",      8.0,   0.0, 30.0,  0.5),]

# Lookup: which keys have children?
PARENT_KEYS: set[str] = {w.parent for w in REWARD_WEIGHTS if w.parent} - {None}

WEIGHT_MAP: Dict[str, WeightInfo] = {w.key: w for w in REWARD_WEIGHTS}

# Unique group names, preserving order of first appearance.
WEIGHT_GROUPS: List[str] = list(dict.fromkeys(w.group for w in REWARD_WEIGHTS))

# Children mapping: parent_key → [child WeightInfo, ...]
WEIGHT_CHILDREN: Dict[str, List[WeightInfo]] = {}
for _w in REWARD_WEIGHTS:
    if _w.parent:
        WEIGHT_CHILDREN.setdefault(_w.parent, []).append(_w)

# Top-level weights only (excludes sub-weights for grouping purposes).
TOP_LEVEL_WEIGHTS: List[WeightInfo] = [w for w in REWARD_WEIGHTS if w.parent is None]


class RewardWeightsState:
    """
    Thread-safe container for reward weight multipliers.

    The tkinter UI calls :meth:`set_weight`;
    the environment calls :meth:`get_weights` each step to get
    the full weight dict for computing total reward.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._weights: Dict[str, float] = {
            w.key: w.default for w in REWARD_WEIGHTS
        }

    # ── Mutators (called from UI thread) ────────────────────────

    def set_weight(self, key: str, value: float) -> None:
        """Set a single weight."""
        with self._lock:
            if key in self._weights:
                self._weights[key] = value

    def set_all(self, weights: Dict[str, float]) -> None:
        """Bulk-set weights from a dict (e.g. loaded from settings)."""
        with self._lock:
            for k, v in weights.items():
                if k in self._weights:
                    self._weights[k] = float(v)

    def reset_defaults(self) -> None:
        """Reset all weights to their default values."""
        with self._lock:
            self._weights = {w.key: w.default for w in REWARD_WEIGHTS}

    # ── Readers (called from env / main thread) ─────────────────

    def get_weight(self, key: str) -> float:
        """Get a single weight value."""
        with self._lock:
            return self._weights.get(key, 0.0)

    def snapshot(self) -> Dict[str, float]:
        """Return a copy of all current weights."""
        with self._lock:
            return dict(self._weights)
