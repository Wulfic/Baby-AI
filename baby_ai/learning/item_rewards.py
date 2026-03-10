"""
Per-item reward values for every obtainable Minecraft 1.21.x item.

Rewards are assigned based on **rarity** and **difficulty to obtain**,
incentivising the AI to seek out, mine, craft, and collect progressively
harder items.  The values are used by ``MinecraftEnv._compute_reward()``
to scale the flat per-event signals coming from the Fabric mod bridge.

Reward Tiers
------------
Tier 0  (0.1)       — Trivial: everywhere, zero effort (dirt, cobblestone).
Tier 1  (0.2–0.4)   — Common: easy gathering (logs, coal, seeds, common drops).
Tier 2  (0.5–1.0)   — Standard: basic mining / crafting (iron, basic tools).
Tier 3  (1.5–2.5)   — Intermediate: deeper mining / progression (gold, redstone, iron gear).
Tier 4  (3.0–5.0)   — Advanced: significant effort (diamond, blaze rods, enchanting).
Tier 5  (6.0–10.0)  — Rare: Nether/End content, boss prerequisites.
Tier 6  (12.0–20.0) — Legendary: boss drops, endgame gear, one-of-a-kind.

Design Principles
-----------------
* Crafted items > raw materials (incentivises crafting).
* Smelted items > raw ores (incentivises furnace use).
* Cooked food > raw food (incentivises cooking).
* Complex recipes > simple recipes.
* Color variants share the same tier as their base item.
* Spawn eggs are creative-mode items — minimal reward (0.05).
* "Already listed" duplicates are omitted; first definition wins.

The bulk of the data lives in :mod:`baby_ai.learning.item_reward_data`
to keep this API module concise.
"""

from __future__ import annotations

# Import the data tables — populates ITEM_REWARDS at import time.
from baby_ai.learning.item_reward_data import (
    EVENT_MULTIPLIERS,
    ITEM_REWARDS,
    _DEFAULT_REWARD,
)


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════

def get_item_reward(
    item_id: str,
    event_type: str = "item_picked_up",
) -> float:
    """
    Return the reward for a specific item and event type.

    Args:
        item_id: Minecraft namespaced item ID (e.g. ``"minecraft:diamond"``).
        event_type: One of ``"item_picked_up"``, ``"item_crafted"``,
                    ``"block_broken"``, ``"block_placed"``.

    Returns:
        Scaled reward value (base × event multiplier).
    """
    base = ITEM_REWARDS.get(item_id, _DEFAULT_REWARD)
    multiplier = EVENT_MULTIPLIERS.get(event_type, 1.0)
    return base * multiplier


def get_item_tier(item_id: str) -> str:
    """Return a human-readable tier label for an item."""
    reward = ITEM_REWARDS.get(item_id, _DEFAULT_REWARD)
    if reward <= 0.1:
        return "trivial"
    elif reward <= 0.4:
        return "common"
    elif reward <= 1.0:
        return "standard"
    elif reward <= 2.5:
        return "intermediate"
    elif reward <= 5.0:
        return "advanced"
    elif reward <= 10.0:
        return "rare"
    else:
        return "legendary"


# ── Quick stats (useful for debugging / logging) ───────────────
def reward_stats() -> dict:
    """Return summary statistics about the reward table."""
    import statistics
    values = list(ITEM_REWARDS.values())
    return {
        "total_items": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "tiers": {
            "trivial":      sum(1 for v in values if v <= 0.1),
            "common":       sum(1 for v in values if 0.1 < v <= 0.4),
            "standard":     sum(1 for v in values if 0.4 < v <= 1.0),
            "intermediate": sum(1 for v in values if 1.0 < v <= 2.5),
            "advanced":     sum(1 for v in values if 2.5 < v <= 5.0),
            "rare":         sum(1 for v in values if 5.0 < v <= 10.0),
            "legendary":    sum(1 for v in values if v > 10.0),
        },
    }
