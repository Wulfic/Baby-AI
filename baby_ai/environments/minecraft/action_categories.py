"""
Action category sets for reward shaping.

Classifies each action index from :data:`MINECRAFT_ACTIONS` into
semantic categories (movement, interaction, attack, look, etc.)
used by the reward computer to decide bonuses and penalties.

All sets are computed once at import time and are read-only thereafter.
"""

from __future__ import annotations

from baby_ai.environments.minecraft.actions import MINECRAFT_ACTIONS
from baby_ai.environments.minecraft.input_controller import VK

# ── Interaction / movement categories ───────────────────────────
# Which action indices involve actual interaction with the world
INTERACTION_ACTIONS: set[int] = set()
MOVEMENT_ACTIONS: set[int] = set()

_move_vks = {VK.get(k) for k in ("W", "S", "A", "D", "SPACE") if k in VK}

for _i, _a in enumerate(MINECRAFT_ACTIONS):
    # Actions that involve left/right mouse (attack / use / place)
    if _a.buttons:
        INTERACTION_ACTIONS.add(_i)
    # Actions that involve movement keys (W, A, S, D, SPACE)
    if _a.keys:
        INTERACTION_ACTIONS.add(_i)  # any non-noop is "doing something"
    # Specifically movement (walking, jumping, strafing)
    if _a.keys & _move_vks:
        MOVEMENT_ACTIONS.add(_i)

# Actions that specifically attack or place/use blocks
BLOCK_INTERACTION_ACTIONS: set[int] = set()
ATTACK_ACTIONS: set[int] = set()
USE_ACTIONS: set[int] = set()
DROP_ACTIONS: set[int] = set()

for _i, _a in enumerate(MINECRAFT_ACTIONS):
    if "left" in _a.buttons or "right" in _a.buttons:
        BLOCK_INTERACTION_ACTIONS.add(_i)
    if "left" in _a.buttons:
        ATTACK_ACTIONS.add(_i)
    if "right" in _a.buttons:
        USE_ACTIONS.add(_i)
    # Q key = drop item — used to suppress pickup reward after drops
    if VK.get("Q") in _a.keys:
        DROP_ACTIONS.add(_i)

# ── Movement quality categories ─────────────────────────────────
# Actions that involve forward movement (W key) — preferred locomotion
FORWARD_ACTIONS: set[int] = set()
# Actions that involve backward movement (S key without W)
BACKWARD_ACTIONS: set[int] = set()
# Actions that involve camera look — preferred for situational awareness
LOOK_ACTIONS: set[int] = set()
# Pure strafe actions (A/D without W) — less desirable wandering
PURE_STRAFE_ACTIONS: set[int] = set()
# Jump actions (SPACE key)
JUMP_ACTIONS: set[int] = set()
# Sprint actions (LCTRL key)
SPRINT_ACTIONS: set[int] = set()
# Hotbar selection actions — spamming these produces no useful behaviour
HOTBAR_ACTIONS: set[int] = set()

_w_vk = VK.get("W")
_s_vk = VK.get("S")
_a_vk = VK.get("A")
_d_vk = VK.get("D")
_space_vk = VK.get("SPACE")
_lctrl_vk = VK.get("LCTRL")

for _i, _a in enumerate(MINECRAFT_ACTIONS):
    if _w_vk and _w_vk in _a.keys:
        FORWARD_ACTIONS.add(_i)
    if _s_vk and _s_vk in _a.keys and not (_w_vk and _w_vk in _a.keys):
        BACKWARD_ACTIONS.add(_i)
    if _a.look is not None:
        LOOK_ACTIONS.add(_i)
    # Pure strafe = has A or D but NOT W
    has_strafe = (_a_vk and _a_vk in _a.keys) or (_d_vk and _d_vk in _a.keys)
    has_forward = _w_vk and _w_vk in _a.keys
    if has_strafe and not has_forward:
        PURE_STRAFE_ACTIONS.add(_i)
    if _space_vk and _space_vk in _a.keys:
        JUMP_ACTIONS.add(_i)
    if _lctrl_vk and _lctrl_vk in _a.keys:
        SPRINT_ACTIONS.add(_i)

# Hotbar actions: indices that ONLY press a number key (no movement/mouse)
for _i, _a in enumerate(MINECRAFT_ACTIONS):
    if _a.name.startswith("hotbar_") and not _a.buttons and not _a.look:
        HOTBAR_ACTIONS.add(_i)
