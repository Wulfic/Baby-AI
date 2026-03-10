"""
Minecraft discrete action space.

Maps each of the 128 action indices to a concrete combination of
keyboard keys, mouse buttons, and camera-look deltas.

The table is designed so that the agent can learn common Minecraft
behaviours — walking, sprinting, jumping, mining, placing,
looking around — through a single categorical policy head.

Categories:
    0           no-op
    1-7         single movement / modifier keys
    8-12        mouse & interaction
    13-22       hotbar selection
    23-30       camera look (4 dirs × 2 speeds)
    31-127      composite actions (movement + look + combat)

Each entry is a ``MinecraftAction`` with:
    keys:    frozenset of VK codes to hold
    buttons: frozenset of mouse-button names to hold
    look:    (dx, dy) pixel delta for camera, or None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

from baby_ai.environments.minecraft.input_controller import VK

# ── Look delta magnitudes (pixels per step) ─────────────────────
LOOK_SM = 40      # ~5-8° depending on Minecraft sensitivity
LOOK_LG = 160     # ~20-30°


@dataclass(frozen=True)
class MinecraftAction:
    """Immutable description of one discrete action."""
    name: str
    keys: FrozenSet[int] = field(default_factory=frozenset)
    buttons: FrozenSet[str] = field(default_factory=frozenset)
    look: Optional[Tuple[int, int]] = None   # (dx_pixels, dy_pixels)


def _k(*names: str) -> FrozenSet[int]:
    """Shorthand: convert key names to a frozenset of VK codes."""
    return frozenset(VK[n] for n in names)


def _b(*btns: str) -> FrozenSet[str]:
    return frozenset(btns)


# ── Lookup vectors for programmatic combo generation ────────────
_LOOK_DIRS: List[Tuple[str, Tuple[int, int]]] = [
    ("up_sm",    (0, -LOOK_SM)),
    ("down_sm",  (0,  LOOK_SM)),
    ("left_sm",  (-LOOK_SM, 0)),
    ("right_sm", (LOOK_SM, 0)),
    ("up_lg",    (0, -LOOK_LG)),
    ("down_lg",  (0,  LOOK_LG)),
    ("left_lg",  (-LOOK_LG, 0)),
    ("right_lg", (LOOK_LG, 0)),
]

# ── Build the full 128-action table ────────────────────────────
_actions: List[MinecraftAction] = []


def _add(name: str, keys=frozenset(), buttons=frozenset(), look=None):
    _actions.append(MinecraftAction(name=name, keys=keys, buttons=buttons, look=look))


# 0: no-op
_add("noop")

# 1-7: single movement / modifier keys
_add("forward",       keys=_k("W"))
_add("back",          keys=_k("S"))
_add("strafe_left",   keys=_k("A"))
_add("strafe_right",  keys=_k("D"))
_add("jump",          keys=_k("SPACE"))
_add("sneak",         keys=_k("LSHIFT"))
_add("sprint",        keys=_k("LCTRL"))

# 8-12: mouse & interaction
_add("attack",        buttons=_b("left"))
_add("use",           buttons=_b("right"))
_add("pick_block",    buttons=_b("middle"))
_add("drop",          keys=_k("Q"))
_add("inventory",     keys=_k("E"))

# 13-22: hotbar 1-0
for i in range(1, 10):
    _add(f"hotbar_{i}", keys=_k(str(i)))
_add("hotbar_0", keys=_k("0"))

# 23-30: camera look (8 directions × 2 speeds)
for lname, ldir in _LOOK_DIRS:
    _add(f"look_{lname}", look=ldir)

# 31-38: forward + look
for lname, ldir in _LOOK_DIRS:
    _add(f"fwd_look_{lname}", keys=_k("W"), look=ldir)

# 39-44: forward combos
_add("fwd_jump",            keys=_k("W", "SPACE"))
_add("fwd_attack",          keys=_k("W"), buttons=_b("left"))
_add("fwd_use",             keys=_k("W"), buttons=_b("right"))
_add("fwd_sprint",          keys=_k("W", "LCTRL"))
_add("fwd_sprint_jump",     keys=_k("W", "LCTRL", "SPACE"))
_add("fwd_sneak",           keys=_k("W", "LSHIFT"))

# 45-48: other movement + jump
_add("back_jump",           keys=_k("S", "SPACE"))
_add("strafe_left_jump",    keys=_k("A", "SPACE"))
_add("strafe_right_jump",   keys=_k("D", "SPACE"))
_add("back_sneak",          keys=_k("S", "LSHIFT"))

# 49-56: back + look
for lname, ldir in _LOOK_DIRS:
    _add(f"back_look_{lname}", keys=_k("S"), look=ldir)

# 57-64: strafe left + look
for lname, ldir in _LOOK_DIRS:
    _add(f"sleft_look_{lname}", keys=_k("A"), look=ldir)

# 65-72: strafe right + look
for lname, ldir in _LOOK_DIRS:
    _add(f"sright_look_{lname}", keys=_k("D"), look=ldir)

# 73-76: forward + jump + look (small)
_add("fwd_jump_look_up",    keys=_k("W", "SPACE"), look=(0, -LOOK_SM))
_add("fwd_jump_look_down",  keys=_k("W", "SPACE"), look=(0,  LOOK_SM))
_add("fwd_jump_look_left",  keys=_k("W", "SPACE"), look=(-LOOK_SM, 0))
_add("fwd_jump_look_right", keys=_k("W", "SPACE"), look=(LOOK_SM, 0))

# 77-80: forward + attack + look (small)
_add("fwd_atk_look_up",     keys=_k("W"), buttons=_b("left"), look=(0, -LOOK_SM))
_add("fwd_atk_look_down",   keys=_k("W"), buttons=_b("left"), look=(0,  LOOK_SM))
_add("fwd_atk_look_left",   keys=_k("W"), buttons=_b("left"), look=(-LOOK_SM, 0))
_add("fwd_atk_look_right",  keys=_k("W"), buttons=_b("left"), look=(LOOK_SM, 0))

# 81-84: forward + use + look (small)
_add("fwd_use_look_up",     keys=_k("W"), buttons=_b("right"), look=(0, -LOOK_SM))
_add("fwd_use_look_down",   keys=_k("W"), buttons=_b("right"), look=(0,  LOOK_SM))
_add("fwd_use_look_left",   keys=_k("W"), buttons=_b("right"), look=(-LOOK_SM, 0))
_add("fwd_use_look_right",  keys=_k("W"), buttons=_b("right"), look=(LOOK_SM, 0))

# 85-88: attack + look (standing still)
_add("atk_look_up",         buttons=_b("left"), look=(0, -LOOK_SM))
_add("atk_look_down",       buttons=_b("left"), look=(0,  LOOK_SM))
_add("atk_look_left",       buttons=_b("left"), look=(-LOOK_SM, 0))
_add("atk_look_right",      buttons=_b("left"), look=(LOOK_SM, 0))

# 89-92: use + look (standing still)
_add("use_look_up",         buttons=_b("right"), look=(0, -LOOK_SM))
_add("use_look_down",       buttons=_b("right"), look=(0,  LOOK_SM))
_add("use_look_left",       buttons=_b("right"), look=(-LOOK_SM, 0))
_add("use_look_right",      buttons=_b("right"), look=(LOOK_SM, 0))

# 93-96: sneak + movement
_add("sneak_fwd",           keys=_k("LSHIFT", "W"))
_add("sneak_back",          keys=_k("LSHIFT", "S"))
_add("sneak_left",          keys=_k("LSHIFT", "A"))
_add("sneak_right",         keys=_k("LSHIFT", "D"))

# 97-104: sprint + forward + look (small & large)
_add("sprint_fwd_look_up_sm",    keys=_k("W", "LCTRL"), look=(0, -LOOK_SM))
_add("sprint_fwd_look_down_sm",  keys=_k("W", "LCTRL"), look=(0,  LOOK_SM))
_add("sprint_fwd_look_left_sm",  keys=_k("W", "LCTRL"), look=(-LOOK_SM, 0))
_add("sprint_fwd_look_right_sm", keys=_k("W", "LCTRL"), look=(LOOK_SM, 0))
_add("sprint_fwd_look_up_lg",    keys=_k("W", "LCTRL"), look=(0, -LOOK_LG))
_add("sprint_fwd_look_down_lg",  keys=_k("W", "LCTRL"), look=(0,  LOOK_LG))
_add("sprint_fwd_look_left_lg",  keys=_k("W", "LCTRL"), look=(-LOOK_LG, 0))
_add("sprint_fwd_look_right_lg", keys=_k("W", "LCTRL"), look=(LOOK_LG, 0))

# 105-108: sprint + forward + attack combos
_add("sprint_fwd_attack",              keys=_k("W", "LCTRL"), buttons=_b("left"))
_add("sprint_fwd_jump",                keys=_k("W", "LCTRL", "SPACE"))
_add("sprint_fwd_jump_attack",         keys=_k("W", "LCTRL", "SPACE"), buttons=_b("left"))
_add("fwd_attack",                     keys=_k("W"), buttons=_b("left"))

# 109-112: jump + attack/use combos
_add("jump_attack",                    keys=_k("SPACE"), buttons=_b("left"))
_add("jump_use",                       keys=_k("SPACE"), buttons=_b("right"))
_add("sneak_attack",                   keys=_k("LSHIFT"), buttons=_b("left"))
_add("sneak_use",                      keys=_k("LSHIFT"), buttons=_b("right"))

# 113-116: forward + look large
_add("fwd_look_up_lg",    keys=_k("W"), look=(0, -LOOK_LG))
_add("fwd_look_down_lg",  keys=_k("W"), look=(0,  LOOK_LG))
_add("fwd_look_left_lg",  keys=_k("W"), look=(-LOOK_LG, 0))
_add("fwd_look_right_lg", keys=_k("W"), look=(LOOK_LG, 0))

# 117-120: attack + look large
_add("atk_look_up_lg",    buttons=_b("left"), look=(0, -LOOK_LG))
_add("atk_look_down_lg",  buttons=_b("left"), look=(0,  LOOK_LG))
_add("atk_look_left_lg",  buttons=_b("left"), look=(-LOOK_LG, 0))
_add("atk_look_right_lg", buttons=_b("left"), look=(LOOK_LG, 0))

# 121: utility (safe ones only — no ESC, no F-keys)
_add("swap_hands",  keys=_k("F"))

# 122-127: additional gameplay combos (replaces removed ESC/F3/F5)
_add("sneak_fwd_jump",         keys=_k("LSHIFT", "W", "SPACE"))
_add("back_attack",             keys=_k("S"), buttons=_b("left"))
_add("strafe_left_attack",      keys=_k("A"), buttons=_b("left"))
_add("sneak_fwd_attack",        keys=_k("LSHIFT", "W"), buttons=_b("left"))
_add("sneak_fwd_use",           keys=_k("LSHIFT", "W"), buttons=_b("right"))
_add("fwd_sprint_use",          keys=_k("W", "LCTRL"),  buttons=_b("right"))

# ── Public API ──────────────────────────────────────────────────

MINECRAFT_ACTIONS: List[MinecraftAction] = list(_actions)
"""Ordered list of all 128 Minecraft actions, indexed by action id."""

NUM_ACTIONS: int = len(MINECRAFT_ACTIONS)

# Verify we hit exactly 128
assert NUM_ACTIONS == 128, (
    f"Action table has {NUM_ACTIONS} entries, expected 128. "
    "Adjust the definitions above."
)


def action_name(action_id: int) -> str:
    """Return the human-readable name for an action index."""
    return MINECRAFT_ACTIONS[action_id].name


def has_look(action_id: int) -> bool:
    """Return True if the action includes a camera-look component."""
    return MINECRAFT_ACTIONS[action_id].look is not None


def describe_action(action_id: int) -> str:
    """Return a verbose description of an action for logging."""
    a = MINECRAFT_ACTIONS[action_id]
    parts = [a.name]
    if a.keys:
        key_names = [k for k, v in VK.items() if v in a.keys]
        parts.append(f"keys={key_names}")
    if a.buttons:
        parts.append(f"buttons={list(a.buttons)}")
    if a.look:
        parts.append(f"look=({a.look[0]},{a.look[1]})")
    return " | ".join(parts)


# ── Imitation learning: map player input → best action_id ──────

# VK codes used by gameplay actions (subset of VK dict that appear
# in the action table).  Modifier/system codes are excluded.
_GAMEPLAY_VKS: FrozenSet[int] = frozenset(
    a.keys for a in MINECRAFT_ACTIONS for _ in (None,)
).union(*(a.keys for a in MINECRAFT_ACTIONS)) if False else frozenset().union(
    *(a.keys for a in MINECRAFT_ACTIONS if a.keys)
)

# Pre-compute a reverse lookup: VK name → VK code (only gameplay keys)
_GAMEPLAY_BUTTON_NAMES: FrozenSet[str] = frozenset().union(
    *(a.buttons for a in MINECRAFT_ACTIONS if a.buttons)
)


def _quantize_look(dyaw: float, dpitch: float) -> Optional[Tuple[int, int]]:
    """
    Convert yaw/pitch deltas (degrees) into the closest discrete
    look vector from the action table, or None if movement is negligible.

    Minecraft yaw:  positive = turning RIGHT, negative = LEFT.
    Minecraft pitch: positive = looking DOWN, negative = UP.

    The action table uses pixel deltas:
        dx>0 = look RIGHT, dx<0 = look LEFT
        dy>0 = look DOWN,  dy<0 = look UP

    We map degrees to the closest LOOK_SM or LOOK_LG magnitude.
    Threshold: < 1.5 degrees of movement → no look component.
    """
    # Rough conversion: LOOK_SM (40px) ≈ 5-8° and LOOK_LG (160px) ≈ 20-30°.
    # We use thresholds in degrees to decide SM vs LG.
    _THRESHOLD_DEG = 1.5
    _LG_THRESHOLD_DEG = 12.0  # above this → use LOOK_LG
    
    abs_dyaw = abs(dyaw)
    abs_dpitch = abs(dpitch)
    
    # If both below threshold, no look
    if abs_dyaw < _THRESHOLD_DEG and abs_dpitch < _THRESHOLD_DEG:
        return None
    
    # Pick the dominant axis
    if abs_dyaw >= abs_dpitch:
        # Horizontal look
        mag = LOOK_LG if abs_dyaw >= _LG_THRESHOLD_DEG else LOOK_SM
        dx = mag if dyaw > 0 else -mag
        return (dx, 0)
    else:
        # Vertical look
        mag = LOOK_LG if abs_dpitch >= _LG_THRESHOLD_DEG else LOOK_SM
        dy = mag if dpitch > 0 else -mag
        return (0, dy)


def match_player_action(
    held_keys: FrozenSet[int],
    held_buttons: FrozenSet[str],
    dyaw: float = 0.0,
    dpitch: float = 0.0,
) -> int:
    """
    Find the action_id in the 128-action table that best matches the
    player's current physical input state.

    Used during imitation learning to record what the human *actually*
    did, rather than the AI's predicted action.

    Scoring for each candidate action:
      +2  for each key in common with the player
      -1  for each key the action requires but the player isn't pressing
      -0.5 for each extra key the player has that the action doesn't use
      +3  for each mouse button in common
      -2  for each mouse button the action requires but missing
      +2  if look direction matches (both have compatible look)
      -1  if action has look but player doesn't (or vice versa)

    Args:
        held_keys:    frozenset of VK codes the player is holding.
        held_buttons: frozenset of mouse button names ("left"/"right"/"middle").
        dyaw:         yaw change in degrees since last step (>0 = right).
        dpitch:       pitch change in degrees since last step (>0 = down).

    Returns:
        The action_id (0-127) of the best matching action.
    """
    # Filter held_keys to only gameplay-relevant VK codes
    player_keys = held_keys & _GAMEPLAY_VKS
    player_buttons = held_buttons & _GAMEPLAY_BUTTON_NAMES
    player_look = _quantize_look(dyaw, dpitch)
    
    best_id = 0
    best_score = -999.0
    
    for idx, action in enumerate(MINECRAFT_ACTIONS):
        score = 0.0
        
        # ── Key matching ────────────────────────────────
        a_keys = action.keys
        common_keys = player_keys & a_keys
        missing_keys = a_keys - player_keys       # action needs, player doesn't have
        extra_keys = player_keys - a_keys          # player has, action doesn't need
        
        score += len(common_keys) * 2.0
        score -= len(missing_keys) * 1.0
        score -= len(extra_keys) * 0.5
        
        # ── Button matching ─────────────────────────────
        a_buttons = action.buttons
        common_buttons = player_buttons & a_buttons
        missing_buttons = a_buttons - player_buttons
        extra_buttons = player_buttons - a_buttons
        
        score += len(common_buttons) * 3.0
        score -= len(missing_buttons) * 2.0
        score -= len(extra_buttons) * 0.5
        
        # ── Look matching ──────────────────────────────
        a_look = action.look
        if player_look is not None and a_look is not None:
            # Both have look — check if directions match
            if player_look == a_look:
                score += 2.0
            else:
                # Same axis, same direction? Partial credit
                p_dx, p_dy = player_look
                a_dx, a_dy = a_look
                if (p_dx != 0 and a_dx != 0 and (p_dx > 0) == (a_dx > 0)):
                    score += 1.0  # same horizontal direction, different magnitude
                elif (p_dy != 0 and a_dy != 0 and (p_dy > 0) == (a_dy > 0)):
                    score += 1.0  # same vertical direction, different magnitude
                else:
                    score -= 1.0  # wrong direction
        elif player_look is None and a_look is None:
            # Neither has look — slight bonus for matching "no look"
            score += 0.5
        elif a_look is not None:
            # Action expects look but player isn't looking
            score -= 1.0
        else:
            # Player is looking but action has no look component
            score -= 0.3
        
        if score > best_score:
            best_score = score
            best_id = idx
    
    return best_id
