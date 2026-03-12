"""
Continuous action decoder for Minecraft.

Maps the 23-dim continuous action vector from the DiffusionPolicyHead
to concrete key presses, mouse buttons, and camera look deltas.

Action vector layout (from DiffusionPolicyHead):
    [0:2]   camera (yaw_delta, pitch_delta) in [-1, 1]
    [2:6]   movement (forward, back, left, right) in [0, 1]
    [6:14]  actions (attack, use, jump, sneak, sprint, inventory, drop, pick_block) in [0, 1]
    [14:23] hotbar (9 slots, softmax-normalised) in [0, 1]
"""

from __future__ import annotations

from typing import FrozenSet, Optional, Set, Tuple

import torch

from baby_ai.environments.minecraft.input_controller import VK

# ── Threshold for binary activations ────────────────────────────
# Actions with activation > this are treated as "pressed"
PRESS_THRESHOLD = 0.5

# Camera sensitivity: maps [-1, 1] → pixel deltas
# LOOK_SM = 40px, LOOK_LG = 160px from actions.py
CAMERA_SCALE_X = 160   # max pixels per tick for yaw
CAMERA_SCALE_Y = 120   # max pixels per tick for pitch
CAMERA_DEADZONE = 0.05  # ignore tiny camera signals


class ContinuousActionDecoder:
    """
    Decodes a 23-dim continuous action vector into Minecraft inputs.

    Provides:
      - Keys, buttons, and look deltas for the InputController
      - Category flags (is_attack, is_movement, etc.) for reward shaping
      - A human-readable description for logging
      - An approximate discrete action ID for the reward computer
    """

    def __init__(self, threshold: float = PRESS_THRESHOLD):
        self.threshold = threshold

    def decode(
        self,
        action: torch.Tensor,
    ) -> dict:
        """
        Decode a continuous action vector into Minecraft inputs.

        Args:
            action: (23,) or (1, 23) continuous action tensor.

        Returns:
            dict with:
                keys:       frozenset of VK codes to hold
                buttons:    frozenset of mouse button names
                look:       (dx, dy) pixel delta or None
                is_attack:  bool
                is_movement: bool
                is_look:    bool
                hotbar_slot: int or None (1-9, None if no switch)
                action_name: str description
                approx_action_id: int (0-127 approximate match)
        """
        if action.dim() > 1:
            action = action.squeeze(0)

        # Detach and move to CPU for decoding
        if action.requires_grad:
            action = action.detach()
        a = action.cpu().float()

        # ── Parse sub-vectors ───────────────────────────────
        cam_yaw = a[0].item()     # [-1, 1] yaw delta
        cam_pitch = a[1].item()   # [-1, 1] pitch delta

        fwd = a[2].item()         # forward
        back = a[3].item()        # back
        left = a[4].item()        # strafe left
        right = a[5].item()       # strafe right

        attack = a[6].item()      # attack (left click)
        use = a[7].item()         # use (right click)
        jump = a[8].item()        # jump
        sneak = a[9].item()       # sneak
        sprint = a[10].item()     # sprint
        inventory = a[11].item()  # inventory (E key)
        drop = a[12].item()       # drop (Q key)
        pick_block = a[13].item() # pick block (middle click)

        hotbar = a[14:23]         # 9 hotbar slots

        # ── Keys ────────────────────────────────────────────
        keys: Set[int] = set()
        t = self.threshold

        # Mutual exclusivity: if both axes fire, only the stronger one wins.
        # This prevents contradictory fwd+back or left+right presses.
        if fwd > t or back > t:
            if fwd >= back:
                keys.add(VK["W"])
            else:
                keys.add(VK["S"])
        if left > t or right > t:
            if left >= right:
                keys.add(VK["A"])
            else:
                keys.add(VK["D"])
        if jump > t:
            keys.add(VK["SPACE"])
        if sneak > t:
            keys.add(VK["LSHIFT"])
        if sprint > t:
            keys.add(VK["LCTRL"])
        if inventory > t:
            keys.add(VK["E"])
        if drop > t:
            keys.add(VK["Q"])

        # ── Buttons ─────────────────────────────────────────
        buttons: Set[str] = set()
        if attack > t:
            buttons.add("left")
        if use > t:
            buttons.add("right")
        if pick_block > t:
            buttons.add("middle")

        # ── Look ────────────────────────────────────────────
        dx = int(cam_yaw * CAMERA_SCALE_X) if abs(cam_yaw) > CAMERA_DEADZONE else 0
        dy = int(cam_pitch * CAMERA_SCALE_Y) if abs(cam_pitch) > CAMERA_DEADZONE else 0
        look = (dx, dy) if dx != 0 or dy != 0 else None

        # ── Hotbar ──────────────────────────────────────────
        hotbar_slot: int | None = None
        max_hotbar = hotbar.max().item()
        if max_hotbar > t:
            slot_idx = hotbar.argmax().item()  # 0-8
            hotbar_slot = slot_idx + 1  # 1-9
            # Map slot number to key VK code
            slot_key = str(hotbar_slot) if hotbar_slot < 10 else "0"
            keys.add(VK[slot_key])

        # ── Category flags ──────────────────────────────────
        is_attack = attack > t
        is_use = use > t
        is_movement = any(v > t for v in [fwd, back, left, right])
        is_look = look is not None
        is_jump = jump > t
        is_sneak = sneak > t
        is_sprint = sprint > t
        is_inventory = inventory > t
        is_drop = drop > t
        is_pick_block = pick_block > t

        # ── Human-readable name ─────────────────────────────
        # Pass the resolved (mutually exclusive) movement flags
        eff_fwd = (fwd > t or back > t) and fwd >= back
        eff_back = (fwd > t or back > t) and back > fwd
        eff_left = (left > t or right > t) and left >= right
        eff_right = (left > t or right > t) and right > left
        name = self._make_name(
            eff_fwd, eff_back, eff_left, eff_right,
            is_attack, is_use, is_jump, is_sneak, is_sprint,
            is_inventory, is_drop, is_pick_block,
            is_look, hotbar_slot, cam_yaw, cam_pitch,
        )

        # ── Approximate discrete action ID ──────────────────
        approx_id = self._approximate_discrete_id(
            keys, buttons, look, is_attack, is_movement, is_look,
        )

        return {
            "keys": frozenset(keys),
            "buttons": frozenset(buttons),
            "look": look,
            "is_attack": is_attack,
            "is_use": is_use,
            "is_movement": is_movement,
            "is_look": is_look,
            "is_jump": is_jump,
            "is_sneak": is_sneak,
            "is_sprint": is_sprint,
            "is_inventory": is_inventory,
            "is_drop": is_drop,
            "is_pick_block": is_pick_block,
            "hotbar_slot": hotbar_slot,
            "action_name": name,
            "approx_action_id": approx_id,
        }

    def _make_name(
        self, fwd, back, left, right,
        attack, use, jump, sneak, sprint,
        inventory, drop, pick_block,
        look, hotbar_slot, cam_yaw, cam_pitch,
    ) -> str:
        """Build a human-readable name for logging."""
        parts = []
        if sneak:
            parts.append("sneak")
        if sprint:
            parts.append("sprint")
        if fwd:
            parts.append("fwd")
        if back:
            parts.append("back")
        if left:
            parts.append("left")
        if right:
            parts.append("right")
        if jump:
            parts.append("jump")
        if attack:
            parts.append("attack")
        if use:
            parts.append("use")
        if inventory:
            parts.append("inv")
        if drop:
            parts.append("drop")
        if pick_block:
            parts.append("pick")
        if look:
            dx_name = "R" if cam_yaw > 0 else "L"
            dy_name = "D" if cam_pitch > 0 else "U"
            if abs(cam_yaw) > abs(cam_pitch):
                parts.append(f"look_{dx_name}")
            elif abs(cam_pitch) > 0.05:
                parts.append(f"look_{dy_name}")
        if hotbar_slot:
            parts.append(f"hotbar_{hotbar_slot}")

        return "+".join(parts) if parts else "noop"

    def _approximate_discrete_id(
        self, keys, buttons, look, is_attack, is_movement, is_look,
    ) -> int:
        """
        Map continuous action to the closest discrete action ID (0-127).

        This is a best-effort match for compatibility with
        logging, reward_computer, and action_categories.
        """
        from baby_ai.environments.minecraft.actions import (
            MINECRAFT_ACTIONS,
            NUM_ACTIONS,
        )

        best_id = 0
        best_score = -999.0

        for idx, action in enumerate(MINECRAFT_ACTIONS):
            score = 0.0

            # Key matching
            common_keys = keys & action.keys
            missing_keys = action.keys - keys
            extra_keys = keys - action.keys

            score += len(common_keys) * 2.0
            score -= len(missing_keys) * 1.5
            score -= len(extra_keys) * 0.5

            # Button matching
            common_buttons = buttons & action.buttons
            missing_buttons = action.buttons - buttons
            extra_buttons = buttons - action.buttons

            score += len(common_buttons) * 3.0
            score -= len(missing_buttons) * 2.0
            score -= len(extra_buttons) * 0.5

            # Look matching
            if look is not None and action.look is not None:
                # Same direction bonus
                adx, ady = action.look
                ldx, ldy = look
                if (adx > 0) == (ldx > 0) and (adx != 0 or ldx == 0):
                    score += 1.0
                if (ady > 0) == (ldy > 0) and (ady != 0 or ldy == 0):
                    score += 1.0
            elif look is None and action.look is None:
                score += 0.5
            elif action.look is not None:
                score -= 1.0

            if score > best_score:
                best_score = score
                best_id = idx

        return best_id


# ── Module-level singleton for convenience ──────────────────────
_decoder = ContinuousActionDecoder()


def decode_continuous_action(action: torch.Tensor) -> dict:
    """Convenience function using the module-level decoder."""
    return _decoder.decode(action)


def continuous_action_name(action: torch.Tensor) -> str:
    """Get a human-readable name for a continuous action vector."""
    return _decoder.decode(action)["action_name"]
