"""
Thread-safe AI input controls state.

Houses the shared state that the tkinter control panel writes to
(UI thread) and the InputController reads from (main thread).
All access is guarded by a threading.Lock so there are no races.

Each control (key, mouse button, camera look) can be individually
enabled or disabled. When disabled, the InputController silently
drops the corresponding input so the AI cannot use it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List


# ── Control definitions ─────────────────────────────────────────

@dataclass(frozen=True)
class ControlInfo:
    """Description of a single AI-controllable input."""
    key: str            # unique identifier (e.g. "key_W", "btn_left", "look")
    label: str          # human-readable label for the UI
    group: str          # group header in the control panel tab
    default: bool       # enabled by default?
    # For keyboard controls, the VK name (matches input_controller.VK)
    vk_name: str = ""
    # For mouse buttons, the button name ("left", "right", "middle")
    button_name: str = ""
    # For camera look
    is_look: bool = False


# Ordered list — UI renders in this order.
AI_CONTROLS: List[ControlInfo] = [
    # ── Movement ────────────────────────────────────────────────
    ControlInfo("key_W",      "W (Forward)",      "Movement",    True,  vk_name="W"),
    ControlInfo("key_A",      "A (Strafe Left)",  "Movement",    True,  vk_name="A"),
    ControlInfo("key_S",      "S (Backward)",     "Movement",    True,  vk_name="S"),
    ControlInfo("key_D",      "D (Strafe Right)", "Movement",    True,  vk_name="D"),
    ControlInfo("key_SPACE",  "Space (Jump)",     "Movement",    True,  vk_name="SPACE"),
    ControlInfo("key_LSHIFT", "Shift (Sneak)",    "Movement",    True,  vk_name="LSHIFT"),
    ControlInfo("key_LCTRL",  "Ctrl (Sprint)",    "Movement",    True,  vk_name="LCTRL"),

    # ── Mouse / Interaction ─────────────────────────────────────
    ControlInfo("btn_left",   "Left Click (Attack/Mine)",   "Mouse",  True,  button_name="left"),
    ControlInfo("btn_right",  "Right Click (Use/Place)",    "Mouse",  True,  button_name="right"),
    ControlInfo("btn_middle", "Middle Click (Pick Block)",  "Mouse",  True,  button_name="middle"),

    # ── Camera ──────────────────────────────────────────────────
    ControlInfo("look",       "Camera Look",      "Camera",      True,  is_look=True),

    # ── Inventory / Items ───────────────────────────────────────
    ControlInfo("key_E",      "E (Inventory)",    "Inventory",   True,  vk_name="E"),
    ControlInfo("key_Q",      "Q (Drop Item)",    "Inventory",   True,  vk_name="Q"),
    ControlInfo("key_F",      "F (Swap Hands)",   "Inventory",   True,  vk_name="F"),

    # ── Hotbar ──────────────────────────────────────────────────
    ControlInfo("key_1",      "1 (Hotbar 1)",     "Hotbar",      True,  vk_name="1"),
    ControlInfo("key_2",      "2 (Hotbar 2)",     "Hotbar",      True,  vk_name="2"),
    ControlInfo("key_3",      "3 (Hotbar 3)",     "Hotbar",      True,  vk_name="3"),
    ControlInfo("key_4",      "4 (Hotbar 4)",     "Hotbar",      True,  vk_name="4"),
    ControlInfo("key_5",      "5 (Hotbar 5)",     "Hotbar",      True,  vk_name="5"),
    ControlInfo("key_6",      "6 (Hotbar 6)",     "Hotbar",      True,  vk_name="6"),
    ControlInfo("key_7",      "7 (Hotbar 7)",     "Hotbar",      True,  vk_name="7"),
    ControlInfo("key_8",      "8 (Hotbar 8)",     "Hotbar",      True,  vk_name="8"),
    ControlInfo("key_9",      "9 (Hotbar 9)",     "Hotbar",      True,  vk_name="9"),
    ControlInfo("key_0",      "0 (Hotbar 10)",    "Hotbar",      True,  vk_name="0"),
]

CONTROL_MAP: Dict[str, ControlInfo] = {c.key: c for c in AI_CONTROLS}

# Unique group names, preserving order of first appearance.
CONTROL_GROUPS: List[str] = list(dict.fromkeys(c.group for c in AI_CONTROLS))

# Build reverse lookups for fast InputController checks.
# VK name → control key  (e.g. "W" → "key_W")
_VK_NAME_TO_CONTROL: Dict[str, str] = {
    c.vk_name: c.key for c in AI_CONTROLS if c.vk_name
}
# Button name → control key  (e.g. "left" → "btn_left")
_BTN_NAME_TO_CONTROL: Dict[str, str] = {
    c.button_name: c.key for c in AI_CONTROLS if c.button_name
}


class AIControlsState:
    """
    Thread-safe container for AI input enable/disable state.

    The tkinter UI calls :meth:`set_enabled`;
    the InputController calls :meth:`is_key_allowed`,
    :meth:`is_button_allowed`, :meth:`is_look_allowed` every step.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Start with each control's default state.
        self._enabled: Dict[str, bool] = {
            c.key: c.default for c in AI_CONTROLS
        }

    # ── Mutators (called from UI thread) ────────────────────────

    def set_enabled(self, key: str, enabled: bool) -> None:
        """Toggle a single control on/off."""
        with self._lock:
            if key in self._enabled:
                self._enabled[key] = enabled

    def set_all(self, enabled: bool) -> None:
        """Enable or disable ALL controls at once."""
        with self._lock:
            for k in self._enabled:
                self._enabled[k] = enabled

    def set_group(self, group: str, enabled: bool) -> None:
        """Enable or disable all controls in the given group."""
        with self._lock:
            for c in AI_CONTROLS:
                if c.group == group:
                    self._enabled[c.key] = enabled

    # ── Readers (called from InputController / main thread) ─────

    def is_key_allowed(self, vk_name: str) -> bool:
        """Check if a keyboard key (by VK name) is allowed."""
        control_key = _VK_NAME_TO_CONTROL.get(vk_name)
        if control_key is None:
            # Unknown key — allow by default (safety/system keys)
            return True
        with self._lock:
            return self._enabled.get(control_key, True)

    def is_vk_allowed(self, vk_code: int, vk_table: dict) -> bool:
        """Check if a VK code is allowed, using the VK name→code table."""
        # Reverse lookup: find the VK name for this code.
        for name, code in vk_table.items():
            if code == vk_code:
                return self.is_key_allowed(name)
        # Not in our control list — allow (could be a system key)
        return True

    def is_button_allowed(self, button: str) -> bool:
        """Check if a mouse button ("left", "right", "middle") is allowed."""
        control_key = _BTN_NAME_TO_CONTROL.get(button)
        if control_key is None:
            return True
        with self._lock:
            return self._enabled.get(control_key, True)

    def is_look_allowed(self) -> bool:
        """Check if camera look is allowed."""
        with self._lock:
            return self._enabled.get("look", True)

    def snapshot(self) -> Dict[str, bool]:
        """Return a copy of current states (for UI refresh)."""
        with self._lock:
            return dict(self._enabled)

    def filter_keys(self, desired_keys: set, vk_table: dict) -> set:
        """Filter a set of VK codes, removing any that are disabled."""
        with self._lock:
            enabled = dict(self._enabled)
        result = set()
        for vk_code in desired_keys:
            # Find the VK name for this code
            found = False
            for name, code in vk_table.items():
                if code == vk_code:
                    control_key = _VK_NAME_TO_CONTROL.get(name)
                    if control_key is None or enabled.get(control_key, True):
                        result.add(vk_code)
                    found = True
                    break
            if not found:
                # Unknown VK code — allow through
                result.add(vk_code)
        return result

    def filter_buttons(self, desired_buttons: set) -> set:
        """Filter a set of button names, removing any that are disabled."""
        with self._lock:
            enabled = dict(self._enabled)
        result = set()
        for btn in desired_buttons:
            control_key = _BTN_NAME_TO_CONTROL.get(btn)
            if control_key is None or enabled.get(control_key, True):
                result.add(btn)
        return result
