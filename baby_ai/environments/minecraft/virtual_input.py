"""
Virtual input controller — camera look via the Fabric mod bridge.

Extends :class:`InputController` so that keyboard and mouse *buttons*
still use the fast, proven PostMessage path, but **camera rotation**
is sent as a ``look`` command over the TCP bridge to the Fabric mod.
The mod applies the yaw/pitch delta directly on the client player
entity, bypassing GLFW raw-input completely.

This means:
- The AI can rotate the camera **without** warping the OS cursor.
- The Minecraft window does **not** need to be focused.
- The user keeps full control of their physical keyboard and mouse.

Usage::

    ctrl = VirtualInputController(window_mgr, mod_bridge)
    ctrl.mouse_look(dx=100, dy=-30)  # routed through mod bridge
    ctrl.press_key("W")              # still PostMessage (unchanged)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from baby_ai.environments.minecraft.input_controller import InputController
from baby_ai.utils.logging import get_logger

if TYPE_CHECKING:
    from baby_ai.environments.minecraft.mod_bridge import ModBridge
    from baby_ai.environments.minecraft.window import WindowManager

log = get_logger("mc_virtual_input")

# Conversion factor: pixels → degrees.
# In active mode the InputController warps the cursor by pixel deltas;
# Minecraft maps ~8 pixels of raw mouse movement to ~1 degree of yaw
# at default sensitivity (0.5).  These defaults produce natural-looking
# rotation that matches the feel of active mode.  Tune them if the
# Minecraft sensitivity slider is changed.
_DEFAULT_PIXELS_PER_DEG_YAW = 8.0
_DEFAULT_PIXELS_PER_DEG_PITCH = 8.0


class VirtualInputController(InputController):
    """InputController with mod-bridge camera look.

    Keyboard and mouse buttons are inherited from
    :class:`InputController` (PostMessage — no cursor hijack).
    Only :meth:`mouse_look` is overridden to route through the
    Fabric mod's TCP command channel.

    Args:
        window: WindowManager for the Minecraft window.
        mod_bridge: Connected :class:`ModBridge` instance.
        pixels_per_deg_yaw: Pixel-to-degree conversion for horizontal
            camera movement.  Matches Minecraft's default sensitivity.
        pixels_per_deg_pitch: Same for vertical movement.
    """

    def __init__(
        self,
        window: "WindowManager",
        mod_bridge: "ModBridge",
        *,
        pixels_per_deg_yaw: float = _DEFAULT_PIXELS_PER_DEG_YAW,
        pixels_per_deg_pitch: float = _DEFAULT_PIXELS_PER_DEG_PITCH,
    ):
        # Always run PostMessage in background mode — no cursor warp.
        super().__init__(window, mode="background")
        self._bridge = mod_bridge
        self._px_per_deg_yaw = pixels_per_deg_yaw
        self._px_per_deg_pitch = pixels_per_deg_pitch

    # ── Camera look (override) ──────────────────────────────────

    def mouse_look(self, dx: int, dy: int) -> bool:
        """Rotate the camera by sending a ``look`` command to the mod.

        The pixel deltas are converted to degrees using the configured
        pixels-per-degree ratios so the action decoder's output range
        produces the same visual rotation as active mode.

        In-game GUI screens (inventory, crafting) still need cursor
        positioning, which is handled by the parent class's background-
        mode ``WM_MOUSEMOVE`` PostMessage.  :meth:`mouse_look` is only
        called for camera rotation; the env distinguishes the two cases
        via the ``has_open_screen`` flag from the mod bridge.

        Returns True if the command was sent, False otherwise.
        """
        if not self._window.is_valid:
            return False
        if self.paused:
            return False

        # Check AI Controls state (UI toggles).
        from baby_ai.environments.minecraft.input_controller import _controls_state
        if _controls_state is not None:
            if not _controls_state.is_look_allowed():
                return False

        # If a GUI screen is open (inventory, chest, etc.), delegate
        # to the PostMessage-based cursor positioning so the AI can
        # click on slots/buttons.
        if self._bridge.has_open_screen:
            return super().mouse_look(dx, dy)

        # Convert pixel deltas → degree deltas
        dyaw = dx / self._px_per_deg_yaw
        dpitch = dy / self._px_per_deg_pitch

        if abs(dyaw) < 0.01 and abs(dpitch) < 0.01:
            return True  # negligible movement

        return self._bridge.send_command({
            "command": "look",
            "dyaw": round(dyaw, 3),
            "dpitch": round(dpitch, 3),
        })

    # ── Properties ──────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Always reports 'virtual' regardless of underlying PostMessage mode."""
        return "virtual"

    @mode.setter
    def mode(self, value: str) -> None:
        # Virtual controller ignores mode changes — it always uses
        # background PostMessage + mod-bridge look.
        if value not in ("background", "active", "virtual"):
            raise ValueError(f"mode must be 'background', 'active', or 'virtual', got '{value}'")
