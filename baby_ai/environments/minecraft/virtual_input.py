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
        """Rotate the camera, or move the GUI cursor, via the mod bridge.

        When **no** GUI screen is open the pixel deltas are converted to
        degrees and sent as a ``look`` command — the mod applies the
        rotation directly on the player entity (no cursor warp, no focus
        needed).

        When a GUI screen **is** open (inventory, crafting, chest, …) the
        same deltas are sent as a ``gui_move`` command instead, so the mod
        drives ``Screen.mouseMoved`` and the AI's virtual cursor glides
        over the slots.  This replaces the old ``WM_MOUSEMOVE`` PostMessage
        path, which GLFW ignored for GUI screens (the cursor never moved).

        The env distinguishes the two cases via the ``has_open_screen``
        flag, which the mod now pushes the instant a screen opens/closes.

        Returns True if a command was sent, False otherwise.
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

        # If a GUI screen is open, move the in-game cursor over the slots.
        # The mod also self-guards: a stray ``look`` arriving during the
        # open/close race is re-interpreted as cursor movement, never
        # camera rotation, so the camera can't spin behind the inventory.
        if self._bridge.has_open_screen:
            return self._bridge.send_gui_move(dx, dy)

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

    # ── Mouse buttons (override for GUI clicks) ─────────────────

    # GLFW button codes used by Screen.mouseClicked / mouseReleased.
    _GLFW_BUTTON = {"left": 0, "right": 1, "middle": 2}

    def mouse_down(self, button: str = "left", x: int = -1, y: int = -1) -> None:
        """Press a mouse button — routed to the open GUI screen if any.

        While a screen is open, a normal PostMessage click does nothing
        (GLFW owns GUI input), so the press is sent to the mod as a
        ``gui_click`` which calls ``Screen.mouseClicked`` at the virtual
        cursor — picking up / placing inventory stacks.  Otherwise the
        parent's PostMessage world-click is used unchanged.
        """
        if self._bridge.has_open_screen:
            if self.paused:
                return
            from baby_ai.environments.minecraft.input_controller import _controls_state
            if _controls_state is not None and not _controls_state.is_button_allowed(button):
                return
            self._bridge.send_gui_click(self._GLFW_BUTTON.get(button, 0), down=True)
            self._held_buttons.add(button)
            return
        super().mouse_down(button, x, y)

    def mouse_up(self, button: str = "left", x: int = -1, y: int = -1) -> None:
        """Release a mouse button — routed to the open GUI screen if any."""
        if self._bridge.has_open_screen:
            self._bridge.send_gui_click(self._GLFW_BUTTON.get(button, 0), down=False)
            self._held_buttons.discard(button)
            return
        super().mouse_up(button, x, y)

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
