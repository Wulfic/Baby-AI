"""
Baby-AI Control Panel — tkinter UI for runtime control.

Provides:
- Pause / Stop buttons
- Training-phase preset buttons (Phase 1–4)
- Per-channel reward toggle checkboxes in a multi-column grid
- AI Controls tab to enable/disable individual keys/buttons/look
- Set New Home button to update the agent's home location
- Live reward / step readout

Runs on a daemon thread so it doesn't block the training loop.
All shared state is mediated through :class:`RewardToggleState`
and :class:`AIControlsState` which are thread-safe.
"""

from __future__ import annotations

import ctypes
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from baby_ai.ui.reward_toggles import (
    CHANNELS,
    GROUPS,
    PHASE_PRESETS,
    RewardToggleState,
)
from baby_ai.ui.controls_state import (
    AI_CONTROLS,
    CONTROL_GROUPS,
    AIControlsState,
)


def _get_dpi_scale() -> float:
    """Return a UI scale factor based on the primary monitor DPI.

    On standard 96-dpi screens this returns 1.0.
    On 4K / HiDPI screens (typically 144-192 dpi) it returns 1.5-2.0+.
    Falls back to 1.0 if DPI detection fails.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return max(dpi / 96.0, 1.0)
    except Exception:
        return 1.0


# ── Colour palette (dark-mode inspired) ────────────────────────
_BG          = "#1e1e2e"
_BG_FRAME    = "#2a2a3c"
_BG_GROUP    = "#33334d"
_FG          = "#cdd6f4"
_FG_DIM      = "#6c7086"
_ACCENT      = "#89b4fa"
_ACCENT_DARK = "#585b70"
_PAUSE_ON    = "#a6e3a1"
_STOP_BG     = "#f38ba8"
_BTN_BG      = "#45475a"
_BTN_FG      = "#cdd6f4"
_PHASE_SEL   = "#89b4fa"
_PHASE_UNSEL = "#45475a"

# Number of columns for the reward-channel grid.
_CHANNEL_COLS = 3


class AIControlPanel:
    """
    Floating control panel for the Baby-AI Minecraft trainer.

    Parameters
    ----------
    on_stop : callable, optional
        Invoked when the user clicks *Stop & Save*.
    toggle_state : RewardToggleState, optional
        Shared reward-channel state.  If ``None`` a private instance
        is created (useful for standalone testing).
    controls_state : AIControlsState, optional
        Shared AI input-control state.  If ``None`` a private instance
        is created.
    on_set_home : callable, optional
        Invoked when the user clicks *Set New Home*.  Should grab the
        current player coordinates and update the env's home location.
    input_guard : object, optional
        Reference to the :class:`InputGuard` so the Pause button can
        toggle keyboard/mouse blocking in sync.
    """

    def __init__(
        self,
        on_stop: Optional[Callable[[], None]] = None,
        toggle_state: Optional[RewardToggleState] = None,
        controls_state: Optional[AIControlsState] = None,
        on_set_home: Optional[Callable[[], None]] = None,
        input_guard: Optional[object] = None,
    ):
        self.is_paused: bool = False
        self.is_stopped: bool = False
        self.on_stop = on_stop
        self.on_set_home = on_set_home
        self._input_guard = input_guard

        # Shared state for reward toggles.
        self.toggle_state: RewardToggleState = (
            toggle_state if toggle_state is not None else RewardToggleState()
        )

        # Shared state for AI input controls.
        self.controls_state: AIControlsState = (
            controls_state if controls_state is not None else AIControlsState()
        )

        self.root: Optional[tk.Tk] = None
        self._check_vars: dict[str, tk.BooleanVar] = {}
        self._ctrl_check_vars: dict[str, tk.BooleanVar] = {}
        self._phase_buttons: dict[str, tk.Button] = {}
        self._reward_label: Optional[tk.Label] = None
        self._step_label: Optional[tk.Label] = None

        self._live_reward: float = 0.0
        self._live_step: int = 0

        # Flag checked by _poll_updates to self-destruct when the
        # training loop signals shutdown from another thread.
        self._close_requested: bool = False

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the UI on a daemon thread."""
        thread = threading.Thread(target=self._run_ui, daemon=True)
        thread.start()

    def update_live_stats(self, reward: float, step: int) -> None:
        """Push live stats from the training loop (thread-safe)."""
        self._live_reward = reward
        self._live_step = step

    # ────────────────────────────────────────────────────────────
    # UI construction
    # ────────────────────────────────────────────────────────────

    def _run_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Baby-AI Control")
        self.root.attributes("-topmost", True)

        # ── DPI scaling ────────────────────────────────────────
        self._scale = _get_dpi_scale()
        s = self._scale

        self.root.configure(bg=_BG, padx=int(8 * s), pady=int(8 * s))
        self.root.resizable(True, True)
        self.root.minsize(int(400 * s), int(300 * s))

        # Position: top-right of screen.
        win_w = int(720 * s)
        win_h = int(580 * s)
        screen_w = self.root.winfo_screenwidth()
        x_pos = screen_w - win_w - 20
        y_pos = 40
        self.root.geometry(f"{win_w}x{win_h}+{x_pos}+{y_pos}")

        # ── Top bar: title + pause/stop ────────────────────────
        top_bar = tk.Frame(self.root, bg=_BG)
        top_bar.pack(fill=tk.X, pady=(0, int(6 * s)))

        tk.Label(
            top_bar, text="Baby-AI  Control",
            font=("Segoe UI", int(14 * s), "bold"),
            bg=_BG, fg=_ACCENT, anchor="w",
        ).pack(side=tk.LEFT)

        # Buttons on the right side of the title row.
        self.btn_stop = tk.Button(
            top_bar, text="\u2b1b  Stop && Save", command=self.trigger_stop,
            bg=_STOP_BG, fg="#1e1e2e", activebackground="#eba0ac",
            font=("Segoe UI", int(10 * s), "bold"),
            relief="flat", bd=0, padx=int(10 * s), pady=int(4 * s),
        )
        self.btn_stop.pack(side=tk.RIGHT, padx=(int(4 * s), 0))

        self.btn_pause = tk.Button(
            top_bar, text="\u23f8  Pause", command=self.toggle_pause,
            bg=_BTN_BG, fg=_BTN_FG, activebackground=_ACCENT_DARK,
            font=("Segoe UI", int(10 * s), "bold"),
            relief="flat", bd=0, padx=int(10 * s), pady=int(4 * s),
        )
        self.btn_pause.pack(side=tk.RIGHT, padx=(int(4 * s), 0))

        # ── Set New Home button ────────────────────────────────
        self.btn_set_home = tk.Button(
            top_bar, text="\U0001f3e0  Set New Home",
            command=self._on_set_home,
            bg=_BTN_BG, fg=_BTN_FG, activebackground=_ACCENT_DARK,
            font=("Segoe UI", int(9 * s), "bold"),
            relief="flat", bd=0, padx=int(8 * s), pady=int(4 * s),
        )
        self.btn_set_home.pack(side=tk.RIGHT, padx=(int(4 * s), 0))

        # ── Phase presets row ──────────────────────────────────
        phase_row = tk.Frame(self.root, bg=_BG)
        phase_row.pack(fill=tk.X, pady=(0, int(6 * s)))

        tk.Label(
            phase_row, text="PHASE",
            font=("Segoe UI", int(9 * s), "bold"),
            bg=_BG, fg=_FG_DIM, anchor="w",
        ).pack(side=tk.LEFT, padx=(0, int(6 * s)))

        for pid, (plabel, _groups) in PHASE_PRESETS.items():
            short = plabel.split("\u2014")[0].strip()  # "Phase 1"
            btn = tk.Button(
                phase_row, text=short,
                command=lambda p=pid: self._on_phase(p),
                bg=_PHASE_UNSEL, fg=_BTN_FG,
                activebackground=_ACCENT_DARK,
                font=("Segoe UI", int(9 * s), "bold"),
                relief="flat", bd=0, padx=int(8 * s), pady=int(3 * s),
            )
            btn.pack(side=tk.LEFT, padx=int(2 * s))
            self._phase_buttons[pid] = btn

        self._highlight_phase(self.toggle_state.active_preset)

        self._sep(self.root)

        # ── Tabbed notebook (Rewards / AI Controls) ────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook", background=_BG, borderwidth=0)
        style.configure(
            "Dark.TNotebook.Tab",
            background=_BTN_BG, foreground=_BTN_FG,
            padding=[int(10 * s), int(4 * s)],
            font=("Segoe UI", int(9 * s), "bold"),
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", _ACCENT)],
            foreground=[("selected", "#1e1e2e")],
        )

        self._notebook = ttk.Notebook(self.root, style="Dark.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True, pady=(0, int(4 * s)))

        # ── Tab 1: Reward Channels ─────────────────────────────
        rewards_frame = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(rewards_frame, text="  Rewards  ")
        self._build_rewards_tab(rewards_frame)

        # ── Tab 2: AI Controls ─────────────────────────────────
        controls_frame = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(controls_frame, text="  AI Controls  ")
        self._build_controls_tab(controls_frame)

        # ── Live stats bar (bottom) ────────────────────────────
        self._sep(self.root)
        stats_frame = tk.Frame(self.root, bg=_BG)
        stats_frame.pack(fill=tk.X)

        self._reward_label = tk.Label(
            stats_frame, text="Reward: \u2014",
            font=("Consolas", int(10 * s)),
            bg=_BG, fg=_ACCENT, anchor="w",
        )
        self._reward_label.pack(side=tk.LEFT, padx=(0, int(10 * s)))

        self._step_label = tk.Label(
            stats_frame, text="Step: \u2014",
            font=("Consolas", int(10 * s)),
            bg=_BG, fg=_FG_DIM, anchor="e",
        )
        self._step_label.pack(side=tk.RIGHT)

        # ── Periodic updates ───────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self.trigger_stop)
        self._poll_updates()
        self.root.mainloop()

    # ────────────────────────────────────────────────────────────
    # Tab builders
    # ────────────────────────────────────────────────────────────

    def _build_rewards_tab(self, parent: tk.Frame) -> None:
        """Build the reward-channel toggle grid inside *parent*."""
        s = self._scale

        # ── Channel toggles label ──────────────────────────────
        tk.Label(
            parent, text="REWARD CHANNELS",
            font=("Segoe UI", int(9 * s), "bold"),
            bg=_BG, fg=_FG_DIM, anchor="w",
        ).pack(fill=tk.X, pady=(int(4 * s), int(3 * s)))

        # ── Scrollable multi-column channel grid ───────────────
        canvas_frame = tk.Frame(parent, bg=_BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, int(4 * s)))

        canvas = tk.Canvas(canvas_frame, bg=_BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(
            canvas_frame, orient="vertical", command=canvas.yview,
        )
        self._inner = tk.Frame(canvas, bg=_BG)

        self._inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse-wheel scrolling.
        def _on_mousewheel(event: tk.Event) -> None:
            canvas.yview_scroll(-1 * (event.delta // 120), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Build group cards in a multi-column grid ───────────
        snapshot = self.toggle_state.snapshot()

        for c in range(_CHANNEL_COLS):
            self._inner.columnconfigure(c, weight=1, uniform="grp")

        from collections import OrderedDict
        grouped: OrderedDict[str, list] = OrderedDict()
        for ch in CHANNELS:
            grouped.setdefault(ch.group, []).append(ch)

        row_idx, col_idx = 0, 0
        for group_name, channels in grouped.items():
            grp_frame = tk.Frame(
                self._inner, bg=_BG_GROUP,
                padx=int(6 * s), pady=int(4 * s),
            )
            grp_frame.grid(
                row=row_idx, column=col_idx,
                sticky="nsew", padx=int(3 * s), pady=int(3 * s),
            )

            tk.Label(
                grp_frame, text=group_name.upper(),
                font=("Segoe UI", int(8 * s), "bold"),
                bg=_BG_GROUP, fg=_FG_DIM, anchor="w",
            ).pack(fill=tk.X)

            for ch in channels:
                var = tk.BooleanVar(value=snapshot.get(ch.key, ch.default))
                self._check_vars[ch.key] = var

                cb = tk.Checkbutton(
                    grp_frame,
                    text=ch.label,
                    variable=var,
                    command=lambda k=ch.key: self._on_toggle(k),
                    bg=_BG_GROUP, fg=_FG,
                    selectcolor=_BG_FRAME,
                    activebackground=_BG_GROUP,
                    activeforeground=_ACCENT,
                    font=("Segoe UI", int(9 * s)),
                    anchor="w",
                    bd=0, highlightthickness=0,
                )
                cb.pack(fill=tk.X, padx=(int(12 * s), 0))

            col_idx += 1
            if col_idx >= _CHANNEL_COLS:
                col_idx = 0
                row_idx += 1

    def _build_controls_tab(self, parent: tk.Frame) -> None:
        """Build the AI input-controls toggle grid inside *parent*."""
        s = self._scale
        _CTRL_COLS = 3

        # ── Header with Enable All / Disable All ───────────────
        header = tk.Frame(parent, bg=_BG)
        header.pack(fill=tk.X, pady=(int(4 * s), int(3 * s)))

        tk.Label(
            header, text="AI INPUT CONTROLS",
            font=("Segoe UI", int(9 * s), "bold"),
            bg=_BG, fg=_FG_DIM, anchor="w",
        ).pack(side=tk.LEFT)

        tk.Button(
            header, text="Disable All",
            command=lambda: self._set_all_controls(False),
            bg=_STOP_BG, fg="#1e1e2e", activebackground="#eba0ac",
            font=("Segoe UI", int(8 * s), "bold"),
            relief="flat", bd=0, padx=int(6 * s), pady=int(2 * s),
        ).pack(side=tk.RIGHT, padx=(int(4 * s), 0))

        tk.Button(
            header, text="Enable All",
            command=lambda: self._set_all_controls(True),
            bg=_PAUSE_ON, fg="#1e1e2e", activebackground="#a6e3a1",
            font=("Segoe UI", int(8 * s), "bold"),
            relief="flat", bd=0, padx=int(6 * s), pady=int(2 * s),
        ).pack(side=tk.RIGHT, padx=(int(4 * s), 0))

        # ── Scrollable grid ────────────────────────────────────
        canvas_frame = tk.Frame(parent, bg=_BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, int(4 * s)))

        canvas = tk.Canvas(canvas_frame, bg=_BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(
            canvas_frame, orient="vertical", command=canvas.yview,
        )
        inner = tk.Frame(canvas, bg=_BG)

        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for c in range(_CTRL_COLS):
            inner.columnconfigure(c, weight=1, uniform="ctrl")

        # ── Group cards ────────────────────────────────────────
        snapshot = self.controls_state.snapshot()

        from collections import OrderedDict
        grouped: OrderedDict[str, list] = OrderedDict()
        for ctrl in AI_CONTROLS:
            grouped.setdefault(ctrl.group, []).append(ctrl)

        row_idx, col_idx = 0, 0
        for group_name, controls in grouped.items():
            grp_frame = tk.Frame(
                inner, bg=_BG_GROUP,
                padx=int(6 * s), pady=int(4 * s),
            )
            grp_frame.grid(
                row=row_idx, column=col_idx,
                sticky="nsew", padx=int(3 * s), pady=int(3 * s),
            )

            tk.Label(
                grp_frame, text=group_name.upper(),
                font=("Segoe UI", int(8 * s), "bold"),
                bg=_BG_GROUP, fg=_FG_DIM, anchor="w",
            ).pack(fill=tk.X)

            for ctrl in controls:
                var = tk.BooleanVar(value=snapshot.get(ctrl.key, ctrl.default))
                self._ctrl_check_vars[ctrl.key] = var

                cb = tk.Checkbutton(
                    grp_frame,
                    text=ctrl.label,
                    variable=var,
                    command=lambda k=ctrl.key: self._on_ctrl_toggle(k),
                    bg=_BG_GROUP, fg=_FG,
                    selectcolor=_BG_FRAME,
                    activebackground=_BG_GROUP,
                    activeforeground=_ACCENT,
                    font=("Segoe UI", int(9 * s)),
                    anchor="w",
                    bd=0, highlightthickness=0,
                )
                cb.pack(fill=tk.X, padx=(int(12 * s), 0))

            col_idx += 1
            if col_idx >= _CTRL_COLS:
                col_idx = 0
                row_idx += 1

    # ────────────────────────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────────────────────────

    def _on_toggle(self, key: str) -> None:
        """Reward checkbox was toggled — push to shared state."""
        var = self._check_vars.get(key)
        if var is not None:
            self.toggle_state.set_enabled(key, var.get())
        self._highlight_phase("custom")

    def _on_ctrl_toggle(self, key: str) -> None:
        """AI-control checkbox was toggled — push to shared state."""
        var = self._ctrl_check_vars.get(key)
        if var is not None:
            self.controls_state.set_enabled(key, var.get())

    def _on_set_home(self) -> None:
        """Set New Home button clicked — invoke the callback."""
        if self.on_set_home:
            self.on_set_home()

    def _set_all_controls(self, enabled: bool) -> None:
        """Enable or disable all AI controls and refresh checkboxes."""
        self.controls_state.set_all(enabled)
        for key, var in self._ctrl_check_vars.items():
            var.set(enabled)

    def _on_phase(self, preset_id: str) -> None:
        """Phase button clicked — apply preset and refresh checkboxes."""
        self.toggle_state.apply_preset(preset_id)
        self._sync_checkboxes()
        self._highlight_phase(preset_id)

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="\u25b6  Resume", bg=_PAUSE_ON, fg="#1e1e2e")
            self._release_cursor()
            # Sync with InputGuard: pause AI + unlock user controls
            if self._input_guard is not None:
                self._input_guard._ai_paused = True
                self._input_guard._kb_blocked = False
                self._input_guard._mouse_blocked = False
        else:
            self.btn_pause.config(text="\u23f8  Pause", bg=_BTN_BG, fg=_BTN_FG)
            # Sync with InputGuard: resume AI + re-lock user controls
            if self._input_guard is not None:
                self._input_guard._ai_paused = False
                self._input_guard._kb_blocked = True
                self._input_guard._mouse_blocked = True

    def trigger_stop(self) -> None:
        self.is_stopped = True
        self._release_cursor()
        if self.on_stop:
            self.on_stop()
        self._destroy()

    def request_close(self) -> None:
        """Ask the tkinter window to close from another thread.

        Sets a flag that ``_poll_updates`` checks on its next tick,
        guaranteeing the destroy runs on the tk event-loop thread
        regardless of whether ``root.after()`` works cross-thread.
        Also attempts the direct ``root.after`` approach as a fast-path.
        """
        self.is_stopped = True
        self._close_requested = True
        # Fast-path: try to schedule on tk thread directly.
        try:
            if self.root is not None:
                self.root.after(0, self._destroy)
        except Exception:
            pass

    def _destroy(self) -> None:
        """Destroy the root window (must run on the tk thread)."""
        try:
            if self.root is not None:
                self.root.quit()
                self.root.destroy()
                self.root = None
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────

    def _sync_checkboxes(self) -> None:
        """Refresh all checkbox vars from the shared toggle state."""
        snapshot = self.toggle_state.snapshot()
        for key, var in self._check_vars.items():
            var.set(snapshot.get(key, False))

    def _highlight_phase(self, active_id: str) -> None:
        """Visually highlight the active phase button."""
        for pid, btn in self._phase_buttons.items():
            if pid == active_id:
                btn.config(bg=_PHASE_SEL, fg="#1e1e2e")
            else:
                btn.config(bg=_PHASE_UNSEL, fg=_BTN_FG)

    def _poll_updates(self) -> None:
        """Periodic UI refresh (live stats + cursor release + auto-close + guard sync)."""
        if self.root is None:
            return

        # ── Auto-close when training loop signals shutdown ─────
        if self._close_requested:
            self._destroy()
            return

        # ── Sync pause state from InputGuard (Ctrl+P pressed) ──
        # If the user pressed Ctrl+P in-game we need to update
        # the UI button to reflect the new state.
        if self._input_guard is not None:
            guard_paused = self._input_guard._ai_paused
            if guard_paused != self.is_paused:
                self.is_paused = guard_paused
                if self.is_paused:
                    self.btn_pause.config(
                        text="\u25b6  Resume", bg=_PAUSE_ON, fg="#1e1e2e",
                    )
                else:
                    self.btn_pause.config(
                        text="\u23f8  Pause", bg=_BTN_BG, fg=_BTN_FG,
                    )

        # Update live reward/step display.
        if self._reward_label is not None:
            self._reward_label.config(text=f"Reward: {self._live_reward:+.2f}")
        if self._step_label is not None:
            self._step_label.config(text=f"Step: {self._live_step}")

        # Release cursor if paused/stopped.
        if self.is_paused or self.is_stopped:
            self._release_cursor()

        self.root.after(500, self._poll_updates)

    @staticmethod
    def _release_cursor() -> None:
        try:
            ctypes.windll.user32.ClipCursor(None)
        except Exception:
            pass

    @staticmethod
    def _sep(parent: tk.Misc) -> None:
        """Draw a thin horizontal separator."""
        tk.Frame(parent, height=1, bg=_ACCENT_DARK).pack(fill=tk.X, pady=4)
