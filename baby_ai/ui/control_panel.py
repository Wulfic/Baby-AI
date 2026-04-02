"""
Baby-AI Control Panel — tkinter UI for runtime control.

Provides:
- Pause / Stop buttons
- Set New Home button to update the agent's home location
- Per-channel reward toggle checkboxes in a multi-column grid
- AI Controls tab to enable/disable individual keys/buttons/look
- Reward Weights tab with sliders for each weight multiplier
- Live reward / step readout
- Persistent settings across runs via SettingsStore

Runs on a daemon thread so it doesn't block the training loop.
All shared state is mediated through :class:`RewardToggleState`,
:class:`AIControlsState`, and :class:`RewardWeightsState` which
are thread-safe.
"""

from __future__ import annotations

import ctypes
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from baby_ai.ui.reward_toggles import (
    CHANNELS,
    RewardToggleState,
)
from baby_ai.ui.controls_state import (
    AI_CONTROLS,
    AIControlsState,
)
from baby_ai.ui.reward_weights import RewardWeightsState
from baby_ai.ui.settings_store import SettingsStore
from baby_ai.ui.theme import (
    get_dpi_scale as _get_dpi_scale,
    BG as _BG, BG_FRAME as _BG_FRAME, BG_GROUP as _BG_GROUP,
    FG as _FG, FG_DIM as _FG_DIM,
    ACCENT as _ACCENT, ACCENT_DARK as _ACCENT_DARK,
    PAUSE_ON as _PAUSE_ON, STOP_BG as _STOP_BG,
    BTN_BG as _BTN_BG, BTN_FG as _BTN_FG,
    CHANNEL_COLS as _CHANNEL_COLS,
)
from baby_ai.ui.weights_tab import (
    build_weights_tab as _build_weights_tab_impl,
    toggle_expand as _toggle_expand_impl,
)

# ── Thread-safe learning rate holder ─────────────────────────
_DEFAULT_LR = 5e-5        # medium-low constant LR
_live_lr: float = _DEFAULT_LR
_lr_lock = threading.Lock()


def get_live_lr() -> float:
    """Return the current GUI learning rate (thread-safe)."""
    with _lr_lock:
        return _live_lr


def set_live_lr(value: float) -> None:
    """Set the live learning rate from the GUI (thread-safe)."""
    global _live_lr
    with _lr_lock:
        _live_lr = value


# ── Thread-safe imitation learning (distillation) toggle ─────
_imitation_enabled: bool = True
_imitation_lock = threading.Lock()


def get_imitation_enabled() -> bool:
    """Return whether imitation learning (distillation) is enabled (thread-safe)."""
    with _imitation_lock:
        return _imitation_enabled


def set_imitation_enabled(value: bool) -> None:
    """Enable or disable imitation learning from the GUI (thread-safe)."""
    global _imitation_enabled
    with _imitation_lock:
        _imitation_enabled = value


# ── Thread-safe record-only mode toggle ──────────────────────
# When active: AI inference is skipped, user controls are fully
# unlocked, and only replay data is recorded.  Training threads
# are paused so the GPU is free and latency is minimal.
_record_only: bool = False
_record_only_lock = threading.Lock()


def get_record_only() -> bool:
    """Return whether record-only mode is active (thread-safe)."""
    with _record_only_lock:
        return _record_only


def set_record_only(value: bool) -> None:
    """Enable or disable record-only mode from the GUI (thread-safe)."""
    global _record_only
    with _record_only_lock:
        _record_only = value


# ── Thread-safe learning-disabled toggle ─────────────────────
# When active: AI inference continues normally but the learner
# and distill threads pause — no gradient updates at all.
_learning_disabled: bool = False
_learning_disabled_lock = threading.Lock()


def get_learning_disabled() -> bool:
    """Return whether learning is disabled (thread-safe)."""
    with _learning_disabled_lock:
        return _learning_disabled


def set_learning_disabled(value: bool) -> None:
    """Enable or disable learning from the GUI (thread-safe)."""
    global _learning_disabled
    with _learning_disabled_lock:
        _learning_disabled = value


# Theme constants and DPI helper are now imported from baby_ai.ui.theme.


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
    reward_weights : RewardWeightsState, optional
        Shared reward weight multipliers.  If ``None`` a private
        instance is created.
    on_set_home : callable, optional
        Invoked when the user clicks *Set New Home*.  Should grab the
        current player coordinates and update the env's home location.
    on_set_home_coords : callable, optional
        Invoked when the user enters manual home coordinates via the
        GUI.  Signature: ``(x: float, y: float, z: float) -> None``.
    input_guard : object, optional
        Reference to the :class:`InputGuard` so the Pause button can
        toggle keyboard/mouse blocking in sync.
    settings_store : SettingsStore, optional
        Persistent settings.  If ``None`` a private instance is created.
    """

    def __init__(
        self,
        on_stop: Optional[Callable[[], None]] = None,
        toggle_state: Optional[RewardToggleState] = None,
        controls_state: Optional[AIControlsState] = None,
        reward_weights: Optional[RewardWeightsState] = None,
        on_set_home: Optional[Callable[[], None]] = None,
        on_set_home_coords: Optional[Callable] = None,
        input_guard: Optional[object] = None,
        settings_store: Optional[SettingsStore] = None,
    ):
        self.is_paused: bool = False
        self.is_stopped: bool = False
        self.on_stop = on_stop
        self.on_set_home = on_set_home
        self.on_set_home_coords = on_set_home_coords
        self._input_guard = input_guard

        # Settings persistence
        self._store: SettingsStore = (
            settings_store if settings_store is not None else SettingsStore()
        )

        # Shared state for reward toggles.
        self.toggle_state: RewardToggleState = (
            toggle_state if toggle_state is not None else RewardToggleState()
        )

        # Shared state for AI input controls.
        self.controls_state: AIControlsState = (
            controls_state if controls_state is not None else AIControlsState()
        )

        # Shared state for reward weight multipliers.
        self.reward_weights: RewardWeightsState = (
            reward_weights if reward_weights is not None else RewardWeightsState()
        )

        # ── Restore persisted settings ─────────────────────────
        self._load_persisted_settings()

        self.root: Optional[tk.Tk] = None
        self._check_vars: dict[str, tk.BooleanVar] = {}
        self._ctrl_check_vars: dict[str, tk.BooleanVar] = {}
        self._weight_vars: dict[str, tk.DoubleVar] = {}
        self._weight_labels: dict[str, tk.Label] = {}
        # Track expand/collapse state for parent weights with sub-weights.
        # key = parent weight key, value = bool (True = expanded).
        self._expand_states: dict[str, bool] = {}
        # Frames that hold the sub-weight rows — shown/hidden on toggle.
        self._sub_frames: dict[str, tk.Frame] = {}
        self._reward_label: Optional[tk.Label] = None
        self._step_label: Optional[tk.Label] = None

        self._live_reward: float = 0.0
        self._live_step: int = 0

        # Learning rate — restore persisted value, else default.
        saved_lr = self._store.get("learning_rate")
        if saved_lr is not None:
            try:
                set_live_lr(float(saved_lr))
            except (ValueError, TypeError):
                pass

        # Flag checked by _poll_updates to self-destruct when the
        # training loop signals shutdown from another thread.
        self._close_requested: bool = False

        # Reference to the daemon thread running tkinter mainloop.
        self._thread: Optional[threading.Thread] = None

        # tk variables for LR (initialised in _build_controls_tab)
        self._lr_var: Optional[tk.DoubleVar] = None
        self._lr_label: Optional[tk.Label] = None

        # tk variable for imitation learning toggle
        self._imit_var: Optional[tk.BooleanVar] = None

        # tk variable for record-only mode toggle
        self._record_only_var: Optional[tk.BooleanVar] = None

        # tk variable for disable-learning toggle
        self._learning_disabled_var: Optional[tk.BooleanVar] = None

        # Home waypoint coordinate labels (updated when home changes)
        self._home_x_label: Optional[tk.Label] = None
        self._home_y_label: Optional[tk.Label] = None
        self._home_z_label: Optional[tk.Label] = None
        self._home_x_entry: Optional[tk.Entry] = None
        self._home_y_entry: Optional[tk.Entry] = None
        self._home_z_entry: Optional[tk.Entry] = None

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the UI on a daemon thread."""
        self._thread = threading.Thread(target=self._run_ui, daemon=True)
        self._thread.start()

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

        # ── Hotkey hints strip ─────────────────────────────────
        hints_bar = tk.Frame(self.root, bg=_BG)
        hints_bar.pack(fill=tk.X, pady=(0, int(4 * s)))
        tk.Label(
            hints_bar,
            text="Ctrl+P  Pause/Resume  ·  Ctrl+Q  Stop & Save  ·  Ctrl+H  Set Home",
            font=("Segoe UI", int(8 * s)),
            bg=_BG, fg=_FG_DIM, anchor="w",
        ).pack(side=tk.LEFT)

        # ── Keyboard shortcuts ─────────────────────────────────
        self.root.bind_all("<Control-p>", lambda e: self.toggle_pause())
        self.root.bind_all("<Control-q>", lambda e: self.trigger_stop())
        self.root.bind_all("<Control-h>", lambda e: self._on_set_home())

        # ── Home Waypoint section ─────────────────────────────
        home_frame = tk.Frame(self.root, bg=_BG_GROUP,
                              padx=int(6 * s), pady=int(4 * s))
        home_frame.pack(fill=tk.X, pady=(int(2 * s), int(2 * s)))

        home_title_row = tk.Frame(home_frame, bg=_BG_GROUP)
        home_title_row.pack(fill=tk.X)

        tk.Label(
            home_title_row, text="\U0001f3e0 HOME WAYPOINT",
            font=("Segoe UI", int(8 * s), "bold"),
            bg=_BG_GROUP, fg=_FG_DIM, anchor="w",
        ).pack(side=tk.LEFT)

        # Apply button for manual coordinate entry
        tk.Button(
            home_title_row, text="Apply",
            command=self._on_apply_home_coords,
            bg=_ACCENT, fg="#1e1e2e", activebackground=_ACCENT_DARK,
            font=("Segoe UI", int(7 * s), "bold"),
            relief="flat", bd=0, padx=int(6 * s), pady=int(1 * s),
        ).pack(side=tk.RIGHT, padx=(int(4 * s), 0))

        # Coordinate display/entry row
        coord_row = tk.Frame(home_frame, bg=_BG_GROUP)
        coord_row.pack(fill=tk.X, pady=(int(2 * s), 0))

        # Restore saved home or show placeholder
        saved_home = self._store.get("home_location")
        hx_text = f"{saved_home['x']:.1f}" if saved_home and saved_home.get('x') is not None else "—"
        hy_text = f"{saved_home['y']:.1f}" if saved_home and saved_home.get('y') is not None else "—"
        hz_text = f"{saved_home['z']:.1f}" if saved_home and saved_home.get('z') is not None else "—"

        _entry_font = ("Consolas", int(9 * s))
        _label_font = ("Segoe UI", int(8 * s), "bold")
        _entry_width = 8

        tk.Label(coord_row, text="X:", font=_label_font,
                 bg=_BG_GROUP, fg=_FG).pack(side=tk.LEFT, padx=(0, int(2 * s)))
        self._home_x_entry = tk.Entry(
            coord_row, width=_entry_width, font=_entry_font,
            bg=_BG_FRAME, fg=_ACCENT, insertbackground=_ACCENT,
            relief="flat", bd=1, highlightthickness=1,
            highlightbackground=_ACCENT_DARK, highlightcolor=_ACCENT,
        )
        self._home_x_entry.insert(0, hx_text)
        self._home_x_entry.pack(side=tk.LEFT, padx=(0, int(6 * s)))

        tk.Label(coord_row, text="Y:", font=_label_font,
                 bg=_BG_GROUP, fg=_FG).pack(side=tk.LEFT, padx=(0, int(2 * s)))
        self._home_y_entry = tk.Entry(
            coord_row, width=_entry_width, font=_entry_font,
            bg=_BG_FRAME, fg=_ACCENT, insertbackground=_ACCENT,
            relief="flat", bd=1, highlightthickness=1,
            highlightbackground=_ACCENT_DARK, highlightcolor=_ACCENT,
        )
        self._home_y_entry.insert(0, hy_text)
        self._home_y_entry.pack(side=tk.LEFT, padx=(0, int(6 * s)))

        tk.Label(coord_row, text="Z:", font=_label_font,
                 bg=_BG_GROUP, fg=_FG).pack(side=tk.LEFT, padx=(0, int(2 * s)))
        self._home_z_entry = tk.Entry(
            coord_row, width=_entry_width, font=_entry_font,
            bg=_BG_FRAME, fg=_ACCENT, insertbackground=_ACCENT,
            relief="flat", bd=1, highlightthickness=1,
            highlightbackground=_ACCENT_DARK, highlightcolor=_ACCENT,
        )
        self._home_z_entry.insert(0, hz_text)
        self._home_z_entry.pack(side=tk.LEFT)

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

        # ── Tab 3: Reward Weights ──────────────────────────────
        weights_frame = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(weights_frame, text="  Reward Weights  ")
        self._build_weights_tab(weights_frame)

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

        # ── Imitation Learning (Distillation) toggle ───────────
        imit_frame = tk.Frame(parent, bg=_BG_GROUP,
                              padx=int(6 * s), pady=int(4 * s))
        imit_frame.pack(fill=tk.X, padx=int(3 * s), pady=(int(6 * s), int(3 * s)))

        tk.Label(
            imit_frame, text="TRAINING OPTIONS",
            font=("Segoe UI", int(8 * s), "bold"),
            bg=_BG_GROUP, fg=_FG_DIM, anchor="w",
        ).pack(fill=tk.X)

        # Restore persisted value, default to True (enabled).
        saved_imit = self._store.get("imitation_learning")
        imit_default = bool(saved_imit) if saved_imit is not None else True
        set_imitation_enabled(imit_default)

        self._imit_var = tk.BooleanVar(value=imit_default)

        def _on_imit_toggle() -> None:
            enabled = self._imit_var.get()
            set_imitation_enabled(enabled)
            self._store.set("imitation_learning", enabled)

        imit_cb = tk.Checkbutton(
            imit_frame,
            text="Imitation Learning (Teacher → Student Distillation)",
            variable=self._imit_var,
            command=_on_imit_toggle,
            bg=_BG_GROUP, fg=_FG,
            selectcolor=_BG_FRAME,
            activebackground=_BG_GROUP,
            activeforeground=_ACCENT,
            font=("Segoe UI", int(9 * s)),
            anchor="w",
            bd=0, highlightthickness=0,
        )
        imit_cb.pack(fill=tk.X, padx=(int(12 * s), 0))

        # ── Record-Only mode toggle ────────────────────────────
        saved_rec = self._store.get("record_only")
        rec_default = bool(saved_rec) if saved_rec is not None else False
        set_record_only(rec_default)

        self._record_only_var = tk.BooleanVar(value=rec_default)

        def _on_record_only_toggle() -> None:
            enabled = self._record_only_var.get()
            set_record_only(enabled)
            self._store.set("record_only", enabled)

        rec_cb = tk.Checkbutton(
            imit_frame,
            text="Record Only (save replay data, no AI)",
            variable=self._record_only_var,
            command=_on_record_only_toggle,
            bg=_BG_GROUP, fg=_FG,
            selectcolor=_BG_FRAME,
            activebackground=_BG_GROUP,
            activeforeground=_ACCENT,
            font=("Segoe UI", int(9 * s)),
            anchor="w",
            bd=0, highlightthickness=0,
        )
        rec_cb.pack(fill=tk.X, padx=(int(12 * s), 0))

        # ── Disable Learning toggle ────────────────────────────
        saved_ld = self._store.get("learning_disabled")
        ld_default = bool(saved_ld) if saved_ld is not None else False
        set_learning_disabled(ld_default)

        self._learning_disabled_var = tk.BooleanVar(value=ld_default)

        def _on_learning_disabled_toggle() -> None:
            enabled = self._learning_disabled_var.get()
            set_learning_disabled(enabled)
            self._store.set("learning_disabled", enabled)

        ld_cb = tk.Checkbutton(
            imit_frame,
            text="Disable Learning (AI runs, no weight updates)",
            variable=self._learning_disabled_var,
            command=_on_learning_disabled_toggle,
            bg=_BG_GROUP, fg=_FG,
            selectcolor=_BG_FRAME,
            activebackground=_BG_GROUP,
            activeforeground=_ACCENT,
            font=("Segoe UI", int(9 * s)),
            anchor="w",
            bd=0, highlightthickness=0,
        )
        ld_cb.pack(fill=tk.X, padx=(int(12 * s), 0))

        # ── Learning Rate slider ───────────────────────────────
        lr_frame = tk.Frame(parent, bg=_BG_GROUP,
                            padx=int(6 * s), pady=int(4 * s))
        lr_frame.pack(fill=tk.X, padx=int(3 * s), pady=(int(6 * s), int(3 * s)))

        tk.Label(
            lr_frame, text="LEARNING RATE",
            font=("Segoe UI", int(8 * s), "bold"),
            bg=_BG_GROUP, fg=_FG_DIM, anchor="w",
        ).pack(fill=tk.X)

        lr_row = tk.Frame(lr_frame, bg=_BG_GROUP)
        lr_row.pack(fill=tk.X, pady=(int(2 * s), 0))
        lr_row.columnconfigure(1, weight=1)

        tk.Label(
            lr_row, text="LR",
            font=("Segoe UI", int(8 * s)),
            bg=_BG_GROUP, fg=_FG, anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, int(4 * s)))

        cur_lr = get_live_lr()
        self._lr_var = tk.DoubleVar(value=cur_lr)
        self._lr_label = tk.Label(
            lr_row, text=f"{cur_lr:.1e}",
            font=("Consolas", int(9 * s)),
            bg=_BG_GROUP, fg=_ACCENT, anchor="e", width=8,
        )
        self._lr_label.grid(row=0, column=2, sticky="e", padx=(int(4 * s), 0))

        # Log-scale mapping: slider goes 1..100, mapped to 1e-6..1e-3
        # via  lr = 10 ^ (slider_val / 100 * 3 - 6)
        #   slider=0  → 1e-6,  slider=50 → ~3.2e-5,  slider=100 → 1e-3
        def _lr_from_slider(v: float) -> float:
            return 10.0 ** (v / 100.0 * 3.0 - 6.0)

        def _slider_from_lr(lr: float) -> float:
            import math
            lr = max(lr, 1e-7)
            return (math.log10(lr) + 6.0) / 3.0 * 100.0

        def _on_lr_slide(val: str) -> None:
            lr = _lr_from_slider(float(val))
            set_live_lr(lr)
            if self._lr_label is not None:
                self._lr_label.config(text=f"{lr:.1e}")
            self._store.set("learning_rate", lr)

        lr_slider = tk.Scale(
            lr_row,
            from_=0, to=100, resolution=1,
            orient=tk.HORIZONTAL,
            showvalue=False,
            command=_on_lr_slide,
            bg=_ACCENT, fg=_FG,
            troughcolor=_ACCENT_DARK,
            activebackground="#b4d0fb",
            highlightthickness=0, bd=0,
            width=int(12 * s), sliderlength=int(16 * s),
            sliderrelief="raised",
        )
        lr_slider.set(int(_slider_from_lr(cur_lr)))
        lr_slider.grid(row=0, column=1, sticky="ew", padx=(int(2 * s), int(2 * s)))

    # ────────────────────────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────────────────────────

    def _on_toggle(self, key: str) -> None:
        """Reward checkbox was toggled — push to shared state + persist."""
        var = self._check_vars.get(key)
        if var is not None:
            self.toggle_state.set_enabled(key, var.get())
            self._persist_reward_toggles()

    def _on_ctrl_toggle(self, key: str) -> None:
        """AI-control checkbox was toggled — push to shared state + persist."""
        var = self._ctrl_check_vars.get(key)
        if var is not None:
            self.controls_state.set_enabled(key, var.get())
            self._persist_ai_controls()

    def _on_weight_change(self, key: str, value: str) -> None:
        """Reward weight slider was moved — push to shared state + persist."""
        try:
            fval = round(float(value), 2)
        except (ValueError, TypeError):
            return
        self.reward_weights.set_weight(key, fval)
        # Update the value label next to the slider.
        lbl = self._weight_labels.get(key)
        if lbl is not None:
            from baby_ai.ui.reward_weights import WEIGHT_MAP
            winfo = WEIGHT_MAP.get(key)
            fmt = "{:.2f}" if (winfo and winfo.step < 0.05) else "{:.1f}"
            lbl.config(text=fmt.format(fval))
        self._persist_reward_weights()

    def _on_set_home(self) -> None:
        """Set New Home button clicked — invoke the callback."""
        if self.on_set_home:
            self.on_set_home()

    def _on_apply_home_coords(self) -> None:
        """Apply manually-entered home coordinates from the GUI fields."""
        try:
            x_text = self._home_x_entry.get().strip() if self._home_x_entry else ""
            y_text = self._home_y_entry.get().strip() if self._home_y_entry else ""
            z_text = self._home_z_entry.get().strip() if self._home_z_entry else ""

            if not x_text or not z_text or x_text == "\u2014" or z_text == "\u2014":
                return

            x = float(x_text)
            y = float(y_text) if y_text and y_text != "\u2014" else 64.0
            z = float(z_text)

            if self.on_set_home_coords:
                self.on_set_home_coords(x, y, z)
        except (ValueError, TypeError):
            pass  # Invalid coordinate input — silently ignore

    def update_home_display(self, x: float, y: float, z: float) -> None:
        """Update the home coordinate entry fields from another thread.

        Called when the home location changes via /sethome or the
        Set New Home button, so the GUI reflects the current value.
        """
        def _update():
            if self._home_x_entry is not None:
                self._home_x_entry.delete(0, tk.END)
                self._home_x_entry.insert(0, f"{x:.1f}")
            if self._home_y_entry is not None:
                self._home_y_entry.delete(0, tk.END)
                self._home_y_entry.insert(0, f"{y:.1f}")
            if self._home_z_entry is not None:
                self._home_z_entry.delete(0, tk.END)
                self._home_z_entry.insert(0, f"{z:.1f}")

        # Schedule on the tk thread to be safe.
        try:
            if self.root is not None:
                self.root.after(0, _update)
        except Exception:
            pass

    def _set_all_controls(self, enabled: bool) -> None:
        """Enable or disable all AI controls and refresh checkboxes."""
        self.controls_state.set_all(enabled)
        for key, var in self._ctrl_check_vars.items():
            var.set(enabled)
        self._persist_ai_controls()

    def _reset_weights_to_defaults(self) -> None:
        """Reset all reward weights to their default values."""
        self.reward_weights.reset_defaults()
        snap = self.reward_weights.snapshot()
        from baby_ai.ui.reward_weights import WEIGHT_MAP
        for key, var in self._weight_vars.items():
            val = snap.get(key, 0.0)
            var.set(val)
            lbl = self._weight_labels.get(key)
            if lbl is not None:
                winfo = WEIGHT_MAP.get(key)
                fmt = "{:.2f}" if (winfo and winfo.step < 0.05) else "{:.1f}"
                lbl.config(text=fmt.format(val))
        self._persist_reward_weights()

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
        # Persist all settings before closing.
        self._persist_reward_toggles()
        self._persist_ai_controls()
        self._persist_reward_weights()
        if self.on_stop:
            self.on_stop()
        self._destroy()

    def request_close(self) -> None:
        """Ask the tkinter window to close from another thread.

        Sets a flag that ``_poll_updates`` checks on its next tick,
        guaranteeing the destroy runs on the tk event-loop thread
        regardless of whether ``root.after()`` works cross-thread.
        Also attempts the direct ``root.after`` approach as a fast-path.

        After signalling, **joins the UI thread** so that all Tcl
        resources are freed on the thread that created them before
        the caller (main thread) continues to exit — preventing
        ``Tcl_AsyncDelete: async handler deleted by the wrong thread``.
        """
        self.is_stopped = True
        self._close_requested = True
        # Fast-path: try to schedule on tk thread directly.
        try:
            if self.root is not None:
                self.root.after(0, self._destroy)
        except Exception:
            pass
        # Wait for the tk thread to finish so Tcl cleanup happens
        # on the correct thread before the process tears down.
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join(timeout=3.0)

    def _destroy(self) -> None:
        """Destroy the root window (must run on the tk thread).

        Clears all tk variable references first so their __del__
        methods don't fire from the wrong thread during GC.
        """
        try:
            # Drop all tk variable refs before destroying root —
            # prevents "main thread is not in main loop" RuntimeError
            # when Python's GC collects them on the main thread.
            self._check_vars.clear()
            self._ctrl_check_vars.clear()
            self._weight_vars.clear()
            self._weight_labels.clear()
            self._sub_frames.clear()
            self._expand_states.clear()
            self._reward_label = None
            self._step_label = None
            self._lr_var = None
            self._lr_label = None
            self._imit_var = None
            self._learning_disabled_var = None
            self._home_x_entry = None
            self._home_y_entry = None
            self._home_z_entry = None
            if self.root is not None:
                self.root.quit()
                self.root.destroy()
                self.root = None
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────
    # Persistence helpers
    # ────────────────────────────────────────────────────────────

    def _load_persisted_settings(self) -> None:
        """Restore all shared state objects from the settings store."""
        # Reward toggles
        saved_toggles = self._store.get("reward_toggles")
        if saved_toggles and isinstance(saved_toggles, dict):
            for key, enabled in saved_toggles.items():
                self.toggle_state.set_enabled(key, bool(enabled))

        # AI controls
        saved_controls = self._store.get("ai_controls")
        if saved_controls and isinstance(saved_controls, dict):
            for key, enabled in saved_controls.items():
                self.controls_state.set_enabled(key, bool(enabled))

        # Reward weights
        saved_weights = self._store.get("reward_weights")
        if saved_weights and isinstance(saved_weights, dict):
            self.reward_weights.set_all(saved_weights)

    def _persist_reward_toggles(self) -> None:
        """Save current reward toggle state to disk."""
        self._store.set("reward_toggles", self.toggle_state.snapshot())

    def _persist_ai_controls(self) -> None:
        """Save current AI controls state to disk."""
        self._store.set("ai_controls", self.controls_state.snapshot())

    def _persist_reward_weights(self) -> None:
        """Save current reward weights to disk."""
        self._store.set("reward_weights", self.reward_weights.snapshot())

    # ────────────────────────────────────────────────────────────
    # Tab builder: Reward Weights (delegated to weights_tab module)
    # ────────────────────────────────────────────────────────────

    def _build_weights_tab(self, parent: tk.Frame) -> None:
        """Build the reward weight sliders inside *parent*."""
        _build_weights_tab_impl(self, parent)

    def _toggle_expand(self, parent_key: str, arrow_label: tk.Label) -> None:
        """Expand or collapse the sub-weight rows for *parent_key*."""
        _toggle_expand_impl(self, parent_key, arrow_label)

    # ────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────

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
