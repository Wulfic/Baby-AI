"""
Reward-weights tab builder for the Baby-AI control panel.

Extracted from ``control_panel.py`` to keep individual modules
under ~800 lines.  All functions accept the panel instance  so
they can read/write its shared state dicts.
"""

from __future__ import annotations

import tkinter as tk
from collections import OrderedDict

from baby_ai.ui.reward_weights import (
    WEIGHT_CHILDREN,
    TOP_LEVEL_WEIGHTS,
)
from baby_ai.ui.theme import (
    BG, BG_GROUP, FG, FG_DIM, ACCENT, ACCENT_DARK, STOP_BG,
)


# ── Public API used by control_panel.AIControlPanel ────────────


def build_weights_tab(panel, parent: tk.Frame) -> None:
    """Build the reward weight sliders inside *parent*.

    Top-level weights are shown normally.  Weights that have
    sub-weights (children) get a clickable ``▸``/``▾`` toggle:
    clicking it reveals/hides the indented child sliders.

    Args:
        panel: The :class:`AIControlPanel` instance that owns the
            shared state dicts (``_weight_vars``, ``_weight_labels``,
            ``_sub_frames``, ``_expand_states``, ``reward_weights``).
        parent: The tkinter Frame to build inside.
    """
    s = panel._scale

    # ── Header with Reset Defaults button ──────────────────
    header = tk.Frame(parent, bg=BG)
    header.pack(fill=tk.X, pady=(int(4 * s), int(3 * s)))

    tk.Label(
        header, text="REWARD WEIGHT MULTIPLIERS",
        font=("Segoe UI", int(9 * s), "bold"),
        bg=BG, fg=FG_DIM, anchor="w",
    ).pack(side=tk.LEFT)

    tk.Button(
        header, text="Reset Defaults",
        command=panel._reset_weights_to_defaults,
        bg=STOP_BG, fg="#1e1e2e", activebackground="#eba0ac",
        font=("Segoe UI", int(8 * s), "bold"),
        relief="flat", bd=0, padx=int(6 * s), pady=int(2 * s),
    ).pack(side=tk.RIGHT, padx=(int(4 * s), 0))

    # ── Scrollable area ────────────────────────────────────
    canvas_frame = tk.Frame(parent, bg=BG)
    canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, int(4 * s)))

    canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0, bd=0)
    scrollbar = tk.Scrollbar(
        canvas_frame, orient="vertical", command=canvas.yview,
    )
    inner = tk.Frame(canvas, bg=BG)

    inner.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Mouse-wheel scrolling.
    def _on_mousewheel_weights(event: tk.Event) -> None:
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel_weights)

    # ── Build groups with sliders ──────────────────────────
    snapshot = panel.reward_weights.snapshot()

    grouped: OrderedDict[str, list] = OrderedDict()
    for w in TOP_LEVEL_WEIGHTS:
        grouped.setdefault(w.group, []).append(w)

    for group_name, weights in grouped.items():
        grp_frame = tk.Frame(
            inner, bg=BG_GROUP,
            padx=int(6 * s), pady=int(4 * s),
        )
        grp_frame.pack(fill=tk.X, padx=int(3 * s), pady=int(3 * s))

        # Group header
        tk.Label(
            grp_frame, text=group_name.upper(),
            font=("Segoe UI", int(8 * s), "bold"),
            bg=BG_GROUP, fg=FG_DIM, anchor="w",
        ).pack(fill=tk.X)

        for w in weights:
            has_children = w.key in WEIGHT_CHILDREN
            build_weight_row(panel, grp_frame, w, snapshot, indent=0, has_children=has_children)

            # If this parent has sub-weights, build them inside
            # a collapsible frame (starts collapsed).
            if has_children:
                sub_frame = tk.Frame(grp_frame, bg=BG_GROUP)
                # Initially hidden — don't pack yet.
                panel._sub_frames[w.key] = sub_frame
                panel._expand_states[w.key] = False

                for child in WEIGHT_CHILDREN[w.key]:
                    build_weight_row(panel, sub_frame, child, snapshot, indent=1, has_children=False)


def build_weight_row(
    panel,
    parent_frame: tk.Frame,
    w,  # WeightInfo
    snapshot: dict,
    *,
    indent: int = 0,
    has_children: bool = False,
) -> None:
    """Create a single label + slider + value row.

    ``indent`` > 0 indents the row (sub-weight).
    ``has_children`` adds a ``▸``/``▾`` expand toggle button.
    """
    s = panel._scale
    left_pad = int((4 + indent * 16) * s)

    row = tk.Frame(parent_frame, bg=BG_GROUP)
    row.pack(fill=tk.X, padx=(left_pad, 0), pady=(int(1 * s), 0))
    row.columnconfigure(2, weight=1)  # slider column stretches

    # Optional expand/collapse toggle (only on parents)
    if has_children:
        arrow_lbl = tk.Label(
            row, text="\u25b8",  # ▸ (collapsed)
            font=("Segoe UI", int(9 * s)),
            bg=BG_GROUP, fg=ACCENT, anchor="w",
            cursor="hand2",
        )
        arrow_lbl.grid(row=0, column=0, sticky="w", padx=(0, int(2 * s)))
        arrow_lbl.bind(
            "<Button-1>",
            lambda e, k=w.key, lbl=arrow_lbl: toggle_expand(panel, k, lbl),
        )
    else:
        # Spacer so child rows align with parent slider
        spacer_w = int(12 * s) if indent > 0 else 0
        if spacer_w > 0:
            tk.Label(
                row, text="\u2022",  # bullet
                font=("Segoe UI", int(7 * s)),
                bg=BG_GROUP, fg=FG_DIM, anchor="w",
                width=1,
            ).grid(row=0, column=0, sticky="w", padx=(0, int(2 * s)))

    # Label
    prefix = "\u2212 " if w.is_penalty else "+ "
    lbl_text = prefix + w.label
    tk.Label(
        row, text=lbl_text,
        font=("Segoe UI", int(8 * s)),
        bg=BG_GROUP, fg=FG, anchor="w",
    ).grid(row=0, column=1, sticky="w",
           padx=(0, int(4 * s)),
           ipadx=int(2 * s))

    # Current value label — right-aligned
    cur_val = snapshot.get(w.key, w.default)
    # Use more decimal places for very small step sizes
    fmt = "{:.2f}" if w.step < 0.05 else "{:.1f}"
    val_label = tk.Label(
        row, text=fmt.format(cur_val),
        font=("Consolas", int(9 * s)),
        bg=BG_GROUP, fg=ACCENT, anchor="e",
        width=5,
    )
    val_label.grid(row=0, column=3, sticky="e",
                   padx=(int(4 * s), 0))
    panel._weight_labels[w.key] = val_label

    # Scale (slider)
    var = tk.DoubleVar(value=cur_val)
    panel._weight_vars[w.key] = var

    slider = tk.Scale(
        row,
        from_=w.min_val,
        to=w.max_val,
        resolution=w.step,
        orient=tk.HORIZONTAL,
        variable=var,
        showvalue=False,
        command=lambda val, k=w.key: panel._on_weight_change(k, val),
        bg=ACCENT,               # thumb color
        fg=FG,
        troughcolor=ACCENT_DARK,  # visible groove
        activebackground="#b4d0fb", # thumb hover
        highlightthickness=0,
        bd=0,
        width=int(12 * s),         # thumb height
        sliderlength=int(16 * s),  # thumb width
        sliderrelief="raised",
    )
    slider.grid(row=0, column=2, sticky="ew",
                padx=(int(2 * s), int(2 * s)))


def toggle_expand(panel, parent_key: str, arrow_label: tk.Label) -> None:
    """Expand or collapse the sub-weight rows for *parent_key*."""
    expanded = panel._expand_states.get(parent_key, False)
    sub_frame = panel._sub_frames.get(parent_key)
    if sub_frame is None:
        return

    if expanded:
        # Collapse
        sub_frame.pack_forget()
        arrow_label.config(text="\u25b8")  # ▸
        panel._expand_states[parent_key] = False
    else:
        # Expand — pack right after the parent row.
        sub_frame.pack(fill=tk.X, padx=(int(4 * panel._scale), 0), pady=0)
        arrow_label.config(text="\u25be")  # ▾
        panel._expand_states[parent_key] = True
