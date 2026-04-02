"""
Model Configuration tab for the Baby-AI control panel.

Lets users adjust Student and Teacher model dimensions so the
architecture can be scaled to fit available GPU memory.  Changes
are persisted to ``baby_ai_settings.json`` and take effect on
next restart (models are constructed once at startup).
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import fields
from typing import Any, Dict, List, Tuple

from baby_ai.config import (
    BabyAIConfig,
    EncoderConfig,
    JambaConfig,
    StudentConfig,
    TeacherConfig,
)
from baby_ai.ui.theme import (
    BG, BG_GROUP, FG, FG_DIM, ACCENT, ACCENT_DARK, STOP_BG, BTN_BG, BTN_FG,
)


# ── Param descriptors ──────────────────────────────────────────

class _Param:
    """Describes a single configurable model parameter."""
    __slots__ = ("key", "label", "tooltip", "lo", "hi", "step", "type_")

    def __init__(
        self,
        key: str,
        label: str,
        tooltip: str,
        lo: float,
        hi: float,
        step: float,
        type_: type = int,
    ):
        self.key = key
        self.label = label
        self.tooltip = tooltip
        self.lo = lo
        self.hi = hi
        self.step = step
        self.type_ = type_


# Parameters exposed per-model.  Keys use dots to map into the
# nested dataclass hierarchy, e.g. "encoder.vision_embed_dim".
_ENCODER_PARAMS: List[_Param] = [
    _Param("encoder.vision_embed_dim", "Vision Embed", "Dimension of vision encoder output", 64, 1024, 64),
    _Param("encoder.audio_embed_dim", "Audio Embed", "Dimension of audio encoder output", 64, 1024, 64),
    _Param("encoder.sensor_embed_dim", "Sensor Embed", "Dimension of sensor encoder output", 32, 512, 32),
    _Param("encoder.fused_dim", "Fused Dim", "Multimodal fusion output dimension", 128, 2048, 128),
]

_CORE_PARAMS: List[_Param] = [
    _Param("hidden_dim", "Hidden Dim", "Jamba temporal core hidden dimension", 128, 2048, 128),
    _Param("jamba.num_layers", "Jamba Layers", "Stacked Jamba blocks", 1, 12, 1),
    _Param("jamba.d_state", "SSM State Dim", "Mamba recurrent state size", 4, 64, 4),
    _Param("jamba.expand", "Mamba Expand", "Inner dimension multiplier", 1, 4, 1),
    _Param("jamba.num_experts", "MoE Experts", "Experts per MoE layer", 1, 16, 1),
    _Param("jamba.top_k_routing", "MoE Top-K", "Experts activated per token", 1, 4, 1),
    _Param("jamba.ffn_mult", "FFN Multiplier", "FFN hidden dimension multiplier", 1, 4, 1),
]

# ── Presets ─────────────────────────────────────────────────────

# Each preset is (student_overrides, teacher_overrides).
# Overrides use the same dotted-key convention as _Param.key.
PRESETS: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {
    "Tiny (4-6 GB VRAM)": (
        {
            "encoder.vision_embed_dim": 128, "encoder.audio_embed_dim": 128,
            "encoder.sensor_embed_dim": 64, "encoder.fused_dim": 256,
            "hidden_dim": 256, "policy_hidden": 256,
            "jamba.num_layers": 2, "jamba.d_state": 8, "jamba.expand": 1,
            "jamba.num_experts": 2, "jamba.top_k_routing": 1, "jamba.ffn_mult": 1,
        },
        {
            "encoder.vision_embed_dim": 256, "encoder.audio_embed_dim": 256,
            "encoder.sensor_embed_dim": 128, "encoder.fused_dim": 512,
            "hidden_dim": 512, "policy_hidden": 512,
            "jamba.num_layers": 2, "jamba.d_state": 8, "jamba.expand": 1,
            "jamba.num_experts": 4, "jamba.top_k_routing": 1, "jamba.ffn_mult": 1,
        },
    ),
    "Small (6-8 GB VRAM)": (
        {
            "encoder.vision_embed_dim": 192, "encoder.audio_embed_dim": 192,
            "encoder.sensor_embed_dim": 96, "encoder.fused_dim": 384,
            "hidden_dim": 384, "policy_hidden": 384,
            "jamba.num_layers": 3, "jamba.d_state": 16, "jamba.expand": 2,
            "jamba.num_experts": 4, "jamba.top_k_routing": 1, "jamba.ffn_mult": 2,
        },
        {
            "encoder.vision_embed_dim": 384, "encoder.audio_embed_dim": 384,
            "encoder.sensor_embed_dim": 192, "encoder.fused_dim": 768,
            "hidden_dim": 768, "policy_hidden": 768,
            "jamba.num_layers": 3, "jamba.d_state": 16, "jamba.expand": 1,
            "jamba.num_experts": 6, "jamba.top_k_routing": 2, "jamba.ffn_mult": 1,
        },
    ),
    "Default (8-12 GB VRAM)": (
        {
            "encoder.vision_embed_dim": 256, "encoder.audio_embed_dim": 256,
            "encoder.sensor_embed_dim": 128, "encoder.fused_dim": 512,
            "hidden_dim": 512, "policy_hidden": 512,
            "jamba.num_layers": 4, "jamba.d_state": 16, "jamba.expand": 2,
            "jamba.num_experts": 4, "jamba.top_k_routing": 1, "jamba.ffn_mult": 2,
        },
        {
            "encoder.vision_embed_dim": 512, "encoder.audio_embed_dim": 512,
            "encoder.sensor_embed_dim": 256, "encoder.fused_dim": 1024,
            "hidden_dim": 1024, "policy_hidden": 1024,
            "jamba.num_layers": 4, "jamba.d_state": 16, "jamba.expand": 1,
            "jamba.num_experts": 8, "jamba.top_k_routing": 2, "jamba.ffn_mult": 1,
        },
    ),
    "Large (16+ GB VRAM)": (
        {
            "encoder.vision_embed_dim": 384, "encoder.audio_embed_dim": 384,
            "encoder.sensor_embed_dim": 192, "encoder.fused_dim": 768,
            "hidden_dim": 768, "policy_hidden": 768,
            "jamba.num_layers": 6, "jamba.d_state": 32, "jamba.expand": 2,
            "jamba.num_experts": 8, "jamba.top_k_routing": 2, "jamba.ffn_mult": 2,
        },
        {
            "encoder.vision_embed_dim": 768, "encoder.audio_embed_dim": 768,
            "encoder.sensor_embed_dim": 384, "encoder.fused_dim": 1536,
            "hidden_dim": 1536, "policy_hidden": 1536,
            "jamba.num_layers": 6, "jamba.d_state": 32, "jamba.expand": 2,
            "jamba.num_experts": 12, "jamba.top_k_routing": 2, "jamba.ffn_mult": 2,
        },
    ),
}


# ── Helpers ─────────────────────────────────────────────────────

def _get_nested(obj: object, dotted_key: str) -> Any:
    """Resolve ``'encoder.vision_embed_dim'`` on a dataclass."""
    for part in dotted_key.split("."):
        obj = getattr(obj, part)
    return obj


def _set_nested(obj: object, dotted_key: str, value: Any) -> None:
    """Set ``'encoder.vision_embed_dim'`` on a dataclass."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def snapshot_model_config(cfg: BabyAIConfig) -> Dict[str, Any]:
    """Read current model params into a flat dict for persistence."""
    data: Dict[str, Any] = {}
    for prefix, model_cfg in [("student", cfg.student), ("teacher", cfg.teacher)]:
        for p in _ENCODER_PARAMS + _CORE_PARAMS:
            data[f"{prefix}.{p.key}"] = _get_nested(model_cfg, p.key)
        # Also capture policy_hidden which scales with hidden_dim
        data[f"{prefix}.policy_hidden"] = model_cfg.policy_hidden
    return data


def apply_model_config(cfg: BabyAIConfig, data: Dict[str, Any]) -> None:
    """Apply a flat dict of model params onto a BabyAIConfig."""
    for key, value in data.items():
        parts = key.split(".", 1)
        if len(parts) != 2:
            continue
        prefix, param_key = parts
        if prefix == "student":
            _set_nested(cfg.student, param_key, value)
        elif prefix == "teacher":
            _set_nested(cfg.teacher, param_key, value)


# ── Tab builder ─────────────────────────────────────────────────

def build_model_tab(panel, parent: tk.Frame) -> None:
    """Build the Model Configuration tab inside *parent*.

    Args:
        panel: The :class:`AIControlPanel` instance.
        parent: The tkinter Frame to build inside.
    """
    s = panel._scale

    # Track tk variables so we can read them later.
    # key → DoubleVar, e.g. "student.encoder.vision_embed_dim"
    panel._model_vars = {}
    panel._model_labels = {}

    # ── Restart warning banner ─────────────────────────────
    warn = tk.Frame(parent, bg="#4a3000")
    warn.pack(fill=tk.X, pady=(int(4 * s), 0), padx=int(3 * s))
    tk.Label(
        warn,
        text="\u26A0  Changes take effect on next restart.  "
             "Existing checkpoints may not load if shapes changed.",
        font=("Segoe UI", int(8 * s)),
        bg="#4a3000", fg="#ffd866", anchor="w",
        wraplength=int(420 * s),
    ).pack(fill=tk.X, padx=int(6 * s), pady=int(3 * s))

    # ── Presets row ────────────────────────────────────────
    preset_frame = tk.Frame(parent, bg=BG)
    preset_frame.pack(fill=tk.X, padx=int(3 * s), pady=(int(4 * s), int(2 * s)))

    tk.Label(
        preset_frame, text="PRESET:",
        font=("Segoe UI", int(8 * s), "bold"),
        bg=BG, fg=FG_DIM,
    ).pack(side=tk.LEFT, padx=(0, int(4 * s)))

    for name in PRESETS:
        tk.Button(
            preset_frame, text=name,
            command=lambda n=name: _apply_preset(panel, n),
            bg=BTN_BG, fg=BTN_FG, activebackground=ACCENT,
            activeforeground="#1e1e2e",
            font=("Segoe UI", int(7 * s)),
            relief="flat", bd=0, padx=int(6 * s), pady=int(2 * s),
        ).pack(side=tk.LEFT, padx=int(2 * s))

    # ── Scrollable area ────────────────────────────────────
    canvas_frame = tk.Frame(parent, bg=BG)
    canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, int(4 * s)))

    canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0, bd=0)
    scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas, bg=BG)

    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_mousewheel(event: tk.Event) -> None:
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    # ── Build sections ─────────────────────────────────────
    from baby_ai.config import DEFAULT_CONFIG
    saved = panel._store.get("model_config")

    for prefix, label, model_cfg in [
        ("student", "Student", DEFAULT_CONFIG.student),
        ("teacher", "Teacher", DEFAULT_CONFIG.teacher),
    ]:
        _build_model_section(panel, inner, prefix, label, model_cfg, saved, s)


def _build_model_section(
    panel,
    parent: tk.Frame,
    prefix: str,
    title: str,
    model_cfg: object,
    saved: Dict[str, Any] | None,
    s: float,
) -> None:
    """Build sliders for one model (student or teacher)."""
    # ── Section header ─────────────────────────────────────
    header = tk.Frame(parent, bg=ACCENT_DARK)
    header.pack(fill=tk.X, padx=int(3 * s), pady=(int(6 * s), int(2 * s)))
    tk.Label(
        header, text=f"  {title.upper()} MODEL  ",
        font=("Segoe UI", int(10 * s), "bold"),
        bg=ACCENT_DARK, fg=FG, anchor="w",
    ).pack(fill=tk.X, padx=int(4 * s), pady=int(2 * s))

    # ── Encoder group ──────────────────────────────────────
    _build_param_group(panel, parent, prefix, "Encoder", _ENCODER_PARAMS, model_cfg, saved, s)

    # ── Core / Jamba group ─────────────────────────────────
    _build_param_group(panel, parent, prefix, "Temporal Core (Jamba)", _CORE_PARAMS, model_cfg, saved, s)


def _build_param_group(
    panel,
    parent: tk.Frame,
    prefix: str,
    group_name: str,
    params: List[_Param],
    model_cfg: object,
    saved: Dict[str, Any] | None,
    s: float,
) -> None:
    """Build a labelled group of parameter sliders."""
    grp = tk.Frame(parent, bg=BG_GROUP, padx=int(6 * s), pady=int(4 * s))
    grp.pack(fill=tk.X, padx=int(6 * s), pady=int(2 * s))

    tk.Label(
        grp, text=group_name.upper(),
        font=("Segoe UI", int(8 * s), "bold"),
        bg=BG_GROUP, fg=FG_DIM, anchor="w",
    ).pack(fill=tk.X)

    for p in params:
        full_key = f"{prefix}.{p.key}"

        # Use saved value if available, otherwise current config default.
        if saved and full_key in saved:
            current = p.type_(saved[full_key])
        else:
            current = _get_nested(model_cfg, p.key)

        _build_slider_row(panel, grp, full_key, p, current, s)


def _build_slider_row(
    panel,
    parent: tk.Frame,
    full_key: str,
    p: _Param,
    current: Any,
    s: float,
) -> None:
    """Build a single label + slider + value row."""
    row = tk.Frame(parent, bg=BG_GROUP)
    row.pack(fill=tk.X, pady=int(1 * s))

    # Label
    tk.Label(
        row, text=p.label, width=14, anchor="w",
        font=("Segoe UI", int(8 * s)),
        bg=BG_GROUP, fg=FG,
    ).pack(side=tk.LEFT)

    # Value readout
    val_label = tk.Label(
        row, text=str(p.type_(current)), width=6, anchor="e",
        font=("Consolas", int(8 * s)),
        bg=BG_GROUP, fg=ACCENT,
    )
    val_label.pack(side=tk.RIGHT, padx=(int(4 * s), 0))
    panel._model_labels[full_key] = val_label

    # Slider
    var = tk.DoubleVar(value=float(current))
    panel._model_vars[full_key] = var

    slider = tk.Scale(
        row, variable=var,
        from_=p.lo, to=p.hi, resolution=p.step,
        orient=tk.HORIZONTAL, length=int(180 * s),
        bg=BG_GROUP, fg=FG, troughcolor=ACCENT_DARK,
        highlightthickness=0, bd=0, sliderrelief="flat",
        font=("Segoe UI", int(7 * s)),
        showvalue=False,
        command=lambda val, k=full_key, pp=p: _on_model_param_change(panel, k, pp, val),
    )
    slider.pack(side=tk.RIGHT)


def _on_model_param_change(panel, full_key: str, p: _Param, val: str) -> None:
    """Called when a slider moves — update label and persist."""
    typed = p.type_(float(val))
    label = panel._model_labels.get(full_key)
    if label:
        label.config(text=str(typed))

    # Persist all model params.
    _persist_model_config(panel)


def _apply_preset(panel, preset_name: str) -> None:
    """Apply a named preset and update all sliders."""
    student_vals, teacher_vals = PRESETS[preset_name]

    for prefix, overrides in [("student", student_vals), ("teacher", teacher_vals)]:
        for key, value in overrides.items():
            full_key = f"{prefix}.{key}"
            var = panel._model_vars.get(full_key)
            if var is not None:
                var.set(float(value))
                label = panel._model_labels.get(full_key)
                if label:
                    label.config(text=str(value))

    _persist_model_config(panel)


def _persist_model_config(panel) -> None:
    """Gather all model slider values and save to settings store."""
    data: Dict[str, Any] = {}
    for full_key, var in panel._model_vars.items():
        # Determine the type from the param descriptors.
        param_key = full_key.split(".", 1)[1]  # strip student./teacher.
        p = _find_param(param_key)
        if p:
            data[full_key] = p.type_(var.get())
        else:
            data[full_key] = int(var.get())
    panel._store.set("model_config", data)


def _find_param(param_key: str) -> _Param | None:
    """Look up a _Param by its dotted key."""
    for p in _ENCODER_PARAMS + _CORE_PARAMS:
        if p.key == param_key:
            return p
    return None
