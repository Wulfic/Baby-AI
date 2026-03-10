"""
Lightweight screen-region analyzers for Minecraft HUD signals.

Extracts game-state signals from raw screenshot pixels — no mods,
no external APIs.  Works with vanilla Minecraft Java Edition by
reading well-known HUD regions (health bar, hotbar, death overlay).

All analysers operate on a **full-resolution BGR uint8 frame**
(the raw capture *before* resize to model input).  Region
coordinates are computed as fractions of frame size so they work
at any resolution.

Detected signals
----------------
- **Death** : the full-screen red "You Died!" overlay.
- **Block break** : sustained crosshair-crack animation while
  attacking (localised pixel churn around screen centre).
- **Item pickup** : change in the hotbar region at the bottom of
  the screen (new stack appears or count increases).
- **Inventory / crafting UI** : large gray structured overlay
  covering the centre of the screen (inventory, crafting table,
  furnace, etc.).
- **Block placement** : right-click induced visual change in the
  forward view (distinct from mining crack animation).
- **Crafting event** : hotbar change that occurs while a crafting
  UI is open or immediately after it closes.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


# ────────────────────────────────────────────────────────────────
# Death detection
# ────────────────────────────────────────────────────────────────

def detect_death(frame: np.ndarray) -> bool:
    """
    Detect the Minecraft "You Died!" screen.

    The death screen in Java Edition 1.21+ is a semi-transparent
    dark-red overlay covering the viewport.  We use two complementary
    checks to be robust across different brightness / gamma settings:

    1. **Global red-shift**: a significant fraction of pixels are
       tinted red (R channel dominates G and B).
    2. **Title region red text**: the top-center portion of the
       screen contains bright-red "You Died!" text on a dark
       background.

    Args:
        frame: (H, W, 3) BGR uint8 array.

    Returns:
        True if the death screen is visible.
    """
    h, w = frame.shape[:2]

    # Work in float to avoid overflow
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)

    # ── Check 1: Global red-tinted overlay ──────────────────────
    # Relaxed vs. the old check — modern MC death screens are darker.
    # Pixel qualifies if red channel clearly dominates green + blue.
    red_mask = (
        (r > 40)            # not too dark (floor lowered from 80)
        & (r > g * 1.6)     # red strongly exceeds green
        & (r > b * 1.6)     # red strongly exceeds blue
        & (g < 100)         # green isn't too bright
        & (b < 100)         # blue isn't too bright
    )
    red_frac = float(red_mask.mean())

    # Require >= 20% of all pixels (lowered from 35%).
    if red_frac >= 0.20:
        return True

    # ── Check 2: "You Died!" title text region ─────────────────
    # The title sits in roughly the top 25-40% vertically,
    # centered horizontally. Look for bright-red pixels there.
    y_top = int(h * 0.20)
    y_bot = int(h * 0.42)
    x_left = int(w * 0.25)
    x_right = int(w * 0.75)
    region_r = r[y_top:y_bot, x_left:x_right]
    region_g = g[y_top:y_bot, x_left:x_right]
    region_b = b[y_top:y_bot, x_left:x_right]

    bright_red = (
        (region_r > 150)
        & (region_g < 80)
        & (region_b < 80)
    )
    bright_frac = float(bright_red.mean())
    # The "You Died!" text is large — even 3% of this region is a
    # strong signal (actual text typically covers 5–10%).
    if bright_frac >= 0.03:
        return True

    return False


# ────────────────────────────────────────────────────────────────
# Hotbar change (item pickup) detection
# ────────────────────────────────────────────────────────────────

def _hotbar_region(frame: np.ndarray) -> np.ndarray:
    """
    Extract the hotbar region from the bottom-center of the frame.

    Minecraft's hotbar is centered horizontally and sits at the very
    bottom of the viewport.  We take the bottom ~7 % of the height
    and the middle ~45 % of the width, which reliably captures all
    9 hotbar slots at any vanilla GUI scale.

    Returns:
        (h, w, 3) BGR uint8 sub-array (a copy).
    """
    h, w = frame.shape[:2]
    y_start = int(h * 0.93)
    x_start = int(w * 0.275)
    x_end = int(w * 0.725)
    return frame[y_start:, x_start:x_end].copy()


class HotbarTracker:
    """
    Detects hotbar changes between consecutive frames.

    A change in the hotbar region (beyond a noise threshold)
    indicates the player picked up / dropped / crafted an item,
    or used a consumable.

    Args:
        change_threshold: Minimum mean pixel diff (0-255 scale) to
            count as a genuine change.  Minecraft's hotbar is mostly
            static so even a small threshold works well.
    """

    def __init__(self, change_threshold: float = 3.0):
        self._prev_hotbar: Optional[np.ndarray] = None
        self._threshold = change_threshold

    def reset(self) -> None:
        """Call on episode reset."""
        self._prev_hotbar = None

    def update(self, frame: np.ndarray) -> float:
        """
        Compare the current hotbar to the previous snapshot.

        Args:
            frame: Full-resolution (H, W, 3) BGR uint8 frame.

        Returns:
            Normalised change magnitude in ``[0, 1]``.
            0.0 means no change; 1.0 means a very large change.
        """
        hotbar = _hotbar_region(frame)

        if self._prev_hotbar is None or self._prev_hotbar.shape != hotbar.shape:
            self._prev_hotbar = hotbar
            return 0.0

        diff = np.abs(
            hotbar.astype(np.float32) - self._prev_hotbar.astype(np.float32)
        ).mean()

        self._prev_hotbar = hotbar

        if diff < self._threshold:
            return 0.0

        # Normalise: 3-30 pixel diff -> 0.0-1.0
        return min(float((diff - self._threshold) / 27.0), 1.0)


# ────────────────────────────────────────────────────────────────
# Block-break detection (crosshair crack animation)
# ────────────────────────────────────────────────────────────────

def _center_region(frame: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """
    Extract a small square region around the screen centre
    where the crosshair and block-crack animation appear.

    Args:
        frame: (H, W, 3) BGR uint8.
        frac:  Fraction of the shorter dimension to use as the
               half-side of the square.

    Returns:
        (h, w, 3) BGR uint8 sub-array (copy).
    """
    h, w = frame.shape[:2]
    half = int(min(h, w) * frac)
    cy, cx = h // 2, w // 2
    return frame[cy - half : cy + half, cx - half : cx + half].copy()


class BlockBreakTracker:
    """
    Detects the sustained visual churn around the crosshair that
    occurs when the player is mining / breaking a block.

    Minecraft shows animated crack lines on the targeted block face
    while the attack button is held.  These cause localised pixel
    change in the centre of the screen over consecutive frames.

    We track a rolling window of centre-region diffs and flag a
    "block break" when the accumulated churn exceeds a threshold
    *and* the player is holding the attack button (caller provides
    that signal).

    Args:
        window_size: Number of frames to accumulate churn over.
        break_threshold: Minimum cumulative churn (0-1 scale per
            frame, summed) to declare a block break event.
    """

    def __init__(self, window_size: int = 10, break_threshold: float = 0.25):
        self._window_size = window_size
        self._break_threshold = break_threshold
        self._prev_center: Optional[np.ndarray] = None
        self._churn_history: list[float] = []

    def reset(self) -> None:
        """Call on episode reset."""
        self._prev_center = None
        self._churn_history.clear()

    def update(self, frame: np.ndarray, is_attacking: bool) -> float:
        """
        Update with the latest frame and return a break score.

        Args:
            frame: Full-resolution (H, W, 3) BGR uint8 frame.
            is_attacking: Whether the agent is currently holding
                the attack (left-click) button.

        Returns:
            Break score in ``[0, 1]``.  Values above ~0.5 strongly
            suggest a block was (or is being) broken.
        """
        center = _center_region(frame)

        if self._prev_center is None or self._prev_center.shape != center.shape:
            self._prev_center = center
            return 0.0

        diff = np.abs(
            center.astype(np.float32) - self._prev_center.astype(np.float32)
        ).mean() / 255.0

        self._prev_center = center

        if is_attacking:
            self._churn_history.append(diff)
        else:
            # Decay history when not attacking
            self._churn_history.clear()

        # Keep only the recent window
        if len(self._churn_history) > self._window_size:
            self._churn_history = self._churn_history[-self._window_size :]

        if not self._churn_history:
            return 0.0

        cumulative = sum(self._churn_history)
        score = min(cumulative / self._break_threshold, 1.0)
        return float(score)


# ────────────────────────────────────────────────────────────────
# Inventory / Crafting UI detection
# ────────────────────────────────────────────────────────────────

def _ui_center_region(frame: np.ndarray) -> np.ndarray:
    """
    Extract the central region where inventory/crafting UIs appear.

    Minecraft overlays inventory screens on the centre of the viewport
    with a distinctive gray-brown background and slot grid.  We sample
    the middle ~50 % × ~50 % of the frame.

    Returns:
        (h, w, 3) BGR uint8 sub-array (copy).
    """
    h, w = frame.shape[:2]
    y0 = int(h * 0.25)
    y1 = int(h * 0.75)
    x0 = int(w * 0.25)
    x1 = int(w * 0.75)
    return frame[y0:y1, x0:x1].copy()


def detect_inventory_open(frame: np.ndarray) -> bool:
    """
    Detect whether a Minecraft inventory / crafting / furnace UI is open.

    When any container UI is open in vanilla Minecraft the centre of the
    screen is dominated by:
    - A gray/brown background (RGB roughly 180-200, 180-200, 180-200 for
      the slot background) with low saturation.
    - Very uniform pixel regions (the flat-colored inventory slots).
    - The 3D world behind is darkened/dimmed.

    We detect this by checking:
    1. A large fraction of the centre region has low colour saturation
       (gray-ish) AND mid-range brightness.
    2. Low local variance — the grid slots are flat rectangles.

    Args:
        frame: (H, W, 3) BGR uint8 array.

    Returns:
        True if an inventory-style UI is likely open.
    """
    region = _ui_center_region(frame)  # (h, w, 3)

    # Downsample large regions to cap memory usage.  Inventory
    # detection only needs coarse colour/variance stats — 240×240
    # is more than enough and prevents OOM on 1080p+ raw frames.
    _MAX_DIM = 240
    rh, rw = region.shape[:2]
    if rh > _MAX_DIM or rw > _MAX_DIM:
        scale = _MAX_DIM / max(rh, rw)
        new_h, new_w = max(1, int(rh * scale)), max(1, int(rw * scale))
        import cv2
        region = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_AREA)

    b, g, r = (
        region[:, :, 0].astype(np.float32),
        region[:, :, 1].astype(np.float32),
        region[:, :, 2].astype(np.float32),
    )

    # ── Criterion 1: low saturation + mid brightness ────────────
    # Saturation approximation: max(R,G,B) - min(R,G,B)
    pix_max = np.maximum(np.maximum(r, g), b)
    pix_min = np.minimum(np.minimum(r, g), b)
    saturation = pix_max - pix_min
    brightness = (r + g + b) / 3.0

    # Inventory-slot pixels: low saturation (<50), brightness 100-220
    low_sat_mask = (saturation < 50) & (brightness > 100) & (brightness < 220)
    low_sat_frac = low_sat_mask.mean()

    # ── Criterion 2: low local variance (flat rectangles) ───────
    # Downsample to 8×8 blocks and compute variance per block
    bh, bw = 8, 8
    rh, rw = region.shape[0] // bh, region.shape[1] // bw
    if rh < 2 or rw < 2:
        return False  # frame too small

    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    # Crop to exact multiple of block size
    gray = gray[: rh * bh, : rw * bw]
    blocks = gray.reshape(rh, bh, rw, bw).transpose(0, 2, 1, 3).reshape(rh * rw, bh * bw)
    block_vars = blocks.var(axis=1)
    low_var_frac = (block_vars < 200).mean()

    # Inventory UI: lots of low-saturation pixels AND lots of
    # uniform blocks.  Thresholds tuned empirically.
    return float(low_sat_frac) > 0.35 and float(low_var_frac) > 0.40


class CraftingTracker:
    """
    Detects crafting events by recognising inventory-UI-open states
    combined with **net** hotbar changes that indicate a *new* item
    was created (as opposed to mere reorganisation of existing items).

    **Anti-exploit measures:**

    * **Net-mass check** -- we snapshot the hotbar when the UI first
      opens and compare it to the hotbar when the UI closes.  Item
      *reorganisation* (moving an item from the inventory grid to the
      hotbar or vice-versa) tends to keep total hotbar pixel "mass"
      (mean brightness) roughly constant, whereas crafting a new item
      and placing it on the hotbar increases the mass.  We require a
      positive net-mass delta above a threshold.
    * **Minimum UI dwell time** -- the UI must stay open for at least
      ``min_frames_in_ui`` frames before a craft can register.
      Quick open/close cycles are ignored.
    * **Cooldown** -- after a craft fires, no new craft can fire for
      ``cooldown_frames`` frames.

    Args:
        hotbar_threshold: Minimum mean pixel diff to count as a
            hotbar change (same units as ``HotbarTracker``).
        mass_gain_threshold: Minimum *increase* in mean hotbar pixel
            brightness (0-255) to consider a net item gain.
        min_frames_in_ui: Minimum consecutive frames UI must be open
            before a craft event can be emitted.
        cooldown_frames: Minimum frames between successive craft
            score emissions.
    """

    def __init__(
        self,
        hotbar_threshold: float = 5.0,
        mass_gain_threshold: float = 2.0,
        min_frames_in_ui: int = 10,
        cooldown_frames: int = 30,
    ):
        self._hotbar_threshold = hotbar_threshold
        self._mass_gain_threshold = mass_gain_threshold
        self._min_frames_in_ui = min_frames_in_ui
        self._cooldown_frames = cooldown_frames

        self._prev_hotbar: Optional[np.ndarray] = None
        self._ui_was_open: bool = False
        self._hotbar_change_while_open: float = 0.0
        self._frames_ui_open: int = 0

        # Snapshot of hotbar at the moment the UI first opens
        self._hotbar_snapshot_on_open: Optional[np.ndarray] = None

        # Cooldown counter — decremented each frame, craft suppressed
        # while > 0.
        self._cooldown_remaining: int = 0

    def reset(self) -> None:
        self._prev_hotbar = None
        self._ui_was_open = False
        self._hotbar_change_while_open = 0.0
        self._frames_ui_open = 0
        self._hotbar_snapshot_on_open = None
        self._cooldown_remaining = 0

    @staticmethod
    def _hotbar_mass(hotbar: np.ndarray) -> float:
        """Mean brightness of the hotbar region (0-255)."""
        return float(hotbar.astype(np.float32).mean())

    def update(self, frame: np.ndarray) -> dict:
        """
        Update with the latest frame.

        Returns:
            dict with:
                ui_open (bool): Whether a crafting / inventory UI is open.
                craft_score (float): 0.0-1.0 crafting event confidence.
                    Non-zero only on the frame a craft is detected.
                frames_in_ui (int): How many consecutive frames the UI
                    has been open (useful for rewarding engagement).
        """
        ui_open = detect_inventory_open(frame)
        craft_score = 0.0

        # Tick cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Track hotbar
        hotbar = _hotbar_region(frame)
        hotbar_diff = 0.0
        if self._prev_hotbar is not None and self._prev_hotbar.shape == hotbar.shape:
            hotbar_diff = float(
                np.abs(
                    hotbar.astype(np.float32) - self._prev_hotbar.astype(np.float32)
                ).mean()
            )
        self._prev_hotbar = hotbar

        if ui_open:
            if not self._ui_was_open:
                # UI just opened -- snapshot current hotbar for later
                # net-mass comparison.
                self._hotbar_snapshot_on_open = hotbar.copy()
                self._hotbar_change_while_open = 0.0
                self._frames_ui_open = 0

            self._frames_ui_open += 1

            # Accumulate frame-to-frame churn while UI is open
            if hotbar_diff > self._hotbar_threshold:
                self._hotbar_change_while_open += hotbar_diff
        else:
            # UI just closed (or was already closed)
            if self._ui_was_open:
                # -- Gate 1: minimum dwell time --
                long_enough = self._frames_ui_open >= self._min_frames_in_ui

                # -- Gate 2: significant frame-to-frame churn --
                churn_ok = self._hotbar_change_while_open > 0

                # -- Gate 3: net hotbar mass gain --
                mass_gained = False
                if self._hotbar_snapshot_on_open is not None:
                    mass_before = self._hotbar_mass(self._hotbar_snapshot_on_open)
                    mass_after = self._hotbar_mass(hotbar)
                    mass_gained = (mass_after - mass_before) > self._mass_gain_threshold

                # -- Gate 4: cooldown --
                off_cooldown = self._cooldown_remaining <= 0

                if long_enough and churn_ok and mass_gained and off_cooldown:
                    craft_score = min(self._hotbar_change_while_open / 50.0, 1.0)
                    self._cooldown_remaining = self._cooldown_frames

            self._hotbar_change_while_open = 0.0
            self._frames_ui_open = 0
            self._hotbar_snapshot_on_open = None

        self._ui_was_open = ui_open

        return {
            "ui_open": ui_open,
            "craft_score": craft_score,
            "frames_in_ui": self._frames_ui_open,
        }


class PlacementTracker:
    """
    Detects block placement by tracking right-click-induced visual
    changes in the forward view.

    Block placement produces a sudden, localised visual change in
    the area around / below the crosshair when the "use" (right-click)
    button is pressed — but *without* the sustained crack-animation
    churn that characterises mining.

    We distinguish placement from mining by:
    - Only triggering on right-click (use) actions, not left-click.
    - Looking for a single-frame visual spike in the wider forward
      region (not just the tiny crosshair patch).
    - Requiring that the change is NOT preceded by sustained churn
      (i.e. it's not just a mining animation).

    Args:
        change_threshold: Minimum single-frame visual diff (0-1)
            to consider as a potential placement.
    """

    def __init__(self, change_threshold: float = 0.015):
        self._threshold = change_threshold
        self._prev_forward: Optional[np.ndarray] = None
        self._recent_churn: deque[float] = deque(maxlen=5)

    def reset(self) -> None:
        self._prev_forward = None
        self._recent_churn.clear()

    def update(self, frame: np.ndarray, is_using: bool) -> float:
        """
        Update with the latest frame.

        Args:
            frame: Full-resolution (H, W, 3) BGR uint8 frame.
            is_using: Whether the agent is currently pressing "use"
                (right-click).

        Returns:
            Placement score in ``[0, 1]``.  >0.5 strongly suggests
            a block was placed.
        """
        # Use a wider forward region than BlockBreakTracker
        forward = _forward_region(frame)

        if self._prev_forward is None or self._prev_forward.shape != forward.shape:
            self._prev_forward = forward
            return 0.0

        diff = float(
            np.abs(
                forward.astype(np.float32) - self._prev_forward.astype(np.float32)
            ).mean() / 255.0
        )
        self._prev_forward = forward
        self._recent_churn.append(diff)

        if not is_using:
            return 0.0

        if diff < self._threshold:
            return 0.0

        # Reject if there's been sustained churn (= mining animation)
        # Mining shows gradual, sustained pixel change; placement shows
        # a single-frame spike.
        if len(self._recent_churn) >= 3:
            older = list(self._recent_churn)[:-1]
            avg_older = sum(older) / len(older)
            # If recent history also has lots of churn, it's mining
            if avg_older > self._threshold * 0.8:
                return 0.0

        # Score: how much of a spike this frame was
        score = min((diff - self._threshold) / 0.10, 1.0)
        return max(score, 0.0)


def _forward_region(frame: np.ndarray, frac: float = 0.25) -> np.ndarray:
    """
    Extract the forward-view region (wider than crosshair centre).

    This captures the area in front of the player where placed blocks
    would appear — roughly the central 25 % of the screen in each
    dimension.

    Args:
        frame: (H, W, 3) BGR uint8.
        frac:  Fraction of the shorter dimension for the half-side.

    Returns:
        (h, w, 3) BGR uint8 sub-array (copy).
    """
    h, w = frame.shape[:2]
    half = int(min(h, w) * frac)
    cy, cx = h // 2, w // 2
    return frame[cy - half : cy + half, cx - half : cx + half].copy()


# ────────────────────────────────────────────────────────────────
# Creative-workflow trackers (extracted to creative_tracking.py)
# Re-exported here for backward compatibility.
# ────────────────────────────────────────────────────────────────
from baby_ai.environments.minecraft.creative_tracking import (  # noqa: E402
    BuildingStreakTracker,
    CreativeSequenceTracker,
)

__all__ = [
    "detect_death",
    "HotbarTracker",
    "BlockBreakTracker",
    "detect_inventory_open",
    "CraftingTracker",
    "PlacementTracker",
    "BuildingStreakTracker",
    "CreativeSequenceTracker",
]
