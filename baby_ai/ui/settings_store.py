"""
Persistent settings store for Baby-AI UI.

Saves and loads all runtime-configurable settings to a JSON file
so they persist across runs.  The file is human-readable and
hand-editable if needed.

Stored settings
---------------
- Reward channel toggles (on/off per channel)
- AI input controls (on/off per key/button/look)
- Reward weight multipliers (float per channel)

Thread safety
-------------
All public methods acquire a file lock via ``threading.Lock`` so
concurrent UI-thread saves and main-thread loads are safe.  The
JSON file is rewritten atomically (write-to-temp then rename) to
prevent half-written files on crash.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from baby_ai.utils.logging import get_logger

log = get_logger("settings_store")

# Default location next to the main.py entry point.
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "baby_ai_settings.json"


class SettingsStore:
    """
    JSON-backed settings persistence for Baby-AI.

    Usage::

        store = SettingsStore()           # loads existing or creates new
        store.set("reward_toggles", {...})
        store.save()                       # explicit save
        data = store.get("reward_toggles") # returns dict or None

    The store auto-loads on construction.  Call :meth:`save` after
    mutating sections, or use :meth:`set` which saves automatically.

    Parameters
    ----------
    path : Path or str, optional
        Location of the JSON file.  Defaults to
        ``<project_root>/baby_ai_settings.json``.
    """

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self.load()

    # ── Public API ──────────────────────────────────────────────

    def load(self) -> None:
        """Load settings from disk.  Missing file → empty dict."""
        with self._lock:
            if self._path.exists():
                try:
                    with open(self._path, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                    log.info("Settings loaded from %s (%d sections)",
                             self._path, len(self._data))
                except (json.JSONDecodeError, OSError) as exc:
                    log.warning("Failed to load settings (%s) — using defaults.", exc)
                    self._data = {}
            else:
                self._data = {}

    def save(self) -> None:
        """Write current settings to disk atomically."""
        with self._lock:
            self._write_unlocked()

    def get(self, section: str) -> Optional[Dict[str, Any]]:
        """Return a section dict, or ``None`` if it doesn't exist."""
        with self._lock:
            return self._data.get(section)

    def set(self, section: str, data: Dict[str, Any], *, auto_save: bool = True) -> None:
        """Set a whole section and optionally save to disk."""
        with self._lock:
            self._data[section] = data
            if auto_save:
                self._write_unlocked()

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get a single value from a section."""
        with self._lock:
            sec = self._data.get(section, {})
            return sec.get(key, default)

    def set_value(self, section: str, key: str, value: Any, *, auto_save: bool = True) -> None:
        """Set a single value in a section and optionally save."""
        with self._lock:
            if section not in self._data:
                self._data[section] = {}
            self._data[section][key] = value
            if auto_save:
                self._write_unlocked()

    @property
    def path(self) -> Path:
        return self._path

    # ── Internal ────────────────────────────────────────────────

    def _write_unlocked(self) -> None:
        """Write JSON to a temp file then rename (atomic on most OS)."""
        tmp_path = self._path.with_suffix(".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            # Atomic rename (Windows: os.replace is atomic within same volume)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            log.warning("Failed to save settings: %s", exc)
            # Clean up temp file if rename failed
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
