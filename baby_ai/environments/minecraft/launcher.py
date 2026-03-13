"""
Minecraft auto-launcher.

Launches Minecraft Java Edition directly using the bundled Java runtime
and version JARs — no official launcher required.  Supports automatic
world loading via ``--quickPlaySingleplayer``.

The launcher:
1. Parses the version JSON to build the classpath and arguments.
2. Filters libraries by platform rules (Windows only).
3. Resolves the Java 21 runtime shipped with the MC launcher.
4. Starts MC as a subprocess and waits for the window to appear.

Usage::

    launcher = MinecraftLauncher(mc_dir="~/.minecraft", version="1.21.11")
    hwnd = launcher.launch(world="Baby_AIs World")
    # ... use hwnd for screen capture and input ...
    launcher.stop()
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from baby_ai.environments.minecraft.window import find_windows_by_title
from baby_ai.utils.logging import get_logger
from baby_ai.environments.minecraft.manifest import (
    find_java as _find_java,
    rules_allow as _rules_allow,
    build_classpath as _build_classpath,
    build_jvm_args as _build_jvm_args,
    build_game_args as _build_game_args,
    find_natives_dir as _find_natives_dir,
)

log = get_logger("mc_launcher")

class MinecraftLauncher:
    """
    Launches Minecraft Java Edition directly (bypassing the official launcher).

    This reads the version JSON from the .minecraft directory, builds the
    full Java command line, and starts MC as a subprocess.  After launch
    it waits for the game window to appear.

    Args:
        mc_dir: Path to the .minecraft directory.
        version: Minecraft version ID (e.g. ``"1.21.11"``).
        player_name: In-game player name (from usercache or custom).
        player_uuid: Player UUID (from usercache or generated).
        max_memory_mb: Maximum JVM heap size in MB.
        window_width: Initial game window width.
        window_height: Initial game window height.
    """

    def __init__(
        self,
        mc_dir: str | Path = "",
        version: str = "1.21.11",
        player_name: str = "",
        player_uuid: str = "",
        max_memory_mb: int = 4096,
        window_width: int = 854,
        window_height: int = 480,
    ):
        self._mc_dir = Path(mc_dir) if mc_dir else Path(os.environ.get("MC_DIR", ""))
        self._version = version
        self._player_name = player_name or os.environ.get("MC_PLAYER_NAME", "Player")
        self._player_uuid = player_uuid or os.environ.get("MC_PLAYER_UUID", "00000000-0000-0000-0000-000000000000")
        self._max_memory_mb = max_memory_mb
        self._window_width = window_width
        self._window_height = window_height
        self._process: Optional[subprocess.Popen] = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: Optional[threading.Thread] = None

        # Validate paths
        version_json = self._mc_dir / "versions" / version / f"{version}.json"
        if not version_json.is_file():
            raise FileNotFoundError(f"Version manifest not found: {version_json}")

    # ── MC options management ───────────────────────────────────

    def ensure_background_options(self) -> None:
        """
        Patch Minecraft's ``options.txt`` (vanilla config file in .minecraft/)
        so the game keeps running when unfocused.

        Sets ``pauseOnLostFocus:false`` which is required for the AI to
        control the game via PostMessage while the window is in the
        background.
        Also sets ``rawMouseInput:false`` to allow simulating mouse movement
        while in the background using WM_MOUSEMOVE instead of hardware inputs.
        """
        opts_path = self._mc_dir / "options.txt"
        if not opts_path.is_file():
            log.warning("options.txt not found at %s — skipping.", opts_path)
            return

        lines = opts_path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        found_pause = False
        found_mouse = False
        found_respawn = False

        for line in lines:
            if line.startswith("pauseOnLostFocus:"):
                old_val = line.split(":", 1)[1].strip()
                if old_val != "false":
                    log.info("Patching options.txt: pauseOnLostFocus:%s → false", old_val)
                new_lines.append("pauseOnLostFocus:false")
                found_pause = True
            elif line.startswith("rawMouseInput:"):
                old_val = line.split(":", 1)[1].strip()
                if old_val != "false":
                    log.info("Patching options.txt: rawMouseInput:%s → false", old_val)
                new_lines.append("rawMouseInput:false")
                found_mouse = True
            elif line.startswith("doImmediateRespawn:"):
                old_val = line.split(":", 1)[1].strip()
                if old_val != "true":
                    log.info("Patching options.txt: doImmediateRespawn:%s → true", old_val)
                new_lines.append("doImmediateRespawn:true")
                found_respawn = True
            elif line.startswith("key_key.inventory:"):
                new_lines.append("key_key.inventory:key.keyboard.e")
            elif line.startswith("key_key.chat:"):
                new_lines.append("key_key.chat:key.keyboard.unknown")
            elif line.startswith("hideServerAddress:"): # we inject our own setting loosely here
                new_lines.append(line)
            else:
                new_lines.append(line)

        if not found_pause:
            log.info("Adding pauseOnLostFocus:false to options.txt")
            new_lines.append("pauseOnLostFocus:false")
            
        if not found_mouse:
            log.info("Adding rawMouseInput:false to options.txt")
            new_lines.append("rawMouseInput:false")
            
        if not found_respawn:
            log.info("Adding doImmediateRespawn:true to options.txt")
            new_lines.append("doImmediateRespawn:true")

        opts_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def launch(self, world: Optional[str] = None, timeout_sec: float = 120.0) -> int:
        """
        Launch Minecraft and wait for its window to appear.

        Args:
            world: World folder name for ``--quickPlaySingleplayer``.
                   If None, MC opens normally to the title screen.
            timeout_sec: Maximum seconds to wait for the window.

        Returns:
            The window handle (HWND) of the Minecraft window.

        Raises:
            TimeoutError: If the window doesn't appear in time.
            FileNotFoundError: If required files are missing.
        """
        if self._process is not None and self._process.poll() is None:
            log.warning("Minecraft is already running (PID %d).", self._process.pid)
            # Try to find existing window
            hwnd = self._wait_for_window(timeout_sec=10.0)
            if hwnd:
                return hwnd
            raise RuntimeError("MC process is running but window not found.")

        # ── Parse version manifest ──────────────────────────────
        version_json = self._mc_dir / "versions" / self._version / f"{self._version}.json"
        with open(version_json) as f:
            version_data = json.load(f)

        # ── Auto-detect Fabric Loader ───────────────────────────
        # If the user has installed Fabric Loader, a version folder
        # like  fabric-loader-0.16.9-1.21.1  will exist alongside
        # the vanilla one.  We prefer it so the Baby-AI Bridge mod
        # (and any other Fabric mods) get loaded automatically.
        fabric_version = self._find_fabric_version()
        if fabric_version:
            log.info("Fabric Loader detected: %s", fabric_version)
            fabric_json_path = (
                self._mc_dir / "versions" / fabric_version
                / f"{fabric_version}.json"
            )
            with open(fabric_json_path) as f:
                fabric_data = json.load(f)

            # Fabric's JSON inherits from the vanilla version —
            # merge so we get the full classpath + Fabric's main class.
            version_data = self._merge_fabric_json(version_data, fabric_data)

        main_class = version_data["mainClass"]
        java_info = version_data.get("javaVersion", {})
        java_component = java_info.get("component", "java-runtime-delta")

        log.info("Minecraft %s — mainClass=%s, java=%s",
                 self._version, main_class, java_component)

        # ── Resolve paths ───────────────────────────────────────
        java_exe = _find_java(self._mc_dir, java_component)
        natives_dir = _find_natives_dir(self._mc_dir, self._version)
        classpath = _build_classpath(self._mc_dir, version_data)

        # ── Build command line ──────────────────────────────────
        jvm_args = _build_jvm_args(
            version_data, natives_dir, classpath,
            max_memory_mb=self._max_memory_mb,
        )
        game_args = _build_game_args(
            version_data, self._mc_dir,
            self._player_name, self._player_uuid,
            world=world,
            width=self._window_width,
            height=self._window_height,
        )

        cmd = [str(java_exe)] + jvm_args + [main_class] + game_args

        # Log the command (truncated for readability)
        cmd_preview = " ".join(cmd[:5]) + " ... " + " ".join(cmd[-6:])
        log.info("Launch command: %s", cmd_preview)
        log.info("Full command has %d arguments", len(cmd))

        # ── Start process ───────────────────────────────────────
        # stdout → DEVNULL (MC logs to file, not console)
        # stderr → PIPE, drained by a background thread to prevent
        #          the 64 KB pipe-buffer deadlock that can freeze MC.
        self._process = subprocess.Popen(
            cmd,
            cwd=str(self._mc_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        log.info("Minecraft started: PID %d", self._process.pid)

        # Drain stderr in background so the pipe never fills up
        self._stderr_lines = []
        def _drain_stderr():
            assert self._process is not None and self._process.stderr is not None
            for raw in self._process.stderr:
                self._stderr_lines.append(raw.decode("utf-8", errors="replace"))
        self._stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        self._stderr_thread.start()

        # ── Wait for window ─────────────────────────────────────
        hwnd = self._wait_for_window(timeout_sec=timeout_sec)
        if hwnd is None:
            # Check if process died
            rc = self._process.poll()
            if rc is not None:
                stderr_text = "".join(self._stderr_lines)
                raise RuntimeError(
                    f"Minecraft exited with code {rc} before window appeared.\n"
                    f"stderr (last 2000 chars):\n{stderr_text[-2000:]}"
                )
            raise TimeoutError(
                f"Minecraft window did not appear within {timeout_sec}s. "
                "The game may still be loading."
            )

        log.info("Minecraft window found: hwnd=%s", hex(hwnd))
        return hwnd

    def _wait_for_window(self, timeout_sec: float = 120.0) -> Optional[int]:
        """Poll for the Minecraft window to appear."""
        deadline = time.monotonic() + timeout_sec
        poll_interval = 1.0  # check every second

        while time.monotonic() < deadline:
            # Check if process is still alive
            if self._process and self._process.poll() is not None:
                return None

            matches = find_windows_by_title("Minecraft")
            for hwnd, title in matches:
                # Filter out launcher windows — we want the game window
                title_lower = title.lower()
                if "launcher" in title_lower:
                    continue
                # The game window title is like "Minecraft 1.21.11"
                if self._version in title or "singleplayer" in title_lower:
                    return hwnd
                # Also match generic "Minecraft*" windows (loading screen)
                if title_lower.startswith("minecraft"):
                    return hwnd

            time.sleep(poll_interval)

        return None

    def wait_for_world_ready(self, timeout_sec: float = 120.0) -> Optional[int]:
        """
        Wait until the Minecraft world is fully loaded.

        Detection uses two parallel signals (whichever fires first):

        1. **Window title** — MC changes to "Minecraft X.Y.Z - Singleplayer"
           once the world is loaded.  We strip the "(Not Responding)" suffix
           that Windows appends during heavy loading.
        2. **Log file** — The integrated server prints
           ``Done (X.XXXs)! For help, type "help"`` in ``latest.log``
           when chunk generation finishes.  This is the most reliable
           signal and works even when the window is unresponsive.

        Returns:
            The (possibly new) window handle if the world loaded,
            or None on timeout.  MC sometimes creates a new window
            when transitioning from the loading screen to the game,
            so callers should use the returned hwnd.
        """
        deadline = time.monotonic() + timeout_sec
        poll_interval = 2.0
        log_file = self._mc_dir / "logs" / "latest.log"
        last_log_check = ""

        log.info("Waiting for world to finish loading (up to %ds)...", int(timeout_sec))

        while time.monotonic() < deadline:
            # ── Check process health ────────────────────────────
            if self._process and self._process.poll() is not None:
                log.warning("Minecraft process died while waiting for world.")
                return None

            # ── Signal 1: window title ──────────────────────────
            matches = find_windows_by_title("Minecraft")
            for found_hwnd, title in matches:
                # Windows appends " (Not Responding)" during heavy load
                clean = title.replace(" (Not Responding)", "").strip()
                if "singleplayer" in clean.lower():
                    log.info("World loaded! Title: '%s' hwnd=%s", title, hex(found_hwnd))
                    # Extra buffer for chunk rendering to finish
                    time.sleep(5.0)
                    return found_hwnd

            # ── Signal 2: MC log file ───────────────────────────
            if log_file.is_file():
                try:
                    content = log_file.read_text(encoding="utf-8", errors="replace")
                    if 'Done (' in content and 'For help' in content:
                        log.info("World loaded! (detected via MC log file)")
                        time.sleep(5.0)
                        # Find the current valid MC window
                        fresh_matches = find_windows_by_title("Minecraft")
                        for fh, ft in fresh_matches:
                            if "launcher" not in ft.lower():
                                return fh
                        # Fallback — return None and let caller re-find
                        return None
                except OSError:
                    pass

            # ── Progress logging ────────────────────────────────
            elapsed = timeout_sec - (deadline - time.monotonic())
            if int(elapsed) % 10 < poll_interval and elapsed > 1:
                log.info("  Still loading... (%.0fs elapsed)", elapsed)

            time.sleep(poll_interval)

        log.warning("World did not finish loading within %ds.", int(timeout_sec))
        return None

    def stop(self, timeout_sec: float = 10.0) -> None:
        """
        Gracefully stop Minecraft.

        Attempts to send WM_CLOSE to the window first. If it doesn't
        exit within the timeout, terminates the process. Crucially,
        it also ensures the OS mouse cursor is unbounded (ClipCursor),
        since hard-killing a game often leaves the mouse trapped.
        """
        if self._process is None or self._process.poll() is not None:
            log.info("Minecraft is not running.")
            return

        pid = self._process.pid
        log.info("Stopping Minecraft (PID %d)...", pid)

        # 1. Try to close cleanly via WM_CLOSE to top-level windows
        import ctypes
        import ctypes.wintypes as wt
        user32 = ctypes.windll.user32

        def _enum_cb(hwnd, lParam):
            win_pid = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(win_pid))
            if win_pid.value == pid:
                # 0x0010 = WM_CLOSE
                user32.PostMessageW(hwnd, 0x0010, 0, 0)
            return True

        CB_TYPE = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        user32.EnumWindows(CB_TYPE(_enum_cb), 0)

        # 2. Wait for graceful exit
        try:
            self._process.wait(timeout=timeout_sec)
            log.info("Minecraft exited cleanly.")
        except subprocess.TimeoutExpired:
            log.warning("Minecraft did not exit in %ds — killing.", timeout_sec)
            self._process.kill()
            self._process.wait(timeout=5.0)

        # 3. SAFETY: Always unclip the cursor!
        # If MC was killed while capturing the mouse, Windows leaves it trapped
        # in an invisible box where the window used to be.
        try:
            user32.ClipCursor(None)
            log.info("OS cursor un-clipped.")
        except Exception as e:
            log.warning("Failed to unclip cursor: %s", e)

    @property
    def is_running(self) -> bool:
        """Check if the MC process is still alive."""
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> Optional[int]:
        """Return the MC process PID, or None."""
        return self._process.pid if self._process else None

    def get_usercache(self) -> Tuple[str, str]:
        """
        Read player name and UUID from the usercache.json.

        Returns:
            (player_name, player_uuid) tuple.
        """
        cache_path = self._mc_dir / "usercache.json"
        if not cache_path.is_file():
            return self._player_name, self._player_uuid

        with open(cache_path) as f:
            entries = json.load(f)

        if entries:
            entry = entries[0]  # most recent player
            return entry.get("name", self._player_name), entry.get("uuid", self._player_uuid)

        return self._player_name, self._player_uuid

    # ── Fabric Loader support ───────────────────────────────────

    def _find_fabric_version(self) -> Optional[str]:
        """
        Find the latest installed Fabric Loader version matching our MC version.

        Fabric Loader installs itself as a version folder named like
        ``fabric-loader-0.16.9-1.21.1`` in ``.minecraft/versions/``.

        Returns:
            The version folder name, or ``None`` if Fabric is not installed.
        """
        versions_dir = self._mc_dir / "versions"
        if not versions_dir.is_dir():
            return None

        prefix = "fabric-loader-"
        suffix = f"-{self._version}"
        best: Optional[str] = None

        for entry in versions_dir.iterdir():
            if (entry.is_dir()
                    and entry.name.startswith(prefix)
                    and entry.name.endswith(suffix)):
                json_file = entry / f"{entry.name}.json"
                if json_file.is_file():
                    # Pick the lexicographically highest (= newest loader)
                    if best is None or entry.name > best:
                        best = entry.name

        return best

    @staticmethod
    def _merge_fabric_json(
        parent: Dict[str, Any],
        child: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge a Fabric Loader version JSON with its parent (vanilla).

        Fabric overrides ``mainClass`` and *prepends* its own libraries
        and JVM arguments to the vanilla ones.
        """
        merged = dict(parent)

        # Main class — Fabric's Knot launcher replaces the vanilla one
        if "mainClass" in child:
            merged["mainClass"] = child["mainClass"]

        # Libraries — Fabric's must come before vanilla's on the classpath
        if "libraries" in child:
            merged["libraries"] = child["libraries"] + parent.get("libraries", [])

        # JVM arguments — prepend Fabric-specific ones
        parent_jvm = parent.get("arguments", {}).get("jvm", [])
        child_jvm  = child.get("arguments", {}).get("jvm", [])
        if child_jvm:
            if "arguments" not in merged:
                merged["arguments"] = {}
            merged["arguments"]["jvm"] = child_jvm + parent_jvm

        return merged
