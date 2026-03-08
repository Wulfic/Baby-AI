"""
TCP client that connects to the Baby-AI Bridge Fabric mod.

The mod runs a TCP server on ``localhost:5556`` and streams
newline-delimited JSON events for block breaks, placements,
item pickups, crafting, and player death.

Usage::

    bridge = ModBridge(port=5556)
    bridge.start()          # spawns a background reader thread

    # ... each environment step ...
    events = bridge.drain_events()
    for ev in events:
        print(ev["event"], ev)

    bridge.stop()

Event dict keys
---------------
All events have ``"event"`` (str) and ``"tick"`` (int).

- ``block_broken``  — block, x, y, z
- ``block_placed``  — block, x, y, z
- ``item_crafted``  — item, count
- ``item_picked_up`` — item, count
- ``player_death``  — source (death message string)
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from baby_ai.utils.logging import get_logger

log = get_logger("mod_bridge")


class ModBridge:
    """
    Connects to the Baby-AI Fabric mod's TCP event stream.

    Events arrive as JSON lines and are buffered in a thread-safe
    :class:`~collections.deque`.  Call :meth:`drain_events` each
    step to retrieve and clear pending events.

    Args:
        host: Hostname (always localhost for same-machine mod).
        port: TCP port the mod listens on.
        reconnect_interval: Seconds between reconnection attempts
            when the mod is not reachable.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        reconnect_interval: float = 5.0,
    ):
        self._host = host
        self._port = port
        self._reconnect_interval = reconnect_interval

        self._socket: Optional[socket.socket] = None
        self._events: deque[Dict[str, Any]] = deque(maxlen=10_000)
        self._connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Properties ──────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """True when the TCP connection to the mod is alive."""
        return self._connected

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        """Start the background reader thread (auto-connects)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mod-bridge",
        )
        self._thread.start()
        log.info("ModBridge started — will connect to %s:%d",
                 self._host, self._port)

    def stop(self) -> None:
        """Stop the reader thread and close the socket."""
        self._running = False
        self._close_socket()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        log.info("ModBridge stopped.")

    # ── Public API ──────────────────────────────────────────────

    def drain_events(self) -> List[Dict[str, Any]]:
        """Return all buffered events and clear the buffer.

        Thread-safe.  Returns an empty list if no events have
        arrived since the last call.
        """
        events: List[Dict[str, Any]] = []
        while self._events:
            try:
                events.append(self._events.popleft())
            except IndexError:
                break
        return events

    # ── Internal ────────────────────────────────────────────────

    def _run(self) -> None:
        """Background loop: connect → read → reconnect."""
        while self._running:
            if not self._connected:
                self._try_connect()
                if not self._connected:
                    time.sleep(self._reconnect_interval)
                    continue
            try:
                self._read_events()
            except Exception as exc:
                if self._running:
                    log.warning("ModBridge read error: %s — reconnecting", exc)
                self._connected = False
                self._close_socket()

    def _try_connect(self) -> None:
        """Attempt a single TCP connection to the mod."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((self._host, self._port))
            sock.settimeout(1.0)  # read timeout for polling
            self._socket = sock
            self._connected = True
            log.info("ModBridge connected to %s:%d", self._host, self._port)
        except (ConnectionRefusedError, OSError, socket.timeout) as exc:
            log.debug("ModBridge connect failed: %s", exc)
            self._connected = False

    def _read_events(self) -> None:
        """Read JSON lines from the socket until disconnected."""
        assert self._socket is not None
        buf = ""
        while self._running and self._connected:
            try:
                data = self._socket.recv(4096)
                if not data:
                    raise ConnectionError("Mod closed the connection")
                buf += data.decode("utf-8", errors="replace")

                # Process all complete lines in the buffer
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        self._events.append(event)
                    except json.JSONDecodeError:
                        log.debug("Bad JSON from mod: %.100s", line)

            except socket.timeout:
                continue  # normal — no data yet, loop back

    def _close_socket(self) -> None:
        """Safely tear down the TCP socket."""
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
        self._connected = False
