"""
Minecraft environment for Baby-AI.

Enables the agent to play Minecraft by:
- Screen-capturing the Minecraft window (via mss)
- Sending keyboard/mouse input via Win32 PostMessage (no input hijacking)
- Defining a 128-action discrete space covering movement, combat, and look
- Auto-launching Minecraft with the correct version and world
- Blocking user input when the AI is in control (input guard)

Input modes:
- ``"virtual"``    — camera via mod bridge (no cursor hijack, user keeps
                     full control of keyboard and mouse)
- ``"background"`` — PostMessage only (no camera look — for GUI-only tasks)
- ``"active"``     — camera via SetCursorPos (moves the real cursor, legacy)

Requirements:
- Minecraft must be running in windowed mode (or use auto_launch=True)
- For virtual mode: the Baby-AI Fabric bridge mod must be installed
- For active mode: disable "Raw Input" in Minecraft's Controls settings
"""

from baby_ai.environments.minecraft.env import MinecraftEnv
from baby_ai.environments.minecraft.focus_guard import FocusGuard
from baby_ai.environments.minecraft.launcher import MinecraftLauncher
from baby_ai.environments.minecraft.virtual_input import VirtualInputController

__all__ = ["MinecraftEnv", "MinecraftLauncher", "FocusGuard", "VirtualInputController"]
