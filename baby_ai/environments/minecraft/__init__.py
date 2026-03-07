"""
Minecraft environment for Baby-AI.

Enables the agent to play Minecraft by:
- Screen-capturing the Minecraft window (via mss)
- Sending keyboard/mouse input via Win32 PostMessage (no input hijacking)
- Defining a 128-action discrete space covering movement, combat, and look
- Auto-launching Minecraft with the correct version and world
- Blocking user input when the AI is in control (input guard)

Requirements:
- Minecraft must be running in windowed mode (or use auto_launch=True)
- For camera look: disable "Raw Input" in Minecraft's Controls settings
- For background mode: only keyboard + mouse click actions (no camera look)
- For active mode: full control including camera look (uses SetCursorPos)
"""

from baby_ai.environments.minecraft.env import MinecraftEnv
from baby_ai.environments.minecraft.focus_guard import FocusGuard
from baby_ai.environments.minecraft.launcher import MinecraftLauncher

__all__ = ["MinecraftEnv", "MinecraftLauncher", "FocusGuard"]
