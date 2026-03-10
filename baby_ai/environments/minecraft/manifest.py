"""
Minecraft version-manifest parsing helpers.

Extracted from ``launcher.py`` to keep individual modules under ~800 lines.
Contains pure functions for resolving Java runtimes, building classpaths,
and constructing JVM / game argument lists from Mojang-format version JSONs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from baby_ai.utils.logging import get_logger

log = get_logger("mc_manifest")

# ── Default paths ───────────────────────────────────────────────

# Java runtime shipped with the Microsoft Store MC launcher
_MS_STORE_RUNTIME = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Packages"
    / "Microsoft.4297127D64EC6_8wekyb3d8bbwe"
    / "LocalCache"
    / "Local"
    / "runtime"
)

# Java runtime shipped with the standalone MC launcher
_STANDALONE_RUNTIME = Path(os.environ.get("APPDATA", "")) / ".minecraft" / "runtime"


def find_java(mc_dir: Path, component: str = "java-runtime-delta") -> Path:
    """
    Locate the Java executable for a given runtime component.

    Search order:
    1. MC dir / runtime / <component>
    2. MS Store launcher cache / runtime / <component>
    3. Standalone launcher / runtime / <component>
    4. System PATH (``java`` / ``javaw``)

    Returns the path to ``javaw.exe`` (or ``java.exe`` as fallback).
    """
    candidates = [
        mc_dir / "runtime" / component / "windows-x64" / component / "bin",
        _MS_STORE_RUNTIME / component / "windows-x64" / component / "bin",
        _STANDALONE_RUNTIME / component / "windows-x64" / component / "bin",
    ]

    for bin_dir in candidates:
        javaw = bin_dir / "javaw.exe"
        if javaw.is_file():
            log.info("Found Java: %s", javaw)
            return javaw
        java = bin_dir / "java.exe"
        if java.is_file():
            log.info("Found Java (console): %s", java)
            return java

    # Fallback: system PATH
    import shutil
    for name in ("javaw", "java"):
        found = shutil.which(name)
        if found:
            log.warning("Using system Java (may not be correct version): %s", found)
            return Path(found)

    raise FileNotFoundError(
        f"Cannot find Java runtime component '{component}'. "
        "Make sure Minecraft has been launched at least once via the official launcher."
    )


def os_matches(os_rule: Dict[str, Any]) -> bool:
    """Check if an OS rule matches Windows."""
    name = os_rule.get("name", "")
    if name and name != "windows":
        return False
    arch = os_rule.get("arch", "")
    if arch and arch != "x86_64":
        return False
    return True


def rules_allow(rules: List[Dict[str, Any]]) -> bool:
    """
    Evaluate Mojang-style rules to determine if a library applies.

    Rules use allow/disallow actions with optional OS filters.
    No rules = always allowed.
    """
    if not rules:
        return True

    allowed = False
    for rule in rules:
        action = rule.get("action", "allow")
        os_info = rule.get("os")
        features = rule.get("features")

        # Feature-gated rules (e.g. is_demo_user) — skip unless
        # we explicitly need them (we handle quick play separately)
        if features:
            continue

        if os_info:
            if os_matches(os_info):
                allowed = action == "allow"
        else:
            # No OS constraint — applies to all platforms
            allowed = action == "allow"

    return allowed


def build_classpath(mc_dir: Path, version_data: Dict[str, Any]) -> str:
    """
    Build the Java classpath from the version manifest.

    Filters libraries by platform rules, verifies each JAR exists,
    and appends the version JAR itself.
    """
    libs_dir = mc_dir / "libraries"
    jars: List[str] = []
    missing: List[str] = []

    for lib in version_data.get("libraries", []):
        rules = lib.get("rules", [])
        if not rules_allow(rules):
            continue

        artifact = lib.get("downloads", {}).get("artifact")
        if artifact:
            path = artifact.get("path", "")
            if path:
                jar = libs_dir / path.replace("/", os.sep)
                if jar.is_file():
                    jars.append(str(jar))
                else:
                    missing.append(path)
        else:
            # Maven-style library (used by Fabric Loader) — derive
            # the JAR path from  group:artifact:version  notation.
            name = lib.get("name", "")
            if name:
                parts = name.split(":")
                if len(parts) >= 3:
                    group, art_name, ver = parts[0], parts[1], parts[2]
                    rel = (
                        group.replace(".", os.sep)
                        + os.sep + art_name
                        + os.sep + ver
                        + os.sep + f"{art_name}-{ver}.jar"
                    )
                    jar = libs_dir / rel
                    if jar.is_file():
                        jars.append(str(jar))
                    else:
                        missing.append(rel)

    # Append the version JAR
    version_id = version_data["id"]
    version_jar = mc_dir / "versions" / version_id / f"{version_id}.jar"
    if version_jar.is_file():
        jars.append(str(version_jar))
    else:
        raise FileNotFoundError(f"Version JAR not found: {version_jar}")

    if missing:
        log.warning("Missing %d library JARs (non-critical if OS-specific):", len(missing))
        for m in missing[:5]:
            log.warning("  %s", m)

    log.info("Classpath: %d JARs", len(jars))
    return ";".join(jars)  # semicolon separator on Windows


def build_jvm_args(
    version_data: Dict[str, Any],
    natives_dir: str,
    classpath: str,
    max_memory_mb: int = 2048,
    min_memory_mb: int = 512,
) -> List[str]:
    """Build JVM arguments from the version manifest."""
    args: List[str] = []

    # Memory
    args.append(f"-Xmx{max_memory_mb}M")
    args.append(f"-Xms{min_memory_mb}M")

    # Parse JVM args from manifest
    raw_jvm = version_data.get("arguments", {}).get("jvm", [])
    substitutions = {
        "${natives_directory}": natives_dir,
        "${launcher_name}": "baby-ai",
        "${launcher_version}": "1.0",
        "${classpath}": classpath,
    }

    for entry in raw_jvm:
        if isinstance(entry, str):
            # Simple string arg — substitute variables
            resolved = entry
            for key, val in substitutions.items():
                resolved = resolved.replace(key, val)
            args.append(resolved)
        elif isinstance(entry, dict):
            # Conditional arg with rules
            rules = entry.get("rules", [])
            if not rules_allow(rules):
                continue
            value = entry.get("value", "")
            if isinstance(value, list):
                for v in value:
                    resolved = v
                    for key, val in substitutions.items():
                        resolved = resolved.replace(key, val)
                    args.append(resolved)
            elif isinstance(value, str):
                resolved = value
                for key, val in substitutions.items():
                    resolved = resolved.replace(key, val)
                args.append(resolved)

    return args


def build_game_args(
    version_data: Dict[str, Any],
    mc_dir: Path,
    player_name: str,
    player_uuid: str,
    world: Optional[str] = None,
    width: int = 854,
    height: int = 480,
) -> List[str]:
    """Build game arguments from the version manifest."""
    version_id = version_data["id"]
    assets_index = version_data.get("assetIndex", {}).get("id", version_id)
    version_type = version_data.get("type", "release")

    substitutions = {
        "${auth_player_name}": player_name,
        "${version_name}": version_id,
        "${game_directory}": str(mc_dir),
        "${assets_root}": str(mc_dir / "assets"),
        "${assets_index_name}": assets_index,
        "${auth_uuid}": player_uuid,
        "${auth_access_token}": "0",
        "${clientid}": "",
        "${auth_xuid}": "",
        "${version_type}": version_type,
        "${resolution_width}": str(width),
        "${resolution_height}": str(height),
        "${quickPlayPath}": str(mc_dir / "quickPlay" / "java" / "log.json"),
        "${quickPlaySingleplayer}": world or "",
    }

    args: List[str] = []
    raw_game = version_data.get("arguments", {}).get("game", [])

    for entry in raw_game:
        if isinstance(entry, str):
            resolved = entry
            for key, val in substitutions.items():
                resolved = resolved.replace(key, val)
            args.append(resolved)
        elif isinstance(entry, dict):
            # Conditional arg — check features
            rules = entry.get("rules", [])
            features_required = set()
            skip = False
            for rule in rules:
                features = rule.get("features", {})
                features_required.update(features.keys())

            # Only enable features we want
            enabled_features = {"has_quick_plays_support", "has_custom_resolution"}
            if world:
                enabled_features.add("is_quick_play_singleplayer")

            # Check if ALL required features are in our enabled set
            if not features_required.issubset(enabled_features):
                continue

            value = entry.get("value", "")
            if isinstance(value, list):
                for v in value:
                    resolved = v
                    for key, val in substitutions.items():
                        resolved = resolved.replace(key, val)
                    args.append(resolved)
            elif isinstance(value, str):
                resolved = value
                for key, val in substitutions.items():
                    resolved = resolved.replace(key, val)
                args.append(resolved)

    return args


def find_natives_dir(mc_dir: Path, version_id: str) -> str:
    """
    Find the natives directory for the given MC version.

    MC stores extracted native DLLs in .minecraft/bin/<hash>/.
    We look for the most recently modified directory that contains
    lwjgl.dll as a heuristic.
    """
    bin_dir = mc_dir / "bin"
    if not bin_dir.is_dir():
        # Fallback: create a temp natives dir
        fallback = mc_dir / "versions" / version_id / f"{version_id}-natives"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)

    # Find the hash dir that has native DLLs
    best: Optional[Path] = None
    best_mtime = 0.0
    for entry in bin_dir.iterdir():
        if entry.is_dir():
            # Check for LWJGL native — good indicator
            if (entry / "lwjgl.dll").exists() or (entry / "glfw.dll").exists():
                mtime = entry.stat().st_mtime
                if mtime > best_mtime:
                    best = entry
                    best_mtime = mtime

    if best:
        log.info("Natives dir: %s", best)
        return str(best)

    # Fallback
    fallback = mc_dir / "versions" / version_id / f"{version_id}-natives"
    fallback.mkdir(parents=True, exist_ok=True)
    log.warning("No natives dir found, using fallback: %s", fallback)
    return str(fallback)
