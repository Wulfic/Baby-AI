package com.babyai.mod;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Thread-safe singleton that stores per-player home waypoints.
 *
 * <p>Used by {@link SetHomeCommand} to save, {@link HomeCommand} to
 * teleport, and the death-respawn handler in {@link BabyAiMod} to
 * relocate the player after auto-respawn.
 *
 * <p>Home locations are kept in memory only — they are also persisted
 * on the Python side via the settings store for cross-session survival.
 * When the mod receives a {@code home_set} event response from Python
 * (or the player runs {@code /sethome}), the coordinates are cached here
 * so the server can teleport without a round-trip to Python.
 */
public class HomeManager {

    public static final HomeManager INSTANCE = new HomeManager();

    private static final Logger LOGGER = LoggerFactory.getLogger("baby-ai-bridge");

    /** Simple immutable record for a 3D position. */
    public record HomePos(double x, double y, double z) {}

    private final Map<UUID, HomePos> homes = new ConcurrentHashMap<>();

    private HomeManager() {}

    /**
     * Save a home location for the given player.
     */
    public void setHome(UUID playerId, double x, double y, double z) {
        homes.put(playerId, new HomePos(x, y, z));
        LOGGER.info("[Baby-AI] Home saved for {}: ({}, {}, {})", playerId, x, y, z);
    }

    /**
     * Retrieve the stored home location, or {@code null} if none is set.
     */
    public HomePos getHome(UUID playerId) {
        return homes.get(playerId);
    }

    /**
     * Check whether a home location is stored for the given player.
     */
    public boolean hasHome(UUID playerId) {
        return homes.containsKey(playerId);
    }

    /**
     * Remove a player's home (e.g. on disconnect cleanup if desired).
     */
    public void removeHome(UUID playerId) {
        homes.remove(playerId);
    }
}
