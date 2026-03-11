package com.babyai.mod;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Lightweight TCP server that broadcasts game events as JSON lines
 * to all connected clients (the Python Baby-AI process).
 *
 * <p>Protocol is newline-delimited JSON (one object per line).  Each
 * event has at least an {@code "event"} key identifying the type.
 *
 * <h3>Event types</h3>
 * <ul>
 *   <li>{@code block_broken}  — block, x, y, z, tick</li>
 *   <li>{@code block_placed}  — block, x, y, z, tick</li>
 *   <li>{@code item_crafted}  — item, count, tick</li>
 *   <li>{@code item_picked_up} — item, count, tick</li>
 *   <li>{@code player_death}  — source, tick</li>
 *   <li>{@code health_changed} — old_health, new_health, delta, tick</li>
 *   <li>{@code food_changed}   — old_food, new_food, delta, tick</li>
 *   <li>{@code xp_gained}      — amount, total_level, tick</li>
 *   <li>{@code position_update} — x, y, z, pitch, yaw, on_ground, light, tick</li>
 *   <li>{@code player_status}  — health, max_health, food, saturation, armor,
 *        xp_level, xp_progress, air, max_air, is_sprinting, is_swimming,
 *        is_sneaking, is_on_fire, game_time, day_time, is_raining,
 *        is_thundering, biome, held_item, velocity_x, velocity_y, velocity_z,
 *        inventory_used_slots, tick</li>
 * </ul>
 */
public class EventBridge {

    public static final EventBridge INSTANCE = new EventBridge();
    public static final int DEFAULT_PORT = 5556;

    private static final Logger LOGGER = LoggerFactory.getLogger("baby-ai-bridge");

    private ServerSocket serverSocket;
    private final CopyOnWriteArrayList<PrintWriter> clients = new CopyOnWriteArrayList<>();
    private final AtomicBoolean running = new AtomicBoolean(false);
    private Thread acceptThread;

    /** Reference to the integrated server — set once the server starts. */
    private final AtomicReference<MinecraftServer> serverRef = new AtomicReference<>(null);

    // ── Server reference ───────────────────────────────────────

    /**
     * Store the {@link MinecraftServer} so command handlers can
     * freeze/resume ticks during System 2 planning.
     */
    public void setServer(MinecraftServer server) {
        serverRef.set(server);
    }

    // ── Lifecycle ──────────────────────────────────────────────

    public void start() {
        if (running.getAndSet(true)) return;

        try {
            serverSocket = new ServerSocket(DEFAULT_PORT, 4, InetAddress.getLoopbackAddress());
            LOGGER.info("Event bridge listening on localhost:{}", DEFAULT_PORT);
        } catch (IOException e) {
            LOGGER.error("Failed to bind port {}: {}", DEFAULT_PORT, e.getMessage());
            running.set(false);
            return;
        }

        acceptThread = new Thread(() -> {
            while (running.get()) {
                try {
                    Socket client = serverSocket.accept();
                    client.setTcpNoDelay(true);
                    PrintWriter writer = new PrintWriter(
                        new BufferedWriter(new OutputStreamWriter(
                            client.getOutputStream(), "UTF-8"
                        )),
                        /* autoFlush */ true
                    );
                    clients.add(writer);
                    LOGGER.info("Baby-AI client connected from {}",
                                client.getRemoteSocketAddress());

                    // Spawn a reader thread that listens for JSON
                    // commands from this client (e.g. pause/resume).
                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(client.getInputStream(), "UTF-8")
                    );
                    Thread readerThread = new Thread(
                        () -> readCommands(reader, writer),
                        "baby-ai-cmd-reader"
                    );
                    readerThread.setDaemon(true);
                    readerThread.start();
                } catch (IOException e) {
                    if (running.get()) {
                        LOGGER.warn("Accept failed: {}", e.getMessage());
                    }
                }
            }
        }, "baby-ai-accept");
        acceptThread.setDaemon(true);
        acceptThread.start();
    }

    public void stop() {
        running.set(false);
        try {
            if (serverSocket != null) serverSocket.close();
        } catch (IOException ignored) { }
        for (PrintWriter w : clients) w.close();
        clients.clear();
        LOGGER.info("Event bridge stopped.");
    }

    // ── Broadcasting ───────────────────────────────────────────

    private void broadcast(JsonObject json) {
        if (clients.isEmpty()) return;
        String line = json.toString();
        for (PrintWriter w : clients) {
            try {
                w.println(line);
                if (w.checkError()) {
                    clients.remove(w);
                }
            } catch (Exception e) {
                clients.remove(w);
            }
        }
    }

    // ── Event emitters (called from listeners / mixins) ────────

    public void onBlockBroken(String block, BlockPos pos, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "block_broken");
        j.addProperty("block", block);
        j.addProperty("x", pos.getX());
        j.addProperty("y", pos.getY());
        j.addProperty("z", pos.getZ());
        j.addProperty("tick", tick);
        broadcast(j);
    }

    public void onBlockPlaced(String block, BlockPos pos, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "block_placed");
        j.addProperty("block", block);
        j.addProperty("x", pos.getX());
        j.addProperty("y", pos.getY());
        j.addProperty("z", pos.getZ());
        j.addProperty("tick", tick);
        broadcast(j);
    }

    public void onItemCrafted(String item, int count, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "item_crafted");
        j.addProperty("item", item);
        j.addProperty("count", count);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    public void onItemPickup(String item, int count, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "item_picked_up");
        j.addProperty("item", item);
        j.addProperty("count", count);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    public void onPlayerDeath(String source, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "player_death");
        j.addProperty("source", source);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    public void onHeartbeat(long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "heartbeat");
        j.addProperty("tick", tick);
        j.addProperty("clients", clients.size());
        broadcast(j);
    }

    /**
     * Fired when the player's health changes (damage taken or healing).
     *
     * @param oldHealth HP before the change (0–20 in half-hearts).
     * @param newHealth HP after the change.
     * @param tick      Current server tick.
     */
    public void onHealthChanged(float oldHealth, float newHealth, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "health_changed");
        j.addProperty("old_health", oldHealth);
        j.addProperty("new_health", newHealth);
        j.addProperty("delta", newHealth - oldHealth);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    /**
     * Fired when the player's food/hunger level changes.
     *
     * @param oldFood Food level before the change (0–20).
     * @param newFood Food level after the change.
     * @param tick    Current server tick.
     */
    public void onFoodChanged(int oldFood, int newFood, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "food_changed");
        j.addProperty("old_food", oldFood);
        j.addProperty("new_food", newFood);
        j.addProperty("delta", newFood - oldFood);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    /**
     * Fired when the player gains experience points.
     *
     * @param amount     Number of XP points gained this tick.
     * @param totalLevel Current player level after the gain.
     * @param tick       Current server tick.
     */
    public void onXpGained(int amount, int totalLevel, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "xp_gained");
        j.addProperty("amount", amount);
        j.addProperty("total_level", totalLevel);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    /**
     * Periodic position update so the Python agent can track height,
     * detect falls, and avoid caves.
     *
     * @param x       Player X coordinate.
     * @param y       Player Y coordinate (height — sea level ≈ 63).
     * @param z       Player Z coordinate.
     * @param pitch   Camera pitch in degrees (−90 = straight up, +90 = down).
     * @param yaw     Camera yaw in degrees.
     * @param onGround True if the player is standing on a solid surface.
     * @param light   Block light level at the player's eye position (0–15).
     * @param tick    Current server tick.
     */
    public void onPositionUpdate(double x, double y, double z,
                                  float pitch, float yaw,
                                  boolean onGround, int light, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "position_update");
        j.addProperty("x", Math.round(x * 100.0) / 100.0);
        j.addProperty("y", Math.round(y * 100.0) / 100.0);
        j.addProperty("z", Math.round(z * 100.0) / 100.0);
        j.addProperty("pitch", pitch);
        j.addProperty("yaw", yaw);
        j.addProperty("on_ground", onGround);
        j.addProperty("light", light);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    /**
     * Rich periodic snapshot of the player's full status.
     *
     * <p>Sent every 10 ticks (~2 Hz) alongside position updates.
     * Provides all the structured game-state data that the Python
     * sensor encoder needs to build a complete observation: vitals,
     * movement state, environment conditions, and inventory summary.
     *
     * @param health        Current health (0–20, half-hearts).
     * @param maxHealth     Max health (usually 20).
     * @param food          Hunger level (0–20).
     * @param saturation    Saturation level (0–20).
     * @param armor         Armor defense value (0–20).
     * @param xpLevel       Experience level.
     * @param xpProgress    Progress to next level (0.0–1.0).
     * @param air           Remaining air ticks (300 = full, 0 = drowning).
     * @param maxAir        Maximum air ticks.
     * @param isSprinting   True if currently sprinting.
     * @param isSwimming    True if currently swimming.
     * @param isSneaking    True if currently sneaking.
     * @param isOnFire      True if the player is burning.
     * @param gameTime      Total world time in ticks (absolute).
     * @param dayTime       Time of day in ticks (0–24000, 0=sunrise).
     * @param isRaining     True if it's currently raining.
     * @param isThundering  True if thunderstorm active.
     * @param biome         Biome ID string (e.g. "minecraft:plains").
     * @param heldItem      ID of item in main hand (e.g. "minecraft:diamond_pickaxe").
     * @param velX          Player velocity X component.
     * @param velY          Player velocity Y component.
     * @param velZ          Player velocity Z component.
     * @param invUsedSlots  Number of inventory slots with items (0–36).
     * @param tick          Current server tick.
     */
    public void onPlayerStatus(
            float health, float maxHealth, int food, float saturation,
            int armor, int xpLevel, float xpProgress,
            int air, int maxAir,
            boolean isSprinting, boolean isSwimming,
            boolean isSneaking, boolean isOnFire,
            long gameTime, long dayTime,
            boolean isRaining, boolean isThundering,
            String biome, String heldItem,
            double velX, double velY, double velZ,
            int invUsedSlots, long tick
    ) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "player_status");
        // Vitals
        j.addProperty("health", health);
        j.addProperty("max_health", maxHealth);
        j.addProperty("food", food);
        j.addProperty("saturation", Math.round(saturation * 100.0f) / 100.0f);
        j.addProperty("armor", armor);
        j.addProperty("xp_level", xpLevel);
        j.addProperty("xp_progress", Math.round(xpProgress * 1000.0f) / 1000.0f);
        // Oxygen
        j.addProperty("air", air);
        j.addProperty("max_air", maxAir);
        // Movement flags
        j.addProperty("is_sprinting", isSprinting);
        j.addProperty("is_swimming", isSwimming);
        j.addProperty("is_sneaking", isSneaking);
        j.addProperty("is_on_fire", isOnFire);
        // World time
        j.addProperty("game_time", gameTime);
        j.addProperty("day_time", dayTime);
        j.addProperty("is_raining", isRaining);
        j.addProperty("is_thundering", isThundering);
        // Environment
        j.addProperty("biome", biome);
        j.addProperty("held_item", heldItem);
        // Velocity
        j.addProperty("velocity_x", Math.round(velX * 1000.0) / 1000.0);
        j.addProperty("velocity_y", Math.round(velY * 1000.0) / 1000.0);
        j.addProperty("velocity_z", Math.round(velZ * 1000.0) / 1000.0);
        // Inventory
        j.addProperty("inventory_used_slots", invUsedSlots);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    /**
     * Fired when the player sets a home waypoint via {@code /sethome}
     * or the GUI "Set New Home" button.
     *
     * <p>The Python agent uses this to update the home-proximity
     * reward channel so the AI is incentivised to stay near the
     * chosen base.
     *
     * @param x    Home X coordinate.
     * @param y    Home Y coordinate.
     * @param z    Home Z coordinate.
     * @param tick Current server tick.
     */
    public void onHomeSet(double x, double y, double z, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "home_set");
        j.addProperty("x", x);
        j.addProperty("y", y);
        j.addProperty("z", z);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    // ── Command reader (Python → Mod) ──────────────────────────

    /**
     * Reads newline-delimited JSON commands from a connected client.
     * Runs on a dedicated daemon thread per client.  When the client
     * disconnects the writer is removed from the broadcast list.
     */
    private void readCommands(BufferedReader reader, PrintWriter writer) {
        try {
            String line;
            while (running.get() && (line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                try {
                    JsonObject cmd = JsonParser.parseString(line).getAsJsonObject();
                    handleCommand(cmd);
                } catch (Exception e) {
                    LOGGER.warn("Bad command JSON: {}", line);
                }
            }
        } catch (IOException e) {
            if (running.get()) {
                LOGGER.debug("Command reader disconnected: {}", e.getMessage());
            }
        } finally {
            clients.remove(writer);
        }
    }

    /**
     * Dispatch a command received from the Python agent.
     *
     * <p>Supported commands:
     * <ul>
     *   <li>{@code {"command":"pause"}}   — freeze game ticks (System 2 thinking)</li>
     *   <li>{@code {"command":"resume"}}  — resume game ticks</li>
     * </ul>
     */
    private void handleCommand(JsonObject cmd) {
        String action = cmd.has("command") ? cmd.get("command").getAsString() : "";
        LOGGER.info("[Baby-AI] Received command: {}", action);
        switch (action) {
            case "pause" -> {
                MinecraftServer server = serverRef.get();
                if (server != null) {
                    server.execute(() -> {
                        server.getTickManager().setFrozen(true);
                        LOGGER.info("[Baby-AI] Game FROZEN (System 2 planning)");
                    });
                } else {
                    LOGGER.warn("[Baby-AI] Cannot pause — serverRef is null (server not started?)");
                }
            }
            case "resume" -> {
                MinecraftServer server = serverRef.get();
                if (server != null) {
                    server.execute(() -> {
                        server.getTickManager().setFrozen(false);
                        LOGGER.info("[Baby-AI] Game RESUMED");
                    });
                } else {
                    LOGGER.warn("[Baby-AI] Cannot resume — serverRef is null");
                }
            }
            default -> LOGGER.warn("[Baby-AI] Unknown command: {}", action);
        }
    }
}
