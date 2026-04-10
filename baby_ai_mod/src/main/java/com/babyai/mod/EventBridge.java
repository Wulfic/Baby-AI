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

    /**
     * When true, physical mouse movement is passed through to Minecraft
     * even while a client is connected.  Set to true in record-only /
     * imitation modes so the human player can look around normally.
     */
    private final AtomicBoolean mousePassthrough = new AtomicBoolean(false);

    /** True when physical mouse movement should reach Minecraft (e.g. record-only mode). */
    public boolean isMousePassthrough() {
        return mousePassthrough.get();
    }

    // ── Client screen state (set by ScreenStateMixin) ──────────

    /**
     * True while ANY GUI screen is open on the client (inventory,
     * chest, crafting table, pause menu, death screen, etc.).
     * Written by the client-side {@code ScreenStateMixin}, read
     * by the server tick when building {@code player_status}.
     */
    private final AtomicBoolean screenOpen = new AtomicBoolean(false);
    private final AtomicReference<String> openScreenName = new AtomicReference<>("");

    /** Called by {@code ScreenStateMixin} when any screen opens. */
    public void setScreenOpen(String screenClassName) {
        screenOpen.set(true);
        openScreenName.set(screenClassName != null ? screenClassName : "");
    }

    /** Called by {@code ScreenStateMixin} when the screen closes. */
    public void setScreenClosed() {
        screenOpen.set(false);
        openScreenName.set("");
    }

    /** Check if any GUI screen is currently open (thread-safe). */
    public boolean isScreenOpen() {
        return screenOpen.get();
    }

    /** Name of the currently open screen class, or empty string. */
    public String getOpenScreenName() {
        return openScreenName.get();
    }

    // ── Server reference ───────────────────────────────────────

    /**
     * True when at least one Python AI client is connected.
     * Used by mixins to disable cursor grab and keyboard processing
     * so the user's physical mouse/keyboard are never captured.
     */
    public boolean hasClients() {
        return !clients.isEmpty();
    }

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

    /**
     * Fired when the player hits (attacks) any entity.
     *
     * @param entityType Registry ID of the target entity (e.g. "minecraft:zombie").
     * @param entityName Display name (e.g. "Zombie", "Cow").
     * @param isHostile  True if the entity is a hostile mob.
     * @param damage     Approximate damage dealt this swing.
     * @param tick       Current server tick.
     */
    public void onEntityHit(String entityType, String entityName,
                            boolean isHostile, float damage, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "entity_hit");
        j.addProperty("entity_type", entityType);
        j.addProperty("entity_name", entityName);
        j.addProperty("is_hostile", isHostile);
        j.addProperty("damage", Math.round(damage * 100.0f) / 100.0f);
        j.addProperty("tick", tick);
        broadcast(j);
    }

    /**
     * Fired when an entity dies and the player was the attacker.
     *
     * @param entityType Registry ID of the killed entity.
     * @param entityName Display name.
     * @param isHostile  True if the entity was hostile.
     * @param tick       Current server tick.
     */
    public void onMobKilled(String entityType, String entityName,
                            boolean isHostile, long tick) {
        JsonObject j = new JsonObject();
        j.addProperty("event", "mob_killed");
        j.addProperty("entity_type", entityType);
        j.addProperty("entity_name", entityName);
        j.addProperty("is_hostile", isHostile);
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
        // Client screen state (from ScreenStateMixin)
        j.addProperty("has_open_screen", screenOpen.get());
        j.addProperty("open_screen_name", openScreenName.get());
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
                    handleCommand(cmd, writer);
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
     * <p>After the server thread applies a freeze/resume, an acknowledgment
     * event ({@code pause_ack} / {@code resume_ack}) is sent back to the
     * requesting client so it can wait for confirmation rather than relying
     * on a fixed sleep timer.
     *
     * <p>Supported commands:
     * <ul>
     *   <li>{@code {"command":"pause"}}   — freeze game ticks (System 2/3 thinking)</li>
     *   <li>{@code {"command":"resume"}}  — resume game ticks</li>
     *   <li>{@code {"command":"look", "dyaw":float, "dpitch":float}}
     *        — rotate the player camera by the given degree deltas
     *        (executed on the client render thread so the change is
     *        immediate and bypasses GLFW raw-input)</li>
     * </ul>
     */
    private void handleCommand(JsonObject cmd, PrintWriter clientWriter) {
        String action = cmd.has("command") ? cmd.get("command").getAsString() : "";
        String reason = cmd.has("reason") ? cmd.get("reason").getAsString() : "";
        LOGGER.info("[Baby-AI] Received command: {} (reason: {})", action, reason);
        switch (action) {
            case "pause" -> {
                MinecraftServer server = serverRef.get();
                if (server != null) {
                    server.execute(() -> {
                        server.getTickManager().setFrozen(true);
                        LOGGER.info("[Baby-AI] Game FROZEN ({})", reason);
                        // Acknowledge back to the Python client so it knows
                        // the freeze is in effect before it starts planning.
                        JsonObject ack = new JsonObject();
                        ack.addProperty("event", "pause_ack");
                        ack.addProperty("frozen", true);
                        ack.addProperty("reason", reason);
                        clientWriter.println(ack.toString());
                    });
                } else {
                    LOGGER.warn("[Baby-AI] Cannot pause — serverRef is null (server not started?)");
                    JsonObject ack = new JsonObject();
                    ack.addProperty("event", "pause_ack");
                    ack.addProperty("frozen", false);
                    ack.addProperty("error", "server_not_available");
                    clientWriter.println(ack.toString());
                }
            }
            case "resume" -> {
                MinecraftServer server = serverRef.get();
                if (server != null) {
                    server.execute(() -> {
                        server.getTickManager().setFrozen(false);
                        LOGGER.info("[Baby-AI] Game RESUMED ({})", reason);
                        JsonObject ack = new JsonObject();
                        ack.addProperty("event", "resume_ack");
                        ack.addProperty("frozen", false);
                        ack.addProperty("reason", reason);
                        clientWriter.println(ack.toString());
                    });
                } else {
                    LOGGER.warn("[Baby-AI] Cannot resume — serverRef is null");
                    JsonObject ack = new JsonObject();
                    ack.addProperty("event", "resume_ack");
                    ack.addProperty("frozen", true);
                    ack.addProperty("error", "server_not_available");
                    clientWriter.println(ack.toString());
                }
            }
            case "look" -> {
                // Rotate the player camera by (dyaw, dpitch) degrees.
                // Executed on the client render thread so the change is
                // reflected immediately without waiting for a server tick,
                // and bypasses GLFW raw-input — no cursor warp needed.
                float dyaw = cmd.has("dyaw") ? cmd.get("dyaw").getAsFloat() : 0f;
                float dpitch = cmd.has("dpitch") ? cmd.get("dpitch").getAsFloat() : 0f;

                net.minecraft.client.MinecraftClient client = net.minecraft.client.MinecraftClient.getInstance();
                client.execute(() -> {
                    if (client.player != null) {
                        float newYaw = client.player.getYaw() + dyaw;
                        float newPitch = client.player.getPitch() + dpitch;
                        // Clamp pitch to Minecraft's limits (-90 up, +90 down)
                        newPitch = Math.max(-90.0f, Math.min(90.0f, newPitch));

                        client.player.setYaw(newYaw);
                        client.player.setPitch(newPitch);
                        // Update head yaw so third-person rendering and
                        // server-side packets reflect the new orientation.
                        client.player.headYaw = newYaw;
                    }
                });
            }
            case "mouse_passthrough" -> {
                boolean enabled = cmd.has("enabled") && cmd.get("enabled").getAsBoolean();
                mousePassthrough.set(enabled);
                LOGGER.info("[Baby-AI] Mouse passthrough: {}", enabled ? "ON" : "OFF");
            }
            case "set_home" -> {
                // Python is pushing a home coordinate so HomeManager stays in sync
                // with the Python settings store (set_home GUI button / restored home).
                double hx = cmd.has("x") ? cmd.get("x").getAsDouble() : 0.0;
                double hy = cmd.has("y") ? cmd.get("y").getAsDouble() : 64.0;
                double hz = cmd.has("z") ? cmd.get("z").getAsDouble() : 0.0;
                MinecraftServer server = serverRef.get();
                if (server != null) {
                    server.execute(() -> {
                        net.minecraft.server.network.ServerPlayerEntity player =
                            server.getPlayerManager().getPlayerList().stream()
                                .findFirst().orElse(null);
                        if (player != null) {
                            HomeManager.INSTANCE.setHome(player.getUuid(), hx, hy, hz);
                        } else {
                            LOGGER.warn("[Baby-AI] set_home: no player found, storing with null UUID");
                        }
                    });
                } else {
                    LOGGER.warn("[Baby-AI] set_home: serverRef is null");
                }
            }
            case "goto_home" -> {
                // Teleport the player to their saved home waypoint.
                // Triggered automatically after the AI accumulates >30 s of
                // idle penalties (stuck safeguard — prevents training on bad data).
                MinecraftServer server = serverRef.get();
                if (server != null) {
                    server.execute(() -> {
                        net.minecraft.server.network.ServerPlayerEntity player =
                            server.getPlayerManager().getPlayerList().stream()
                                .findFirst().orElse(null);
                        if (player == null) {
                            LOGGER.warn("[Baby-AI] goto_home: no player found");
                            return;
                        }
                        HomeManager.HomePos home =
                            HomeManager.INSTANCE.getHome(player.getUuid());
                        if (home == null) {
                            LOGGER.warn("[Baby-AI] goto_home: no home set for player {}",
                                        player.getName().getString());
                            return;
                        }
                        net.minecraft.server.world.ServerWorld world =
                            player.getServerWorld();
                        player.teleport(
                            world,
                            home.x(), home.y(), home.z(),
                            java.util.Set.of(),
                            player.getYaw(),
                            player.getPitch()
                        );
                        LOGGER.info("[Baby-AI] Teleported AI to home ({}, {}, {}) after idle safeguard",
                                    home.x(), home.y(), home.z());
                    });
                } else {
                    LOGGER.warn("[Baby-AI] goto_home: serverRef is null");
                }
            }
            default -> LOGGER.warn("[Baby-AI] Unknown command: {}", action);
        }
    }
}
