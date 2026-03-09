package com.babyai.mod;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.minecraft.entity.Entity;
import net.minecraft.registry.Registries;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.LightType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Baby-AI Bridge — Fabric mod entry point.
 *
 * Registers game-event listeners that forward signals to the Python
 * training loop via {@link EventBridge}.  Block breaks and player
 * deaths use Fabric API callbacks; block placement, item pickup, and
 * crafting are handled by Mixin injections (see {@code mixin/} package).
 */
public class BabyAiMod implements ModInitializer {

    public static final String MOD_ID = "baby-ai-bridge";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    // Per-player stat snapshots for delta tracking each tick
    private final Map<UUID, Float> lastHealth = new HashMap<>();
    private final Map<UUID, Integer> lastFood = new HashMap<>();
    private final Map<UUID, Integer> lastXpTotal = new HashMap<>();

    @Override
    public void onInitialize() {
        LOGGER.info("[Baby-AI] Bridge mod initializing...");

        // Start the TCP event bridge (accepts Python connections)
        EventBridge.INSTANCE.start();

        // ── Heartbeat tick (every 100 ticks = 5 seconds) ─────
        // Lets the Python client verify the event pipeline works
        // even when no game events are happening.
        //
        // Also tracks per-tick health, food, and XP deltas for all
        // online players so we can emit fine-grained reward signals.
        ServerTickEvents.END_SERVER_TICK.register(server -> {
            long tick = server.getTicks();
            if (tick % 100 == 0) {
                EventBridge.INSTANCE.onHeartbeat(tick);
            }

            // ── Position update (every 20 ticks = 1 second) ────
            // Sends player coordinates, camera angles, on_ground,
            // and light level so the Python agent can detect caves,
            // falls, and underground navigation.
            if (tick % 20 == 0) {
                for (ServerPlayerEntity player : server.getPlayerManager().getPlayerList()) {
                    BlockPos eyePos = BlockPos.ofFloored(
                        player.getX(), player.getEyeY(), player.getZ()
                    );
                    int light = player.getWorld().getLightLevel(LightType.SKY, eyePos);
                    EventBridge.INSTANCE.onPositionUpdate(
                        player.getX(), player.getY(), player.getZ(),
                        player.getPitch(), player.getYaw(),
                        player.isOnGround(), light, tick
                    );
                }
            }

            // ── Per-player stat delta tracking (every tick) ────
            for (ServerPlayerEntity player : server.getPlayerManager().getPlayerList()) {
                UUID uid = player.getUuid();

                // Health
                float curHealth = player.getHealth();
                Float prevHealth = lastHealth.get(uid);
                if (prevHealth != null && Math.abs(curHealth - prevHealth) > 0.01f) {
                    EventBridge.INSTANCE.onHealthChanged(prevHealth, curHealth, tick);
                    if (curHealth < prevHealth) {
                        LOGGER.debug("[Baby-AI] Damage: {:.1f} -> {:.1f}",
                                     prevHealth, curHealth);
                    } else {
                        LOGGER.debug("[Baby-AI] Heal: {:.1f} -> {:.1f}",
                                     prevHealth, curHealth);
                    }
                }
                lastHealth.put(uid, curHealth);

                // Food level
                int curFood = player.getHungerManager().getFoodLevel();
                Integer prevFood = lastFood.get(uid);
                if (prevFood != null && curFood != prevFood) {
                    EventBridge.INSTANCE.onFoodChanged(prevFood, curFood, tick);
                }
                lastFood.put(uid, curFood);

                // XP (use total experience points for precise delta)
                int curXp = player.totalExperience;
                Integer prevXp = lastXpTotal.get(uid);
                if (prevXp != null && curXp > prevXp) {
                    int gained = curXp - prevXp;
                    EventBridge.INSTANCE.onXpGained(
                        gained, player.experienceLevel, tick
                    );
                }
                lastXpTotal.put(uid, curXp);
            }
        });

        // ── Block break (Fabric API) ─────────────────────────
        PlayerBlockBreakEvents.AFTER.register((world, player, pos, state, blockEntity) -> {
            if (!world.isClient()) {
                String blockId = Registries.BLOCK.getId(state.getBlock()).toString();
                long tick = world.getServer() != null ? world.getServer().getTicks() : 0;
                LOGGER.info("[Baby-AI] Block broken: {} at ({},{},{})", blockId, pos.getX(), pos.getY(), pos.getZ());
                EventBridge.INSTANCE.onBlockBroken(blockId, pos, tick);
            }
        });

        // ── Player death (Fabric API) ────────────────────────
        ServerLivingEntityEvents.AFTER_DEATH.register((entity, damageSource) -> {
            if (entity instanceof ServerPlayerEntity player) {
                // Extract the death message for logging.  The exact
                // API surface may shift between MC versions — the
                // try/catch keeps the mod from crashing on changes.
                String deathMsg = "unknown";
                try {
                    deathMsg = entity.getDamageTracker()
                                     .getDeathMessage()
                                     .getString();
                } catch (Exception ignored) { }

                long tick = entity.getServer() != null
                        ? entity.getServer().getTicks() : 0;
                EventBridge.INSTANCE.onPlayerDeath(deathMsg, tick);
                LOGGER.info("[Baby-AI] Player death: {}", deathMsg);

                // ── Auto-respawn ────────────────────────────────
                // Immediately respawn the player server-side so the
                // Python trainer never has to deal with the death
                // screen.  Scheduled for next tick to ensure the
                // death event processing is complete.
                player.getServer().execute(() -> {
                    if (player.isDead()) {
                        player.getServer().getPlayerManager()
                              .respawnPlayer(player, false, Entity.RemovalReason.KILLED);
                        LOGGER.info("[Baby-AI] Auto-respawned player");
                    }
                });
            }
        });

        // ── Shutdown cleanup ─────────────────────────────────
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            LOGGER.info("[Baby-AI] Server stopping — shutting down bridge");
            EventBridge.INSTANCE.stop();
        });

        LOGGER.info("[Baby-AI] Bridge mod ready — TCP on localhost:{}",
                     EventBridge.DEFAULT_PORT);
    }
}
