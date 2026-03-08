package com.babyai.mod;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.minecraft.registry.Registries;
import net.minecraft.server.network.ServerPlayerEntity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    @Override
    public void onInitialize() {
        LOGGER.info("[Baby-AI] Bridge mod initializing...");

        // Start the TCP event bridge (accepts Python connections)
        EventBridge.INSTANCE.start();

        // ── Heartbeat tick (every 100 ticks = 5 seconds) ─────
        // Lets the Python client verify the event pipeline works
        // even when no game events are happening.
        ServerTickEvents.END_SERVER_TICK.register(server -> {
            long tick = server.getTicks();
            if (tick % 100 == 0) {
                EventBridge.INSTANCE.onHeartbeat(tick);
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
            if (entity instanceof ServerPlayerEntity) {
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
