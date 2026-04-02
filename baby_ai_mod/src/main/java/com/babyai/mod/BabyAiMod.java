package com.babyai.mod;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.entity.event.v1.ServerLivingEntityEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.player.AttackEntityCallback;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.util.ActionResult;
import net.minecraft.util.Hand;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.LightType;
import net.minecraft.world.biome.Biome;
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

        // ── Server started — store reference for command handling ──
        ServerLifecycleEvents.SERVER_STARTED.register(server -> {
            EventBridge.INSTANCE.setServer(server);
            LOGGER.info("[Baby-AI] Server reference stored for tick control");
        });

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

            // ── Position update (every 10 ticks = 0.5 seconds) ───
            // Sends player coordinates, camera angles, on_ground,
            // and light level so the Python agent can detect caves,
            // falls, and underground navigation.
            if (tick % 10 == 0) {
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

                    // ── Full player status snapshot ────────────
                    // Rich telemetry for the sensor encoder: vitals,
                    // movement state, world conditions, inventory.
                    Vec3d vel = player.getVelocity();
                    BlockPos feetPos = player.getBlockPos();
                    RegistryEntry<Biome> biomeEntry = player.getWorld().getBiome(feetPos);
                    String biomeId = biomeEntry.getKey()
                            .map(k -> k.getValue().toString())
                            .orElse("unknown");
                    ItemStack mainHand = player.getMainHandStack();
                    String heldItemId = mainHand.isEmpty()
                            ? "minecraft:air"
                            : Registries.ITEM.getId(mainHand.getItem()).toString();

                    // Count non-empty inventory slots (main 36 slots)
                    int usedSlots = 0;
                    for (int i = 0; i < player.getInventory().main.size(); i++) {
                        if (!player.getInventory().main.get(i).isEmpty()) {
                            usedSlots++;
                        }
                    }

                    EventBridge.INSTANCE.onPlayerStatus(
                        player.getHealth(), player.getMaxHealth(),
                        player.getHungerManager().getFoodLevel(),
                        player.getHungerManager().getSaturationLevel(),
                        player.getArmor(),
                        player.experienceLevel, player.experienceProgress,
                        player.getAir(), player.getMaxAir(),
                        player.isSprinting(), player.isSwimming(),
                        player.isSneaking(), player.isOnFire(),
                        player.getWorld().getTime(),
                        player.getWorld().getTimeOfDay(),
                        player.getWorld().isRaining(),
                        player.getWorld().isThundering(),
                        biomeId, heldItemId,
                        vel.x, vel.y, vel.z,
                        usedSlots, tick
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

        // ── Entity attack (Fabric API) ───────────────────────
        // Fires when the player swings at any entity (mob, animal,
        // item frame, etc.).  We emit an event so the Python reward
        // system can incentivise combat and self-defence.
        AttackEntityCallback.EVENT.register((player, world, hand, entity, hitResult) -> {
            if (world.isClient() || hand != Hand.MAIN_HAND) {
                return ActionResult.PASS;
            }
            if (!(entity instanceof LivingEntity target)) {
                return ActionResult.PASS;
            }

            String entityType = Registries.ENTITY_TYPE.getId(target.getType()).toString();
            String entityName = target.getName().getString();
            boolean hostile = target instanceof HostileEntity;

            // Estimate damage from the player's weapon base damage.
            // We read the health before/after on the next tick for
            // precise tracking, but here we just use getAttributeValue
            // for a rough approximation.
            float dmg = 1.0f;
            if (player instanceof ServerPlayerEntity serverPlayer) {
                try {
                    dmg = (float) serverPlayer.getAttributeValue(
                        net.minecraft.entity.attribute.EntityAttributes.GENERIC_ATTACK_DAMAGE
                    );
                } catch (Exception ignored) {
                    dmg = 1.0f;
                }
            }

            long tick = world.getServer() != null ? world.getServer().getTicks() : 0;
            EventBridge.INSTANCE.onEntityHit(entityType, entityName, hostile, dmg, tick);
            LOGGER.debug("[Baby-AI] Entity hit: {} (hostile={}, dmg={:.1f})", entityName, hostile, dmg);
            return ActionResult.PASS;
        });

        // ── Player death AND mob killed (Fabric API) ─────────
        ServerLivingEntityEvents.AFTER_DEATH.register((entity, damageSource) -> {
            // ── Case 1: Player death ──
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
                //
                // If a home waypoint is set, teleport the player
                // there after respawning instead of the world spawn.
                player.getServer().execute(() -> {
                    if (player.isDead()) {
                        ServerPlayerEntity respawned = player.getServer()
                              .getPlayerManager()
                              .respawnPlayer(player, false, Entity.RemovalReason.KILLED);
                        LOGGER.info("[Baby-AI] Auto-respawned player");

                        // Teleport to home waypoint if one exists.
                        HomeManager.HomePos home = HomeManager.INSTANCE
                                .getHome(respawned.getUuid());
                        if (home != null) {
                            respawned.teleport(
                                respawned.getServerWorld(),
                                home.x(), home.y(), home.z(),
                                java.util.Set.of(),
                                respawned.getYaw(),
                                respawned.getPitch()
                            );
                            LOGGER.info("[Baby-AI] Teleported to home after respawn: ({}, {}, {})",
                                        home.x(), home.y(), home.z());
                        }
                    }
                });
            }

            // ── Case 2: Non-player mob killed by the player ──
            if (!(entity instanceof ServerPlayerEntity)
                    && damageSource.getAttacker() instanceof ServerPlayerEntity) {
                String entityType = Registries.ENTITY_TYPE.getId(entity.getType()).toString();
                String entityName = entity.getName().getString();
                boolean hostile = entity instanceof HostileEntity;
                long tick = entity.getServer() != null
                        ? entity.getServer().getTicks() : 0;
                EventBridge.INSTANCE.onMobKilled(entityType, entityName, hostile, tick);
                LOGGER.info("[Baby-AI] Mob killed: {} (hostile={})", entityName, hostile);
            }
        });

        // ── /sethome and /home commands ────────────────────────
        CommandRegistrationCallback.EVENT.register(
            (dispatcher, registryAccess, environment) -> {
                SetHomeCommand.register(dispatcher);
                HomeCommand.register(dispatcher);
            }
        );

        // ── Shutdown cleanup ─────────────────────────────────
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            LOGGER.info("[Baby-AI] Server stopping — shutting down bridge");
            EventBridge.INSTANCE.setServer(null);
            EventBridge.INSTANCE.stop();
        });

        LOGGER.info("[Baby-AI] Bridge mod ready — TCP on localhost:{}",
                     EventBridge.DEFAULT_PORT);
    }
}
