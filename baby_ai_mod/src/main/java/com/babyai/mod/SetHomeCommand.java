package com.babyai.mod;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.context.CommandContext;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Registers the {@code /sethome} command which marks the player's
 * current position as the Baby-AI "home" waypoint.
 *
 * <p>When executed the command:
 * <ol>
 *   <li>Grabs the player's current X/Y/Z coordinates.</li>
 *   <li>Broadcasts a {@code home_set} event via {@link EventBridge}
 *       so the Python training loop updates its home-proximity
 *       reward channel in real time.</li>
 *   <li>Sends a confirmation chat message to the player.</li>
 * </ol>
 *
 * <p>Works alongside the GUI "Set New Home" button — both paths
 * converge on the same {@code home_set} TCP event.
 */
public class SetHomeCommand {

    private static final Logger LOGGER = LoggerFactory.getLogger("baby-ai-bridge");

    /**
     * Register {@code /sethome} on the given dispatcher.
     *
     * <p>Called once from {@link BabyAiMod#onInitialize()} via
     * {@code CommandRegistrationCallback}.
     */
    public static void register(CommandDispatcher<ServerCommandSource> dispatcher) {
        dispatcher.register(
            CommandManager.literal("sethome")
                .requires(source -> source.isExecutedByPlayer())
                .executes(SetHomeCommand::execute)
        );
    }

    private static int execute(CommandContext<ServerCommandSource> ctx) {
        ServerPlayerEntity player = ctx.getSource().getPlayer();
        if (player == null) {
            return 0;
        }

        double x = Math.round(player.getX() * 100.0) / 100.0;
        double y = Math.round(player.getY() * 100.0) / 100.0;
        double z = Math.round(player.getZ() * 100.0) / 100.0;
        long tick = player.getServer() != null ? player.getServer().getTicks() : 0;

        // Save in the server-side HomeManager.
        HomeManager.INSTANCE.setHome(player.getUuid(), x, y, z);

        // Update Minecraft's actual respawn point so the player
        // respawns here natively (like sleeping in a bed).
        BlockPos spawnPos = BlockPos.ofFloored(x, y, z);
        player.setSpawnPoint(
            player.getWorld().getRegistryKey(), // current dimension
            spawnPos,
            0.0f,   // spawn angle
            true,    // forced — bypass bed/anchor checks
            true     // send update to client
        );

        // Broadcast to Python via the TCP event bridge.
        EventBridge.INSTANCE.onHomeSet(x, y, z, tick);

        // Confirm in chat.
        player.sendMessage(
            Text.literal(String.format(
                "§a[Baby-AI]§r Home set to §e%.1f§r, §e%.1f§r, §e%.1f§r", x, y, z
            )),
            false
        );

        LOGGER.info("[Baby-AI] Home set via /sethome: ({}, {}, {})", x, y, z);
        return 1;
    }
}
