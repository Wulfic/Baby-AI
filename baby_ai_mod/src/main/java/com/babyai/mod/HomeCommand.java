package com.babyai.mod;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.context.CommandContext;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.text.Text;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Set;

/**
 * Registers the {@code /home} command which teleports the player
 * to their saved home waypoint.
 *
 * <p>If no home has been set yet the player receives a hint to use
 * {@code /sethome} first.
 */
public class HomeCommand {

    private static final Logger LOGGER = LoggerFactory.getLogger("baby-ai-bridge");

    /**
     * Register {@code /home} on the given dispatcher.
     */
    public static void register(CommandDispatcher<ServerCommandSource> dispatcher) {
        dispatcher.register(
            CommandManager.literal("home")
                .requires(source -> source.isExecutedByPlayer())
                .executes(HomeCommand::execute)
        );
    }

    private static int execute(CommandContext<ServerCommandSource> ctx) {
        ServerPlayerEntity player = ctx.getSource().getPlayer();
        if (player == null) {
            return 0;
        }

        HomeManager.HomePos home = HomeManager.INSTANCE.getHome(player.getUuid());
        if (home == null) {
            player.sendMessage(
                Text.literal("§c[Baby-AI]§r No home set! Use §e/sethome§r first."),
                false
            );
            return 0;
        }

        // Teleport the player to home in their current world.
        ServerWorld world = player.getServerWorld();
        player.teleport(
            world,
            home.x(), home.y(), home.z(),
            Set.of(),         // no relative flags — absolute coordinates
            player.getYaw(),  // keep current yaw
            player.getPitch() // keep current pitch
        );

        player.sendMessage(
            Text.literal(String.format(
                "§a[Baby-AI]§r Teleported home to §e%.1f§r, §e%.1f§r, §e%.1f§r",
                home.x(), home.y(), home.z()
            )),
            false
        );

        LOGGER.info("[Baby-AI] Teleported player to home: ({}, {}, {})",
                     home.x(), home.y(), home.z());
        return 1;
    }
}
