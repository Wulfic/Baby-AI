package com.babyai.mod.mixin;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.DeathScreen;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Auto-respawn the player on the client side when the death screen
 * appears.  Vanilla enables the respawn button after 20 ticks (1 s);
 * we fire {@code player.requestRespawn()} at exactly that moment and
 * close the screen, so the AI never has to deal with the death overlay.
 *
 * <p>The server-side respawn is already handled by {@code BabyAiMod}'s
 * {@code AFTER_DEATH} listener — this mixin covers the <b>client</b>
 * GUI so the death screen is dismissed without a mouse click.
 */
@Mixin(DeathScreen.class)
public abstract class DeathScreenMixin {

    private static final Logger LOGGER = LoggerFactory.getLogger("Baby-AI");

    @Shadow
    private int ticksSinceDeath;

    @Inject(method = "tick", at = @At("TAIL"))
    private void babyai$autoRespawn(CallbackInfo ci) {
        // Vanilla enables buttons at ticksSinceDeath == 20.
        // We trigger respawn at that exact moment.
        if (this.ticksSinceDeath == 20) {
            MinecraftClient client = MinecraftClient.getInstance();
            if (client.player != null) {
                client.player.requestRespawn();
                client.setScreen(null);
                LOGGER.info("[Baby-AI] Client auto-respawn triggered");
            }
        }
    }
}
