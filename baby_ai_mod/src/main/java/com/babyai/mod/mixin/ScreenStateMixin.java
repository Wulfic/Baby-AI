package com.babyai.mod.mixin;

import com.babyai.mod.EventBridge;
import net.minecraft.client.gui.screen.Screen;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Client-side mixin that tracks when ANY GUI screen is open.
 *
 * <p>{@link Screen} is the base class for every GUI overlay in
 * Minecraft — inventory, chests, crafting tables, the pause menu,
 * the death screen, etc.  By hooking {@code init()} and
 * {@code removed()} we get a reliable signal that the Python
 * side can use to avoid sending ESC/key presses while a screen
 * is active.
 *
 * <p>The state is stored on {@link EventBridge} as an
 * {@code AtomicBoolean} and included in the {@code player_status}
 * event every tick.
 */
@Mixin(Screen.class)
public abstract class ScreenStateMixin {

    /**
     * Called when a screen is initialised (opened or resized).
     * We record the screen class name so the Python side knows
     * which screen type is open.
     */
    @Inject(method = "init()V", at = @At("TAIL"))
    private void babyai$onScreenInit(CallbackInfo ci) {
        String name = this.getClass().getSimpleName();
        EventBridge.INSTANCE.setScreenOpen(name);
    }

    /**
     * Called when the screen is removed (closed).
     */
    @Inject(method = "removed", at = @At("HEAD"))
    private void babyai$onScreenRemoved(CallbackInfo ci) {
        EventBridge.INSTANCE.setScreenClosed();
    }
}
