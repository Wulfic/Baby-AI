package com.babyai.mod.mixin;

import com.babyai.mod.EventBridge;
import net.minecraft.client.Mouse;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Prevents Minecraft from capturing (grabbing) the mouse cursor
 * while the AI Python client is connected.
 *
 * <p>Cancels {@code lockCursor()} so GLFW never hides or confines
 * the cursor, and cancels {@code onCursorPos()} so physical mouse
 * movement doesn't rotate the in-game camera.
 *
 * <p>The AI uses PostMessage for clicks and the mod bridge TCP
 * {@code look} command for camera rotation, so neither path goes
 * through GLFW mouse callbacks.
 */
@Mixin(Mouse.class)
public abstract class MouseGrabMixin {

    /**
     * Cancel cursor locking when the AI is connected and NOT in passthrough
     * mode.  In passthrough mode (record-only / imitation) we allow
     * lockCursor() through so Minecraft can re-grab the cursor after a GUI
     * screen (inventory, pause menu) is closed.
     */
    @Inject(method = "lockCursor", at = @At("HEAD"), cancellable = true)
    private void babyai$preventCursorLock(CallbackInfo ci) {
        if (EventBridge.INSTANCE.hasClients() && !EventBridge.INSTANCE.isMousePassthrough()) {
            ci.cancel();
        }
    }

    /**
     * Block physical mouse movement from rotating the camera.
     * Skipped when mouse_passthrough is active (record-only / imitation mode)
     * so the human player retains full camera control.
     */
    @Inject(method = "onCursorPos", at = @At("HEAD"), cancellable = true)
    private void babyai$blockPhysicalMouseMove(long window, double x, double y, CallbackInfo ci) {
        if (EventBridge.INSTANCE.hasClients() && !EventBridge.INSTANCE.isMousePassthrough()) {
            ci.cancel();
        }
    }
}
