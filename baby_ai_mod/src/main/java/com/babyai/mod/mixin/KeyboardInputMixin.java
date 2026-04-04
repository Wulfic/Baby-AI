package com.babyai.mod.mixin;

import com.babyai.mod.EventBridge;
import net.minecraft.client.Keyboard;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Prevents Minecraft from processing physical keyboard input while
 * the AI Python client is connected.
 *
 * <p>GLFW delivers physical key events through {@code Keyboard.onKey()}.
 * The AI sends keys via Win32 {@code PostMessage}, which goes through
 * the GLFW window message pump and reaches {@code onKey()} as normal
 * GLFW key events — they are indistinguishable from physical keys.
 *
 * <p>However, the user's physical keypresses also arrive through
 * {@code onKey()} and would conflict with AI actions (e.g. the user
 * accidentally pressing W while the AI wants to stand still, or
 * pressing ESC and opening the pause menu).
 *
 * <p>Since PostMessage events and physical events both go through
 * {@code onKey()}, we cannot simply block all calls.  Instead, we
 * rely on the Python-side {@code InputGuard} (low-level Win32 hooks)
 * to block physical key events <b>before</b> they enter the GLFW
 * message queue.  This mixin is kept minimal — it only blocks the
 * {@code onKey()} path for F-keys and ESC that the InputGuard
 * explicitly allows through for Windows system shortcuts.
 *
 * <p><b>Note:</b> The primary keyboard blocking is handled by the
 * Python InputGuard.  This mixin acts as a safety net.
 */
@Mixin(Keyboard.class)
public abstract class KeyboardInputMixin {

    /**
     * Block physical key events that slip through the InputGuard.
     *
     * <p>We detect "likely physical" events by checking for keys that
     * the InputGuard intentionally passes through (ESC, F-keys, Tab)
     * because they serve as Windows system shortcuts — but we do NOT
     * want Minecraft itself to process them and open pause menus or
     * toggle debug screens.
     *
     * <p>Parameters: onKey(long window, int key, int scancode, int action, int modifiers)
     */
    @Inject(method = "onKey", at = @At("HEAD"), cancellable = true)
    private void babyai$filterKeyEvents(long window, int key, int scancode, int action, int modifiers, CallbackInfo ci) {
        if (!EventBridge.INSTANCE.hasClients()) {
            return;  // No AI connected — let everything through
        }

        // Block ESC (256), Tab (258), F3 (292), F5 (294) from being
        // processed by Minecraft.  These keys pass through the
        // InputGuard's Win32 hooks (needed for Alt+Tab, etc.) but
        // should NOT open MC's pause menu or debug overlays.
        // GLFW key codes: ESC=256, TAB=258, F1=290, F2=291, F3=292, ...
        if (key == 256 || key == 258 || key == 292 || key == 294) {
            ci.cancel();
        }
    }
}
