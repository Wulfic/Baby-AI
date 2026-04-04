package com.babyai.mod.mixin;

import net.minecraft.util.SystemDetails;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Skips the oshi hardware detection during startup.
 *
 * <p>Minecraft calls {@code SystemDetails.addHardwareGroup()} which
 * queries the CPU through oshi → WMI.  When the Windows WMI service
 * is unresponsive the call blocks the main thread indefinitely.
 *
 * <p>This mixin cancels that method so MC can start even when WMI
 * is broken.  The only effect is that F3 debug and crash reports
 * won't list CPU/RAM details — acceptable for an AI agent.
 */
@Mixin(SystemDetails.class)
public abstract class SystemDetailsMixin {

    @Inject(method = "addHardwareGroup", at = @At("HEAD"), cancellable = true)
    private void babyai$skipOshiHardwareQuery(oshi.SystemInfo si, CallbackInfo ci) {
        ci.cancel();
    }
}
