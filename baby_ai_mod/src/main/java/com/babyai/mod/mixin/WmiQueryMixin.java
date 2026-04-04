package com.babyai.mod.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import com.sun.jna.platform.win32.COM.WbemcliUtil;

/**
 * Prevents oshi from making WMI queries that hang when the Windows
 * WMI service is unresponsive.
 *
 * <p>MC calls oshi in two places (SystemDetails and GLX._init),
 * both of which eventually call
 * {@code WmiQueryHandler.queryWMI()}.  When WMI is broken the
 * native COM {@code ConnectServer()} call blocks the thread
 * indefinitely.
 *
 * <p>This mixin cancels all WMI queries and returns an empty result,
 * which makes oshi fall back to default/unknown values.
 */
@Mixin(targets = "oshi.util.platform.windows.WmiQueryHandler", remap = false)
public abstract class WmiQueryMixin {

    @SuppressWarnings({"rawtypes", "unchecked"})
    @Inject(method = "queryWMI(Lcom/sun/jna/platform/win32/COM/WbemcliUtil$WmiQuery;Z)Lcom/sun/jna/platform/win32/COM/WbemcliUtil$WmiResult;",
            at = @At("HEAD"), cancellable = true, remap = false)
    private void babyai$skipWmiQuery(WbemcliUtil.WmiQuery query, boolean initCom, CallbackInfoReturnable cir) {
        // Return an empty result — oshi handles missing data gracefully
        cir.setReturnValue(com.sun.jna.platform.win32.COM.WbemcliUtil.INSTANCE.new WmiResult<>(query.getPropertyEnum()));
    }
}
