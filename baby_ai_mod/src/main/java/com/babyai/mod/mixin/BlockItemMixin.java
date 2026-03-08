package com.babyai.mod.mixin;

import com.babyai.mod.EventBridge;
import net.minecraft.item.BlockItem;
import net.minecraft.item.ItemPlacementContext;
import net.minecraft.registry.Registries;
import net.minecraft.util.ActionResult;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Detects block placement by injecting at the return of
 * {@link BlockItem#place(ItemPlacementContext)}.
 *
 * Only emits an event when the ActionResult indicates the block
 * was actually placed (SUCCESS / CONSUME), and only on the
 * server side.
 *
 * <p><b>Mapping note:</b> If the build fails with "cannot find
 * method 'place'", the yarn method name may have changed for
 * your MC version.  Check the Fabric yarn browser for the
 * correct name.
 */
@Mixin(BlockItem.class)
public class BlockItemMixin {

    @Inject(method = "place", at = @At("RETURN"))
    private void babyai$onBlockPlaced(
            ItemPlacementContext context,
            CallbackInfoReturnable<ActionResult> cir
    ) {
        // Only fire on the server when the placement actually succeeded
        if (context.getWorld() == null || context.getWorld().isClient()) return;
        if (!cir.getReturnValue().isAccepted()) return;

        BlockItem self = (BlockItem) (Object) this;
        String blockId = Registries.BLOCK.getId(self.getBlock()).toString();
        long tick = context.getWorld().getServer() != null
                ? context.getWorld().getServer().getTicks() : 0;

        EventBridge.INSTANCE.onBlockPlaced(
                blockId, context.getBlockPos(), tick
        );
    }
}
