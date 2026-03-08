package com.babyai.mod.mixin;

import com.babyai.mod.EventBridge;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.screen.slot.CraftingResultSlot;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Detects crafting completions by hooking
 * {@link CraftingResultSlot#onTakeItem}.
 *
 * Fires when the player takes the crafted item out of the result
 * slot — this is the authoritative moment a craft is "done".
 *
 * <p><b>Mapping note:</b> The class is in
 * {@code net.minecraft.screen.slot.CraftingResultSlot} in some
 * yarn versions.  If the build fails with a missing class, search
 * for {@code CraftingResultSlot} in the yarn browser.
 */
@Mixin(CraftingResultSlot.class)
public class CraftingMixin {

    private static final Logger LOGGER = LoggerFactory.getLogger("Baby-AI");

    @Inject(method = "onTakeItem", at = @At("HEAD"))
    private void babyai$onCraftTake(
            PlayerEntity player,
            ItemStack stack,
            CallbackInfo ci
    ) {
        if (player.getWorld() == null || player.getWorld().isClient()) return;
        if (stack.isEmpty()) return;

        String itemId = Registries.ITEM.getId(stack.getItem()).toString();
        int count = stack.getCount();
        long tick = player.getServer() != null
                ? player.getServer().getTicks() : 0;

        LOGGER.info("[Baby-AI] Item crafted: {} x{}", itemId, count);
        EventBridge.INSTANCE.onItemCrafted(itemId, count, tick);
    }
}
