package com.babyai.mod.mixin;

import com.babyai.mod.EventBridge;
import net.minecraft.entity.ItemEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Detects item pickups by hooking {@link ItemEntity#onPlayerCollision}.
 *
 * We snapshot the stack at the start of the method and check
 * whether the entity was removed (= fully picked up) at the end.
 *
 * <p><b>Mapping note:</b> In some MC versions the method is called
 * {@code onPlayerCollision}, in others {@code playerTouch} or
 * {@code onCollideWithPlayer}.  If the build fails, check the
 * Fabric yarn browser for the correct 1.21.x name.
 */
@Mixin(ItemEntity.class)
public abstract class ItemPickupMixin {

    @Shadow
    public abstract ItemStack getStack();

    /** Stack snapshot taken at HEAD — needed because the stack is
     *  emptied by the time TAIL fires after a successful pickup. */
    @Unique
    private ItemStack babyai$capturedStack = ItemStack.EMPTY;

    @Inject(method = "onPlayerCollision", at = @At("HEAD"))
    private void babyai$captureBeforePickup(PlayerEntity player, CallbackInfo ci) {
        babyai$capturedStack = getStack().copy();
    }

    @Inject(method = "onPlayerCollision", at = @At("TAIL"))
    private void babyai$checkAfterPickup(PlayerEntity player, CallbackInfo ci) {
        ItemEntity self = (ItemEntity) (Object) this;

        // The entity is removed after a full pickup.  Partial pickups
        // (when the player's inventory is nearly full) reduce the
        // stack count but keep the entity alive.
        boolean fullyPickedUp = self.isRemoved();
        boolean partialPickup = !fullyPickedUp
                && getStack().getCount() < babyai$capturedStack.getCount();

        if ((fullyPickedUp || partialPickup) && !babyai$capturedStack.isEmpty()) {
            int pickedCount = fullyPickedUp
                    ? babyai$capturedStack.getCount()
                    : babyai$capturedStack.getCount() - getStack().getCount();

            String itemId = Registries.ITEM
                    .getId(babyai$capturedStack.getItem()).toString();
            long tick = self.getWorld().getServer() != null
                    ? self.getWorld().getServer().getTicks() : 0;

            EventBridge.INSTANCE.onItemPickup(itemId, pickedCount, tick);
        }

        babyai$capturedStack = ItemStack.EMPTY;
    }
}
