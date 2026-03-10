"""
Item reward data tables -- all Minecraft item-to-reward mappings.

This module is auto-populated at import time via _register() calls.
It is kept separate from the public API (item_rewards.py) to keep
the data (~1400 lines of registrations) out of the logic module.
"""

from __future__ import annotations

from typing import Dict

# ═══════════════════════════════════════════════════════════════════
#  MASTER REWARD TABLE  —  minecraft:id → float reward
# ═══════════════════════════════════════════════════════════════════

ITEM_REWARDS: Dict[str, float] = {}


def _register(reward: float, *ids: str) -> None:
    """Batch-register items at the same reward value."""
    for item_id in ids:
        ITEM_REWARDS[item_id] = reward


# ───────────────────────────────────────────────────────────────────
# TIER 0 — Trivial (0.1)
# Items found everywhere with zero effort.
# ───────────────────────────────────────────────────────────────────
_register(
    0.1,
    # Terrain blocks
    "minecraft:dirt",
    "minecraft:coarse_dirt",
    "minecraft:grass_block",
    "minecraft:sand",
    "minecraft:red_sand",
    "minecraft:gravel",
    "minecraft:cobblestone",
    "minecraft:cobbled_deepslate",
    "minecraft:netherrack",
    "minecraft:mud",
    "minecraft:rooted_dirt",
    "minecraft:podzol",
    # Common drops
    "minecraft:rotten_flesh",
    # Trivial craft
    "minecraft:stick",
    # Very common plants
    "minecraft:wheat_seeds",
    "minecraft:short_grass",
    "minecraft:tall_grass",
    "minecraft:fern",
    "minecraft:large_fern",
    "minecraft:dead_bush",
    "minecraft:seagrass",
    "minecraft:kelp",
    # Snow / ice
    "minecraft:snow_block",
    "minecraft:snow",
    "minecraft:ice",
    "minecraft:packed_ice",
    # Air / barriers (shouldn't fire, but safety)
    "minecraft:air",
    "minecraft:barrier",
)


# ───────────────────────────────────────────────────────────────────
# TIER 1 — Common (0.2–0.4)
# Easy gathering, surface resources, common mob drops.
# ───────────────────────────────────────────────────────────────────

# 0.2 — Surface blocks / very easy
_register(
    0.2,
    "minecraft:stone",
    "minecraft:granite",
    "minecraft:polished_granite",
    "minecraft:diorite",
    "minecraft:polished_diorite",
    "minecraft:andesite",
    "minecraft:polished_andesite",
    "minecraft:deepslate",
    "minecraft:polished_deepslate",
    "minecraft:sandstone",
    "minecraft:red_sandstone",
    "minecraft:clay",
    "minecraft:clay_ball",
    "minecraft:smooth_stone",
    "minecraft:mossy_cobblestone",
    "minecraft:tuff",
    "minecraft:calcite",
    "minecraft:dripstone_block",
    "minecraft:pointed_dripstone",
    "minecraft:moss_block",
    "minecraft:moss_carpet",
    "minecraft:flint",
    "minecraft:snowball",
    "minecraft:egg",
    "minecraft:cactus",
    "minecraft:sugar_cane",
    "minecraft:bamboo",
    "minecraft:vine",
    "minecraft:lily_pad",
    "minecraft:glass",
    "minecraft:glass_pane",
    "minecraft:torch",
    "minecraft:ladder",
    "minecraft:scaffolding",
)

# 0.3 — Logs, leaves, saplings, basic wood products, common flowers
_register(
    0.3,
    # Logs (all wood types)
    "minecraft:oak_log",
    "minecraft:spruce_log",
    "minecraft:birch_log",
    "minecraft:jungle_log",
    "minecraft:acacia_log",
    "minecraft:dark_oak_log",
    "minecraft:cherry_log",
    "minecraft:mangrove_log",
    "minecraft:bamboo_block",
    "minecraft:stripped_oak_log",
    "minecraft:stripped_spruce_log",
    "minecraft:stripped_birch_log",
    "minecraft:stripped_jungle_log",
    "minecraft:stripped_acacia_log",
    "minecraft:stripped_dark_oak_log",
    "minecraft:stripped_cherry_log",
    "minecraft:stripped_mangrove_log",
    "minecraft:stripped_bamboo_block",
    "minecraft:crimson_stem",
    "minecraft:warped_stem",
    "minecraft:stripped_crimson_stem",
    "minecraft:stripped_warped_stem",
    # Leaves
    "minecraft:oak_leaves",
    "minecraft:spruce_leaves",
    "minecraft:birch_leaves",
    "minecraft:jungle_leaves",
    "minecraft:acacia_leaves",
    "minecraft:dark_oak_leaves",
    "minecraft:cherry_leaves",
    "minecraft:mangrove_leaves",
    "minecraft:azalea_leaves",
    "minecraft:flowering_azalea_leaves",
    # Saplings
    "minecraft:oak_sapling",
    "minecraft:spruce_sapling",
    "minecraft:birch_sapling",
    "minecraft:jungle_sapling",
    "minecraft:acacia_sapling",
    "minecraft:dark_oak_sapling",
    "minecraft:cherry_sapling",
    "minecraft:mangrove_propagule",
    # Common flowers
    "minecraft:dandelion",
    "minecraft:poppy",
    "minecraft:blue_orchid",
    "minecraft:allium",
    "minecraft:azure_bluet",
    "minecraft:red_tulip",
    "minecraft:orange_tulip",
    "minecraft:white_tulip",
    "minecraft:pink_tulip",
    "minecraft:oxeye_daisy",
    "minecraft:cornflower",
    "minecraft:lily_of_the_valley",
    "minecraft:sunflower",
    "minecraft:lilac",
    "minecraft:rose_bush",
    "minecraft:peony",
    "minecraft:wither_rose",
    "minecraft:torchflower",
    "minecraft:pitcher_plant",
    # Mushrooms
    "minecraft:brown_mushroom",
    "minecraft:red_mushroom",
    "minecraft:brown_mushroom_block",
    "minecraft:red_mushroom_block",
    "minecraft:mushroom_stem",
    # Common mob drops
    "minecraft:bone",
    "minecraft:string",
    "minecraft:feather",
    "minecraft:ink_sac",
    "minecraft:glow_ink_sac",
    "minecraft:spider_eye",
    # Raw food (easy to get)
    "minecraft:chicken",
    "minecraft:beef",
    "minecraft:porkchop",
    "minecraft:mutton",
    "minecraft:cod",
    "minecraft:salmon",
    "minecraft:rabbit",
    "minecraft:tropical_fish",
    "minecraft:pufferfish",
    "minecraft:sweet_berries",
    "minecraft:glow_berries",
    "minecraft:apple",
    "minecraft:melon_slice",
    "minecraft:potato",
    "minecraft:carrot",
    "minecraft:beetroot",
    "minecraft:wheat",
    "minecraft:pumpkin",
    "minecraft:melon",
    "minecraft:sugar",
    "minecraft:cocoa_beans",
)

# 0.4 — Planks, basic crafted wood items, coal, charcoal, basic foods
_register(
    0.4,
    # Planks (all types)
    "minecraft:oak_planks",
    "minecraft:spruce_planks",
    "minecraft:birch_planks",
    "minecraft:jungle_planks",
    "minecraft:acacia_planks",
    "minecraft:dark_oak_planks",
    "minecraft:cherry_planks",
    "minecraft:mangrove_planks",
    "minecraft:bamboo_planks",
    "minecraft:crimson_planks",
    "minecraft:warped_planks",
    # Coal (surface veins, very common ore)
    "minecraft:coal",
    "minecraft:charcoal",
    "minecraft:coal_ore",
    "minecraft:deepslate_coal_ore",
    "minecraft:coal_block",
    # Wood slabs, stairs, fences, doors — basic building
    "minecraft:oak_slab",
    "minecraft:oak_stairs",
    "minecraft:oak_fence",
    "minecraft:oak_fence_gate",
    "minecraft:oak_door",
    "minecraft:oak_trapdoor",
    "minecraft:spruce_slab",
    "minecraft:spruce_stairs",
    "minecraft:spruce_fence",
    "minecraft:spruce_fence_gate",
    "minecraft:spruce_door",
    "minecraft:spruce_trapdoor",
    "minecraft:birch_slab",
    "minecraft:birch_stairs",
    "minecraft:birch_fence",
    "minecraft:birch_fence_gate",
    "minecraft:birch_door",
    "minecraft:birch_trapdoor",
    "minecraft:jungle_slab",
    "minecraft:jungle_stairs",
    "minecraft:jungle_fence",
    "minecraft:jungle_fence_gate",
    "minecraft:jungle_door",
    "minecraft:jungle_trapdoor",
    "minecraft:acacia_slab",
    "minecraft:acacia_stairs",
    "minecraft:acacia_fence",
    "minecraft:acacia_fence_gate",
    "minecraft:acacia_door",
    "minecraft:acacia_trapdoor",
    "minecraft:dark_oak_slab",
    "minecraft:dark_oak_stairs",
    "minecraft:dark_oak_fence",
    "minecraft:dark_oak_fence_gate",
    "minecraft:dark_oak_door",
    "minecraft:dark_oak_trapdoor",
    "minecraft:cherry_slab",
    "minecraft:cherry_stairs",
    "minecraft:cherry_fence",
    "minecraft:cherry_fence_gate",
    "minecraft:cherry_door",
    "minecraft:cherry_trapdoor",
    "minecraft:mangrove_slab",
    "minecraft:mangrove_stairs",
    "minecraft:mangrove_fence",
    "minecraft:mangrove_fence_gate",
    "minecraft:mangrove_door",
    "minecraft:mangrove_trapdoor",
    "minecraft:bamboo_slab",
    "minecraft:bamboo_stairs",
    "minecraft:bamboo_fence",
    "minecraft:bamboo_fence_gate",
    "minecraft:bamboo_door",
    "minecraft:bamboo_trapdoor",
    "minecraft:crimson_slab",
    "minecraft:crimson_stairs",
    "minecraft:crimson_fence",
    "minecraft:crimson_fence_gate",
    "minecraft:crimson_door",
    "minecraft:crimson_trapdoor",
    "minecraft:warped_slab",
    "minecraft:warped_stairs",
    "minecraft:warped_fence",
    "minecraft:warped_fence_gate",
    "minecraft:warped_door",
    "minecraft:warped_trapdoor",
    # Signs
    "minecraft:oak_sign",
    "minecraft:spruce_sign",
    "minecraft:birch_sign",
    "minecraft:jungle_sign",
    "minecraft:acacia_sign",
    "minecraft:dark_oak_sign",
    "minecraft:cherry_sign",
    "minecraft:mangrove_sign",
    "minecraft:bamboo_sign",
    "minecraft:crimson_sign",
    "minecraft:warped_sign",
    "minecraft:oak_hanging_sign",
    "minecraft:spruce_hanging_sign",
    "minecraft:birch_hanging_sign",
    "minecraft:jungle_hanging_sign",
    "minecraft:acacia_hanging_sign",
    "minecraft:dark_oak_hanging_sign",
    "minecraft:cherry_hanging_sign",
    "minecraft:mangrove_hanging_sign",
    "minecraft:bamboo_hanging_sign",
    "minecraft:crimson_hanging_sign",
    "minecraft:warped_hanging_sign",
    # Buttons and pressure plates
    "minecraft:oak_button",
    "minecraft:spruce_button",
    "minecraft:birch_button",
    "minecraft:jungle_button",
    "minecraft:acacia_button",
    "minecraft:dark_oak_button",
    "minecraft:cherry_button",
    "minecraft:mangrove_button",
    "minecraft:bamboo_button",
    "minecraft:crimson_button",
    "minecraft:warped_button",
    "minecraft:oak_pressure_plate",
    "minecraft:spruce_pressure_plate",
    "minecraft:birch_pressure_plate",
    "minecraft:jungle_pressure_plate",
    "minecraft:acacia_pressure_plate",
    "minecraft:dark_oak_pressure_plate",
    "minecraft:cherry_pressure_plate",
    "minecraft:mangrove_pressure_plate",
    "minecraft:bamboo_pressure_plate",
    "minecraft:crimson_pressure_plate",
    "minecraft:warped_pressure_plate",
    "minecraft:stone_pressure_plate",
    "minecraft:stone_button",
    "minecraft:polished_blackstone_button",
    "minecraft:polished_blackstone_pressure_plate",
    # Boats
    "minecraft:oak_boat",
    "minecraft:spruce_boat",
    "minecraft:birch_boat",
    "minecraft:jungle_boat",
    "minecraft:acacia_boat",
    "minecraft:dark_oak_boat",
    "minecraft:cherry_boat",
    "minecraft:mangrove_boat",
    "minecraft:bamboo_raft",
    "minecraft:oak_chest_boat",
    "minecraft:spruce_chest_boat",
    "minecraft:birch_chest_boat",
    "minecraft:jungle_chest_boat",
    "minecraft:acacia_chest_boat",
    "minecraft:dark_oak_chest_boat",
    "minecraft:cherry_chest_boat",
    "minecraft:mangrove_chest_boat",
    "minecraft:bamboo_chest_raft",
    # Cooked food (basic)
    "minecraft:cooked_chicken",
    "minecraft:cooked_beef",
    "minecraft:cooked_porkchop",
    "minecraft:cooked_mutton",
    "minecraft:cooked_cod",
    "minecraft:cooked_salmon",
    "minecraft:cooked_rabbit",
    "minecraft:bread",
    "minecraft:baked_potato",
    "minecraft:dried_kelp",
    "minecraft:dried_kelp_block",
    # Basic crafted food
    "minecraft:cookie",
    "minecraft:mushroom_stew",
    "minecraft:beetroot_soup",
    "minecraft:rabbit_stew",
    # Leather
    "minecraft:leather",
    # Bone meal (from bones)
    "minecraft:bone_meal",
    "minecraft:bone_block",
    # Seeds and farming
    "minecraft:beetroot_seeds",
    "minecraft:pumpkin_seeds",
    "minecraft:melon_seeds",
    "minecraft:torchflower_seeds",
    "minecraft:pitcher_pod",
    # Misc common
    "minecraft:bowl",
    "minecraft:paper",
    "minecraft:gunpowder",
)


# ───────────────────────────────────────────────────────────────────
# TIER 2 — Standard (0.5–1.0)
# Require basic mining, smelting, or slightly more effort.
# ───────────────────────────────────────────────────────────────────

# 0.5 — Stone building blocks, bricks, terracotta
_register(
    0.5,
    "minecraft:stone_bricks",
    "minecraft:mossy_stone_bricks",
    "minecraft:cracked_stone_bricks",
    "minecraft:chiseled_stone_bricks",
    "minecraft:stone_brick_slab",
    "minecraft:stone_brick_stairs",
    "minecraft:stone_brick_wall",
    "minecraft:stone_slab",
    "minecraft:stone_stairs",
    "minecraft:cobblestone_slab",
    "minecraft:cobblestone_stairs",
    "minecraft:cobblestone_wall",
    "minecraft:bricks",
    "minecraft:brick",
    "minecraft:brick_slab",
    "minecraft:brick_stairs",
    "minecraft:brick_wall",
    "minecraft:nether_bricks",
    "minecraft:nether_brick",
    "minecraft:nether_brick_slab",
    "minecraft:nether_brick_stairs",
    "minecraft:nether_brick_wall",
    "minecraft:nether_brick_fence",
    "minecraft:red_nether_bricks",
    "minecraft:red_nether_brick_slab",
    "minecraft:red_nether_brick_stairs",
    "minecraft:red_nether_brick_wall",
    "minecraft:terracotta",
    "minecraft:smooth_sandstone",
    "minecraft:smooth_red_sandstone",
    "minecraft:cut_sandstone",
    "minecraft:cut_red_sandstone",
    "minecraft:sandstone_slab",
    "minecraft:sandstone_stairs",
    "minecraft:sandstone_wall",
    "minecraft:red_sandstone_slab",
    "minecraft:red_sandstone_stairs",
    "minecraft:red_sandstone_wall",
    "minecraft:smooth_sandstone_slab",
    "minecraft:smooth_sandstone_stairs",
    "minecraft:smooth_red_sandstone_slab",
    "minecraft:smooth_red_sandstone_stairs",
    "minecraft:cut_sandstone_slab",
    "minecraft:cut_red_sandstone_slab",
    "minecraft:deepslate_brick_slab",
    "minecraft:deepslate_brick_stairs",
    "minecraft:deepslate_brick_wall",
    "minecraft:deepslate_bricks",
    "minecraft:deepslate_tile_slab",
    "minecraft:deepslate_tile_stairs",
    "minecraft:deepslate_tile_wall",
    "minecraft:deepslate_tiles",
    "minecraft:chiseled_deepslate",
    "minecraft:polished_deepslate_slab",
    "minecraft:polished_deepslate_stairs",
    "minecraft:polished_deepslate_wall",
    "minecraft:cobbled_deepslate_slab",
    "minecraft:cobbled_deepslate_stairs",
    "minecraft:cobbled_deepslate_wall",
    "minecraft:tuff_slab",
    "minecraft:tuff_stairs",
    "minecraft:tuff_wall",
    "minecraft:polished_tuff",
    "minecraft:polished_tuff_slab",
    "minecraft:polished_tuff_stairs",
    "minecraft:polished_tuff_wall",
    "minecraft:tuff_bricks",
    "minecraft:tuff_brick_slab",
    "minecraft:tuff_brick_stairs",
    "minecraft:tuff_brick_wall",
    "minecraft:chiseled_tuff",
    "minecraft:chiseled_tuff_bricks",
    "minecraft:mud_bricks",
    "minecraft:mud_brick_slab",
    "minecraft:mud_brick_stairs",
    "minecraft:mud_brick_wall",
    "minecraft:packed_mud",
    # Flower pots, composters, misc crafted
    "minecraft:flower_pot",
    "minecraft:composter",
    "minecraft:lever",
    "minecraft:tripwire_hook",
    # Rails (iron + stick, common crafting)
    "minecraft:rail",
    # Minecart
    "minecraft:minecart",
    # Carpet / candles / simple decoration
    "minecraft:candle",
)

# 0.6 — Wooden tools — easy to craft but first progression milestone.
#        Slightly above raw materials to reward crafting tools early.
_register(
    0.6,
    "minecraft:wooden_pickaxe",
    "minecraft:wooden_axe",
    "minecraft:wooden_shovel",
    "minecraft:wooden_hoe",
    "minecraft:wooden_sword",
)

# 0.6 — Copper, iron ore (raw), wool, beds, basic dyes
_register(
    0.6,
    # Copper (common ore but requires smelting for use)
    "minecraft:copper_ore",
    "minecraft:deepslate_copper_ore",
    "minecraft:raw_copper",
    "minecraft:raw_copper_block",
    # Iron ore (raw — must smelt to be useful)
    "minecraft:iron_ore",
    "minecraft:deepslate_iron_ore",
    "minecraft:raw_iron",
    "minecraft:raw_iron_block",
    # Wool (all 16 colours)
    "minecraft:white_wool",
    "minecraft:orange_wool",
    "minecraft:magenta_wool",
    "minecraft:light_blue_wool",
    "minecraft:yellow_wool",
    "minecraft:lime_wool",
    "minecraft:pink_wool",
    "minecraft:gray_wool",
    "minecraft:light_gray_wool",
    "minecraft:cyan_wool",
    "minecraft:purple_wool",
    "minecraft:blue_wool",
    "minecraft:brown_wool",
    "minecraft:green_wool",
    "minecraft:red_wool",
    "minecraft:black_wool",
    # Dyes (all 16 colours)
    "minecraft:white_dye",
    "minecraft:orange_dye",
    "minecraft:magenta_dye",
    "minecraft:light_blue_dye",
    "minecraft:yellow_dye",
    "minecraft:lime_dye",
    "minecraft:pink_dye",
    "minecraft:gray_dye",
    "minecraft:light_gray_dye",
    "minecraft:cyan_dye",
    "minecraft:purple_dye",
    "minecraft:blue_dye",
    "minecraft:brown_dye",
    "minecraft:green_dye",
    "minecraft:red_dye",
    "minecraft:black_dye",
    # Slimeball (swamp biome, somewhat common)
    "minecraft:slime_ball",
    "minecraft:slime_block",
    # Honey
    "minecraft:honeycomb",
    "minecraft:honeycomb_block",
    "minecraft:honey_bottle",
    "minecraft:honey_block",
    # Misc easy drops
    "minecraft:rabbit_hide",
    "minecraft:rabbit_foot",
    "minecraft:phantom_membrane",
)

# 0.8 — Copper ingot, furnaces, chests, beds, carpets, stained terracotta,
#        stained glass, concrete powder, stone tools
_register(
    0.8,
    # Copper ingot (smelted)
    "minecraft:copper_ingot",
    "minecraft:copper_block",
    "minecraft:exposed_copper",
    "minecraft:weathered_copper",
    "minecraft:oxidized_copper",
    "minecraft:cut_copper",
    "minecraft:cut_copper_slab",
    "minecraft:cut_copper_stairs",
    "minecraft:waxed_copper_block",
    "minecraft:waxed_exposed_copper",
    "minecraft:waxed_weathered_copper",
    "minecraft:waxed_oxidized_copper",
    "minecraft:waxed_cut_copper",
    "minecraft:waxed_cut_copper_slab",
    "minecraft:waxed_cut_copper_stairs",
    "minecraft:copper_bulb",
    "minecraft:copper_door",
    "minecraft:copper_trapdoor",
    "minecraft:copper_grate",
    "minecraft:chiseled_copper",
    # Functional blocks — basic station crafting
    "minecraft:crafting_table",
    "minecraft:furnace",
    "minecraft:chest",
    "minecraft:barrel",
)

# 0.95 — Stone tools — first real mining tier.
#         Premium over 0.8 raw materials to reward upgrading tools.
_register(
    0.95,
    "minecraft:stone_pickaxe",
    "minecraft:stone_axe",
    "minecraft:stone_shovel",
    "minecraft:stone_hoe",
    "minecraft:stone_sword",
)

_register(
    0.8,
    # Beds (all 16 colours)
    "minecraft:white_bed",
    "minecraft:orange_bed",
    "minecraft:magenta_bed",
    "minecraft:light_blue_bed",
    "minecraft:yellow_bed",
    "minecraft:lime_bed",
    "minecraft:pink_bed",
    "minecraft:gray_bed",
    "minecraft:light_gray_bed",
    "minecraft:cyan_bed",
    "minecraft:purple_bed",
    "minecraft:blue_bed",
    "minecraft:brown_bed",
    "minecraft:green_bed",
    "minecraft:red_bed",
    "minecraft:black_bed",
    # Carpets (all 16 colours)
    "minecraft:white_carpet",
    "minecraft:orange_carpet",
    "minecraft:magenta_carpet",
    "minecraft:light_blue_carpet",
    "minecraft:yellow_carpet",
    "minecraft:lime_carpet",
    "minecraft:pink_carpet",
    "minecraft:gray_carpet",
    "minecraft:light_gray_carpet",
    "minecraft:cyan_carpet",
    "minecraft:purple_carpet",
    "minecraft:blue_carpet",
    "minecraft:brown_carpet",
    "minecraft:green_carpet",
    "minecraft:red_carpet",
    "minecraft:black_carpet",
    # Stained terracotta (all 16 colours — requires dye + terracotta)
    "minecraft:white_terracotta",
    "minecraft:orange_terracotta",
    "minecraft:magenta_terracotta",
    "minecraft:light_blue_terracotta",
    "minecraft:yellow_terracotta",
    "minecraft:lime_terracotta",
    "minecraft:pink_terracotta",
    "minecraft:gray_terracotta",
    "minecraft:light_gray_terracotta",
    "minecraft:cyan_terracotta",
    "minecraft:purple_terracotta",
    "minecraft:blue_terracotta",
    "minecraft:brown_terracotta",
    "minecraft:green_terracotta",
    "minecraft:red_terracotta",
    "minecraft:black_terracotta",
    # Glazed terracotta (smelted stained terracotta)
    "minecraft:white_glazed_terracotta",
    "minecraft:orange_glazed_terracotta",
    "minecraft:magenta_glazed_terracotta",
    "minecraft:light_blue_glazed_terracotta",
    "minecraft:yellow_glazed_terracotta",
    "minecraft:lime_glazed_terracotta",
    "minecraft:pink_glazed_terracotta",
    "minecraft:gray_glazed_terracotta",
    "minecraft:light_gray_glazed_terracotta",
    "minecraft:cyan_glazed_terracotta",
    "minecraft:purple_glazed_terracotta",
    "minecraft:blue_glazed_terracotta",
    "minecraft:brown_glazed_terracotta",
    "minecraft:green_glazed_terracotta",
    "minecraft:red_glazed_terracotta",
    "minecraft:black_glazed_terracotta",
    # Stained glass (all 16 colours)
    "minecraft:white_stained_glass",
    "minecraft:orange_stained_glass",
    "minecraft:magenta_stained_glass",
    "minecraft:light_blue_stained_glass",
    "minecraft:yellow_stained_glass",
    "minecraft:lime_stained_glass",
    "minecraft:pink_stained_glass",
    "minecraft:gray_stained_glass",
    "minecraft:light_gray_stained_glass",
    "minecraft:cyan_stained_glass",
    "minecraft:purple_stained_glass",
    "minecraft:blue_stained_glass",
    "minecraft:brown_stained_glass",
    "minecraft:green_stained_glass",
    "minecraft:red_stained_glass",
    "minecraft:black_stained_glass",
    # Stained glass panes (all 16 colours)
    "minecraft:white_stained_glass_pane",
    "minecraft:orange_stained_glass_pane",
    "minecraft:magenta_stained_glass_pane",
    "minecraft:light_blue_stained_glass_pane",
    "minecraft:yellow_stained_glass_pane",
    "minecraft:lime_stained_glass_pane",
    "minecraft:pink_stained_glass_pane",
    "minecraft:gray_stained_glass_pane",
    "minecraft:light_gray_stained_glass_pane",
    "minecraft:cyan_stained_glass_pane",
    "minecraft:purple_stained_glass_pane",
    "minecraft:blue_stained_glass_pane",
    "minecraft:brown_stained_glass_pane",
    "minecraft:green_stained_glass_pane",
    "minecraft:red_stained_glass_pane",
    "minecraft:black_stained_glass_pane",
    # Concrete powder (all 16 — sand + gravel + dye)
    "minecraft:white_concrete_powder",
    "minecraft:orange_concrete_powder",
    "minecraft:magenta_concrete_powder",
    "minecraft:light_blue_concrete_powder",
    "minecraft:yellow_concrete_powder",
    "minecraft:lime_concrete_powder",
    "minecraft:pink_concrete_powder",
    "minecraft:gray_concrete_powder",
    "minecraft:light_gray_concrete_powder",
    "minecraft:cyan_concrete_powder",
    "minecraft:purple_concrete_powder",
    "minecraft:blue_concrete_powder",
    "minecraft:brown_concrete_powder",
    "minecraft:green_concrete_powder",
    "minecraft:red_concrete_powder",
    "minecraft:black_concrete_powder",
    # Concrete (powder + water — requires placement)
    "minecraft:white_concrete",
    "minecraft:orange_concrete",
    "minecraft:magenta_concrete",
    "minecraft:light_blue_concrete",
    "minecraft:yellow_concrete",
    "minecraft:lime_concrete",
    "minecraft:pink_concrete",
    "minecraft:gray_concrete",
    "minecraft:light_gray_concrete",
    "minecraft:cyan_concrete",
    "minecraft:purple_concrete",
    "minecraft:blue_concrete",
    "minecraft:brown_concrete",
    "minecraft:green_concrete",
    "minecraft:red_concrete",
    "minecraft:black_concrete",
    # Candles (all 16 + uncoloured)
    "minecraft:white_candle",
    "minecraft:orange_candle",
    "minecraft:magenta_candle",
    "minecraft:light_blue_candle",
    "minecraft:yellow_candle",
    "minecraft:lime_candle",
    "minecraft:pink_candle",
    "minecraft:gray_candle",
    "minecraft:light_gray_candle",
    "minecraft:cyan_candle",
    "minecraft:purple_candle",
    "minecraft:blue_candle",
    "minecraft:brown_candle",
    "minecraft:green_candle",
    "minecraft:red_candle",
    "minecraft:black_candle",
    # Misc
    "minecraft:book",
    "minecraft:glass_bottle",
    "minecraft:map",
    "minecraft:compass",
    "minecraft:clock",
    "minecraft:bucket",
    "minecraft:water_bucket",
    "minecraft:fishing_rod",
    "minecraft:lead",
    "minecraft:painting",
    "minecraft:item_frame",
    "minecraft:armor_stand",
    "minecraft:lantern",
    "minecraft:soul_torch",
    "minecraft:soul_lantern",
    "minecraft:chain",
    "minecraft:lightning_rod",
    "minecraft:flower_pot",
    "minecraft:cauldron",
    "minecraft:campfire",
    "minecraft:soul_campfire",
    # Food craft
    "minecraft:cake",
    "minecraft:pumpkin_pie",
    "minecraft:suspicious_stew",
    # Beehive (crafted from planks + honeycomb)
    "minecraft:beehive",
    "minecraft:bee_nest",
)

# 1.0 — Iron ingot, iron tools, iron blocks, shields, shears, basic redstone
_register(
    1.0,
    # Iron (smelted — key progression milestone)
    "minecraft:iron_ingot",
    "minecraft:iron_block",
    "minecraft:iron_nugget",
)

# 1.2 — Iron tools — premium over raw iron to reward tool crafting.
_register(
    1.2,
    "minecraft:iron_pickaxe",
    "minecraft:iron_axe",
    "minecraft:iron_shovel",
    "minecraft:iron_hoe",
    "minecraft:iron_sword",
)

_register(
    1.0,
    # Shield (iron + planks)
    "minecraft:shield",
    # Shears (2 iron)
    "minecraft:shears",
    # Flint and steel
    "minecraft:flint_and_steel",
    # Iron door and trapdoor
    "minecraft:iron_door",
    "minecraft:iron_trapdoor",
    # Basic bucket variants
    "minecraft:lava_bucket",
    "minecraft:milk_bucket",
    "minecraft:powder_snow_bucket",
    # Basic redstone
    "minecraft:redstone_torch",
    "minecraft:repeater",
    "minecraft:comparator",
    "minecraft:lever",
    "minecraft:piston",
    "minecraft:sticky_piston",
    "minecraft:dispenser",
    "minecraft:dropper",
    "minecraft:observer",
    "minecraft:hopper",
    "minecraft:tripwire_hook",
    "minecraft:trapped_chest",
    "minecraft:note_block",
    "minecraft:jukebox",
    "minecraft:target",
    # Iron / powered rails
    "minecraft:powered_rail",
    "minecraft:detector_rail",
    "minecraft:activator_rail",
    "minecraft:chest_minecart",
    "minecraft:furnace_minecart",
    "minecraft:hopper_minecart",
    "minecraft:tnt_minecart",
    # TNT (gunpowder + sand)
    "minecraft:tnt",
    # Bow (string + sticks)
    "minecraft:bow",
    "minecraft:arrow",
    "minecraft:spectral_arrow",
    # Fire charge
    "minecraft:fire_charge",
    # Banners (all 16 colours — 6 wool + stick)
    "minecraft:white_banner",
    "minecraft:orange_banner",
    "minecraft:magenta_banner",
    "minecraft:light_blue_banner",
    "minecraft:yellow_banner",
    "minecraft:lime_banner",
    "minecraft:pink_banner",
    "minecraft:gray_banner",
    "minecraft:light_gray_banner",
    "minecraft:cyan_banner",
    "minecraft:purple_banner",
    "minecraft:blue_banner",
    "minecraft:brown_banner",
    "minecraft:green_banner",
    "minecraft:red_banner",
    "minecraft:black_banner",
    # Iron armor
    "minecraft:iron_helmet",
    "minecraft:iron_chestplate",
    "minecraft:iron_leggings",
    "minecraft:iron_boots",
    # Leather armor (cheap but crafted)
    "minecraft:leather_helmet",
    "minecraft:leather_chestplate",
    "minecraft:leather_leggings",
    "minecraft:leather_boots",
    # Chainmail armor (only from mob drops / trading — somewhat rare)
    "minecraft:chainmail_helmet",
    "minecraft:chainmail_chestplate",
    "minecraft:chainmail_leggings",
    "minecraft:chainmail_boots",
    # Leather horse armor (crafted)
    "minecraft:leather_horse_armor",
    # Spyglass (copper + amethyst)
    "minecraft:spyglass",
    # Glow item frame
    "minecraft:glow_item_frame",
    # Book and quill
    "minecraft:writable_book",
    # Empty map
    "minecraft:filled_map",
    # Firework rocket (basic)
    "minecraft:firework_rocket",
    "minecraft:firework_star",
    # Crossbow (iron + string + sticks + tripwire hook)
    "minecraft:crossbow",
    # Carrot on a stick
    "minecraft:carrot_on_a_stick",
)


# ───────────────────────────────────────────────────────────────────
# TIER 3 — Intermediate (1.5–2.5)
# Deeper mining, progression, combat, or significant crafting.
# ───────────────────────────────────────────────────────────────────

# 1.5 — Gold ore/ingot, redstone, lapis, functional stations
_register(
    1.5,
    # Gold (deeper mining, less common)
    "minecraft:gold_ore",
    "minecraft:deepslate_gold_ore",
    "minecraft:nether_gold_ore",
    "minecraft:raw_gold",
    "minecraft:raw_gold_block",
    "minecraft:gold_ingot",
    "minecraft:gold_nugget",
    "minecraft:gold_block",
)

# 1.8 — Golden tools — premium over raw gold to reward tool crafting.
_register(
    1.8,
    "minecraft:golden_pickaxe",
    "minecraft:golden_axe",
    "minecraft:golden_shovel",
    "minecraft:golden_hoe",
    "minecraft:golden_sword",
)

_register(
    1.5,
    # Golden armor
    "minecraft:golden_helmet",
    "minecraft:golden_chestplate",
    "minecraft:golden_leggings",
    "minecraft:golden_boots",
    # Iron horse armor (dungeon / structure loot)
    "minecraft:iron_horse_armor",
    # Redstone ore
    "minecraft:redstone_ore",
    "minecraft:deepslate_redstone_ore",
    "minecraft:redstone",
    "minecraft:redstone_block",
    # Lapis lazuli (requires iron pick, somewhat deep)
    "minecraft:lapis_ore",
    "minecraft:deepslate_lapis_ore",
    "minecraft:lapis_lazuli",
    "minecraft:lapis_block",
    # Functional crafting stations (require ingredients)
    "minecraft:smoker",
    "minecraft:blast_furnace",
    "minecraft:loom",
    "minecraft:cartography_table",
    "minecraft:fletching_table",
    "minecraft:smithing_table",
    "minecraft:stonecutter",
    "minecraft:grindstone",
    "minecraft:lectern",
    "minecraft:bell",
    # Anvil (significant iron cost: 31 ingots)
    "minecraft:anvil",
    "minecraft:chipped_anvil",
    "minecraft:damaged_anvil",
    # Golden apple (gold + apple — notable crafting)
    "minecraft:golden_apple",
    "minecraft:golden_carrot",
    "minecraft:glistering_melon_slice",
    # Amethyst (requires finding geode)
    "minecraft:amethyst_shard",
    "minecraft:amethyst_block",
    "minecraft:amethyst_cluster",
    "minecraft:budding_amethyst",
    "minecraft:small_amethyst_bud",
    "minecraft:medium_amethyst_bud",
    "minecraft:large_amethyst_bud",
    # Obsidian (diamond pick required to mine, or lava + water)
    "minecraft:obsidian",
    # Nether wart (found in Nether fortresses)
    "minecraft:nether_wart",
    "minecraft:nether_wart_block",
    # Soul sand / soil
    "minecraft:soul_sand",
    "minecraft:soul_soil",
    # Basalt / blackstone
    "minecraft:basalt",
    "minecraft:smooth_basalt",
    "minecraft:polished_basalt",
    "minecraft:blackstone",
    "minecraft:polished_blackstone",
    "minecraft:polished_blackstone_bricks",
    "minecraft:polished_blackstone_brick_slab",
    "minecraft:polished_blackstone_brick_stairs",
    "minecraft:polished_blackstone_brick_wall",
    "minecraft:polished_blackstone_slab",
    "minecraft:polished_blackstone_stairs",
    "minecraft:polished_blackstone_wall",
    "minecraft:chiseled_polished_blackstone",
    "minecraft:gilded_blackstone",
    "minecraft:blackstone_slab",
    "minecraft:blackstone_stairs",
    "minecraft:blackstone_wall",
    # Prismarine (ocean monument — requires some effort)
    "minecraft:prismarine",
    "minecraft:prismarine_bricks",
    "minecraft:dark_prismarine",
    "minecraft:prismarine_slab",
    "minecraft:prismarine_brick_slab",
    "minecraft:dark_prismarine_slab",
    "minecraft:prismarine_stairs",
    "minecraft:prismarine_brick_stairs",
    "minecraft:dark_prismarine_stairs",
    "minecraft:prismarine_wall",
    "minecraft:prismarine_shard",
    "minecraft:prismarine_crystals",
    "minecraft:sea_lantern",
    # Quartz (Nether mining)
    "minecraft:nether_quartz_ore",
    "minecraft:quartz",
    "minecraft:quartz_block",
    "minecraft:quartz_bricks",
    "minecraft:quartz_pillar",
    "minecraft:quartz_slab",
    "minecraft:quartz_stairs",
    "minecraft:smooth_quartz",
    "minecraft:smooth_quartz_slab",
    "minecraft:smooth_quartz_stairs",
    "minecraft:chiseled_quartz_block",
    # Glowstone (Nether)
    "minecraft:glowstone_dust",
    "minecraft:glowstone",
    # Magma (Nether)
    "minecraft:magma_block",
    "minecraft:magma_cream",
    # Potions (need brewing stand + Nether ingredients)
    "minecraft:potion",
    "minecraft:splash_potion",
    "minecraft:lingering_potion",
    # Tipped arrows (potion-tipped)
    "minecraft:tipped_arrow",
    # Glow lichen
    "minecraft:glow_lichen",
    # Sculk (deep underground)
    "minecraft:sculk",
    "minecraft:sculk_vein",
    # Written / enchanted books (village loot / librarian trade)
    "minecraft:written_book",
    "minecraft:name_tag",
    "minecraft:saddle",
    # Bundle
    "minecraft:bundle",
    # Golden horse armor (structure loot)
    "minecraft:golden_horse_armor",
    # Warped fungus on a stick (Nether + carrot on a stick variant)
    "minecraft:warped_fungus_on_a_stick",
    # Nether vegetation
    "minecraft:crimson_fungus",
    "minecraft:warped_fungus",
    "minecraft:crimson_roots",
    "minecraft:warped_roots",
    "minecraft:nether_sprouts",
    "minecraft:shroomlight",
    "minecraft:weeping_vines",
    "minecraft:twisting_vines",
    "minecraft:crimson_nylium",
    "minecraft:warped_nylium",
)

# 2.0 — Emerald, brewing stand, ender pearl, blaze rod ingredients
_register(
    2.0,
    # Emerald (only in mountain biomes, rare; or from villager trading)
    "minecraft:emerald_ore",
    "minecraft:deepslate_emerald_ore",
    "minecraft:emerald",
    "minecraft:emerald_block",
    # Brewing stand (blaze rod + cobblestone — Nether prerequisite)
    "minecraft:brewing_stand",
    # Blaze rod / powder (Nether fortress mob)
    "minecraft:blaze_rod",
    "minecraft:blaze_powder",
    # Ender pearl (Enderman drop — somewhat rare, combat required)
    "minecraft:ender_pearl",
    # Ghast tear (Nether mob, hard to fight)
    "minecraft:ghast_tear",
    # Fermented spider eye (crafted potion ingredient)
    "minecraft:fermented_spider_eye",
    # Respawn anchor (crying obsidian + glowstone — Nether utility)
    "minecraft:respawn_anchor",
    "minecraft:crying_obsidian",
    # Lodestone (chiseled stone bricks + netherite ingot)
    "minecraft:lodestone",
    # End rod (chorus fruit processing + blaze rod)
    "minecraft:end_rod",
    # Purpur (End city — post-dragon)
    "minecraft:purpur_block",
    "minecraft:purpur_pillar",
    "minecraft:purpur_slab",
    "minecraft:purpur_stairs",
    # End stone (End dimension)
    "minecraft:end_stone",
    "minecraft:end_stone_bricks",
    "minecraft:end_stone_brick_slab",
    "minecraft:end_stone_brick_stairs",
    "minecraft:end_stone_brick_wall",
    # Chorus (End islands, post-dragon)
    "minecraft:chorus_fruit",
    "minecraft:chorus_flower",
    "minecraft:chorus_plant",
    "minecraft:popped_chorus_fruit",
    # Sculk sensor / catalyst / shrieker (deep dark — dangerous)
    "minecraft:sculk_sensor",
    "minecraft:sculk_catalyst",
    "minecraft:sculk_shrieker",
    "minecraft:calibrated_sculk_sensor",
    # Music discs (mob drops / dungeon loot — collectable)
    "minecraft:music_disc_13",
    "minecraft:music_disc_cat",
    "minecraft:music_disc_blocks",
    "minecraft:music_disc_chirp",
    "minecraft:music_disc_far",
    "minecraft:music_disc_mall",
    "minecraft:music_disc_mellohi",
    "minecraft:music_disc_stal",
    "minecraft:music_disc_strad",
    "minecraft:music_disc_ward",
    "minecraft:music_disc_11",
    "minecraft:music_disc_wait",
    "minecraft:music_disc_otherside",
    "minecraft:music_disc_5",
    "minecraft:music_disc_pigstep",
    "minecraft:music_disc_relic",
    "minecraft:music_disc_creator",
    "minecraft:music_disc_creator_music_box",
    "minecraft:music_disc_precipice",
    "minecraft:disc_fragment_5",
    # Wither skeleton skull (rare Nether fortress drop)
    "minecraft:wither_skeleton_skull",
    "minecraft:skeleton_skull",
    "minecraft:zombie_head",
    "minecraft:creeper_head",
    "minecraft:player_head",
    "minecraft:piglin_head",
    # Shulker shell (End city mobs)
    "minecraft:shulker_shell",
    # Dragon's Breath (End, fighting the dragon)
    "minecraft:dragon_breath",
    # Experience bottle (loot / villager trade)
    "minecraft:experience_bottle",
    # Banner patterns (rare recipes)
    "minecraft:creeper_banner_pattern",
    "minecraft:skull_banner_pattern",
    "minecraft:flower_banner_pattern",
    "minecraft:mojang_banner_pattern",
    "minecraft:globe_banner_pattern",
    "minecraft:piglin_banner_pattern",
    # Trident (drowned drop — rare)
    "minecraft:trident",
    # Conduit (Heart of the Sea + nautilus shells)
    "minecraft:conduit",
    "minecraft:heart_of_the_sea",
    "minecraft:nautilus_shell",
    # Sponge (ocean monument — guardian drops)
    "minecraft:sponge",
    "minecraft:wet_sponge",
    # Recovery compass (echo shards — ancient city / deep dark)
    "minecraft:recovery_compass",
    "minecraft:echo_shard",
)

# 2.5 — Eye of Ender, enchanting table, shulker boxes
_register(
    2.5,
    # Eye of Ender (blaze powder + ender pearl — key End progression)
    "minecraft:ender_eye",
    # Enchanting table (diamond + obsidian + book)
    "minecraft:enchanting_table",
    # Enchanted book (valuable find)
    "minecraft:enchanted_book",
    # Shulker boxes (End city — shulker shells + chest)
    "minecraft:shulker_box",
    "minecraft:white_shulker_box",
    "minecraft:orange_shulker_box",
    "minecraft:magenta_shulker_box",
    "minecraft:light_blue_shulker_box",
    "minecraft:yellow_shulker_box",
    "minecraft:lime_shulker_box",
    "minecraft:pink_shulker_box",
    "minecraft:gray_shulker_box",
    "minecraft:light_gray_shulker_box",
    "minecraft:cyan_shulker_box",
    "minecraft:purple_shulker_box",
    "minecraft:blue_shulker_box",
    "minecraft:brown_shulker_box",
    "minecraft:green_shulker_box",
    "minecraft:red_shulker_box",
    "minecraft:black_shulker_box",
    # Ender chest (obsidian + eye of ender)
    "minecraft:ender_chest",
    # Beacon (nether star + glass + obsidian — endgame utility)
    "minecraft:beacon",
)


# ───────────────────────────────────────────────────────────────────
# TIER 4 — Advanced (3.0–5.0)
# Diamond tier, significant progression, dangerous areas.
# ───────────────────────────────────────────────────────────────────

# 3.0 — Diamond ore/gem, diamond tools
_register(
    3.0,
    # Diamond (deep mining, Y < 16, rare vein)
    "minecraft:diamond_ore",
    "minecraft:deepslate_diamond_ore",
    "minecraft:diamond",
    "minecraft:diamond_block",
)

# 3.5 — Diamond tools — premium over raw diamond to reward tool crafting.
_register(
    3.5,
    "minecraft:diamond_pickaxe",
    "minecraft:diamond_axe",
    "minecraft:diamond_shovel",
    "minecraft:diamond_hoe",
    "minecraft:diamond_sword",
)

_register(
    3.0,
    # Diamond horse armor (dungeon loot only)
    "minecraft:diamond_horse_armor",
)

# 4.0 — Diamond armor
_register(
    4.0,
    "minecraft:diamond_helmet",
    "minecraft:diamond_chestplate",
    "minecraft:diamond_leggings",
    "minecraft:diamond_boots",
)

# 5.0 — Ancient debris (Nether, Y 8-22, blast-resistant, very rare)
_register(
    5.0,
    "minecraft:ancient_debris",
)


# ───────────────────────────────────────────────────────────────────
# TIER 5 — Rare (6.0–10.0)
# Endgame materials, boss prerequisites, extremely rare finds.
# ───────────────────────────────────────────────────────────────────

# 6.0 — Netherite scrap (smelt ancient debris — 4 needed per ingot)
_register(
    6.0,
    "minecraft:netherite_scrap",
)

# 8.0 — Netherite ingot (4 scrap + 4 gold), netherite tools, nether star
_register(
    8.0,
    "minecraft:netherite_ingot",
    "minecraft:netherite_block",
)

# 9.0 — Netherite tools — premium over raw netherite to reward endgame tool crafting.
_register(
    9.0,
    "minecraft:netherite_pickaxe",
    "minecraft:netherite_axe",
    "minecraft:netherite_shovel",
    "minecraft:netherite_hoe",
    "minecraft:netherite_sword",
)

_register(
    8.0,
    # Smithing templates (rare structure loot)
    "minecraft:netherite_upgrade_smithing_template",
    "minecraft:sentry_armor_trim_smithing_template",
    "minecraft:dune_armor_trim_smithing_template",
    "minecraft:coast_armor_trim_smithing_template",
    "minecraft:wild_armor_trim_smithing_template",
    "minecraft:ward_armor_trim_smithing_template",
    "minecraft:eye_armor_trim_smithing_template",
    "minecraft:vex_armor_trim_smithing_template",
    "minecraft:tide_armor_trim_smithing_template",
    "minecraft:snout_armor_trim_smithing_template",
    "minecraft:rib_armor_trim_smithing_template",
    "minecraft:spire_armor_trim_smithing_template",
    "minecraft:wayfinder_armor_trim_smithing_template",
    "minecraft:shaper_armor_trim_smithing_template",
    "minecraft:silence_armor_trim_smithing_template",
    "minecraft:raiser_armor_trim_smithing_template",
    "minecraft:host_armor_trim_smithing_template",
    "minecraft:flow_armor_trim_smithing_template",
    "minecraft:bolt_armor_trim_smithing_template",
    # Nether star (Wither boss drop)
    "minecraft:nether_star",
    # Elytra (End ship — post-dragon, End city raiding)
    "minecraft:elytra",
    # Totem of Undying (Evoker drop — raid / mansion)
    "minecraft:totem_of_undying",
    # Turtle shell (scute collection — breed turtles, wait for babies)
    "minecraft:turtle_helmet",
    "minecraft:turtle_scute",
)

# 10.0 — Dragon head (End ship), End crystal
_register(
    10.0,
    # Dragon head (End ship — very dangerous to reach)
    "minecraft:dragon_head",
    # End crystal (ghast tear + eye of ender + glass)
    "minecraft:end_crystal",
)


# ───────────────────────────────────────────────────────────────────
# TIER 6 — Legendary (12.0–20.0)
# One-of-a-kind / extreme endgame.
# ───────────────────────────────────────────────────────────────────

# 12.0 — Netherite armor (netherite ingot + diamond armor + template)
_register(
    12.0,
    "minecraft:netherite_helmet",
    "minecraft:netherite_chestplate",
    "minecraft:netherite_leggings",
    "minecraft:netherite_boots",
)

# 15.0 — Enchanted golden apple (only in loot chests — cannot be crafted)
_register(
    15.0,
    "minecraft:enchanted_golden_apple",
)

# 20.0 — Dragon egg (one per world, Ender Dragon defeat)
_register(
    20.0,
    "minecraft:dragon_egg",
)


# ───────────────────────────────────────────────────────────────────
# SPAWN EGGS — Creative-mode only, minimal reward (shouldn't appear
# in survival, but handle gracefully).
# ───────────────────────────────────────────────────────────────────
_SPAWN_EGGS = [
    "minecraft:bat_spawn_egg",
    "minecraft:bee_spawn_egg",
    "minecraft:blaze_spawn_egg",
    "minecraft:cat_spawn_egg",
    "minecraft:cave_spider_spawn_egg",
    "minecraft:chicken_spawn_egg",
    "minecraft:cod_spawn_egg",
    "minecraft:cow_spawn_egg",
    "minecraft:creeper_spawn_egg",
    "minecraft:dolphin_spawn_egg",
    "minecraft:donkey_spawn_egg",
    "minecraft:drowned_spawn_egg",
    "minecraft:elder_guardian_spawn_egg",
    "minecraft:enderman_spawn_egg",
    "minecraft:endermite_spawn_egg",
    "minecraft:evoker_spawn_egg",
    "minecraft:fox_spawn_egg",
    "minecraft:ghast_spawn_egg",
    "minecraft:guardian_spawn_egg",
    "minecraft:hoglin_spawn_egg",
    "minecraft:horse_spawn_egg",
    "minecraft:husk_spawn_egg",
    "minecraft:iron_golem_spawn_egg",
    "minecraft:llama_spawn_egg",
    "minecraft:magma_cube_spawn_egg",
    "minecraft:mooshroom_spawn_egg",
    "minecraft:mule_spawn_egg",
    "minecraft:ocelot_spawn_egg",
    "minecraft:panda_spawn_egg",
    "minecraft:parrot_spawn_egg",
    "minecraft:phantom_spawn_egg",
    "minecraft:pig_spawn_egg",
    "minecraft:piglin_spawn_egg",
    "minecraft:piglin_brute_spawn_egg",
    "minecraft:pillager_spawn_egg",
    "minecraft:polar_bear_spawn_egg",
    "minecraft:pufferfish_spawn_egg",
    "minecraft:rabbit_spawn_egg",
    "minecraft:ravager_spawn_egg",
    "minecraft:salmon_spawn_egg",
    "minecraft:sheep_spawn_egg",
    "minecraft:shulker_spawn_egg",
    "minecraft:silverfish_spawn_egg",
    "minecraft:skeleton_spawn_egg",
    "minecraft:skeleton_horse_spawn_egg",
    "minecraft:slime_spawn_egg",
    "minecraft:snow_golem_spawn_egg",
    "minecraft:spider_spawn_egg",
    "minecraft:squid_spawn_egg",
    "minecraft:stray_spawn_egg",
    "minecraft:trader_llama_spawn_egg",
    "minecraft:tropical_fish_spawn_egg",
    "minecraft:turtle_spawn_egg",
    "minecraft:vex_spawn_egg",
    "minecraft:villager_spawn_egg",
    "minecraft:vindicator_spawn_egg",
    "minecraft:wandering_trader_spawn_egg",
    "minecraft:witch_spawn_egg",
    "minecraft:wither_skeleton_spawn_egg",
    "minecraft:wolf_spawn_egg",
    "minecraft:zoglin_spawn_egg",
    "minecraft:zombie_spawn_egg",
    "minecraft:zombie_horse_spawn_egg",
    "minecraft:zombie_villager_spawn_egg",
    "minecraft:zombified_piglin_spawn_egg",
    "minecraft:sniffer_spawn_egg",
    "minecraft:camel_spawn_egg",
    "minecraft:breeze_spawn_egg",
    "minecraft:armadillo_spawn_egg",
    "minecraft:bogged_spawn_egg",
]
_register(0.05, *_SPAWN_EGGS)


# ───────────────────────────────────────────────────────────────────
# PROJECTILES & MISC COMBAT ITEMS (already registered above mostly)
# Ensure nothing is missed.
# ───────────────────────────────────────────────────────────────────
for _id, _val in [
    ("minecraft:snowball", 0.2),
    ("minecraft:egg", 0.2),
]:
    ITEM_REWARDS.setdefault(_id, _val)


# ═══════════════════════════════════════════════════════════════════
#  DEFAULT REWARD  — for items not explicitly listed above.
#  A moderate 0.5 ensures unknown / new items still produce signal.
# ═══════════════════════════════════════════════════════════════════
_DEFAULT_REWARD = 0.5

# Multipliers applied on top of the base item reward depending on
# the event type.  Crafting something is harder than just picking
# it up, so it gets a bonus multiplier.
EVENT_MULTIPLIERS = {
    "item_picked_up": 1.0,
    "item_crafted":   2.0,     # crafting is worth 2× the base reward
    "block_broken":   0.8,     # breaking gives the raw material
    "block_placed":   1.2,     # placing shows intentional building
}
