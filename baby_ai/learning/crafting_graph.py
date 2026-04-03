"""
Minecraft crafting-graph prior for goal conditioning.

Encodes the Minecraft crafting dependency DAG as a fixed graph
and learns node embeddings via a lightweight Graph Attention Network
(GAT-lite).  These embeddings are used to:

  1. Bias the GoalProposer toward achievable goals (recipe-aware).
  2. Provide structured goal representations aligned with the
     game's progression (e.g. wood → planks → sticks → crafting table).

Graph structure:
  - Nodes: item types (~150 key items from Minecraft 1.21)
  - Edges: directed dependency edges (A → B means B requires A)
  - Node features: one-hot item category + tier

The GAT produces a (num_items, embed_dim) embedding matrix that
can be looked up by item index to get a goal embedding.

Usage::

    graph = CraftingGraph(embed_dim=128)
    embeddings = graph()    # (num_items, embed_dim)
    goal_embed = graph.lookup(item_ids)  # (B, embed_dim)
    bias = graph.goal_bias(core_state)   # (B, num_candidates, embed_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Minecraft item list (condensed — key progression items) ────────────────
# Format: (item_name, category_id, tier)
# category: 0=raw, 1=wood, 2=stone, 3=metal, 4=tool, 5=food, 6=redstone,
#           7=combat, 8=nether, 9=end, 10=misc
_ITEMS = [
    # Raw materials
    ("log",             0, 0), ("cobblestone",    0, 1), ("iron_ore",       0, 2),
    ("gold_ore",        0, 2), ("coal",           0, 1), ("diamond",        0, 3),
    ("gravel",          0, 0), ("sand",           0, 0), ("clay",           0, 0),
    ("leather",         0, 1), ("string",         0, 0), ("feather",        0, 0),
    # Wood tier
    ("planks",          1, 1), ("stick",          1, 1), ("crafting_table", 1, 1),
    ("wooden_pickaxe",  4, 1), ("wooden_axe",     4, 1), ("wooden_shovel",  4, 1),
    ("wooden_sword",    7, 1), ("chest",          1, 1), ("fence",          1, 1),
    ("door_wood",       1, 1), ("bed",            1, 2),
    # Stone tier
    ("stone_pickaxe",   4, 2), ("stone_axe",      4, 2), ("stone_shovel",   4, 2),
    ("stone_sword",     7, 2), ("furnace",        2, 2), ("stone_slab",     2, 1),
    # Iron tier
    ("iron_ingot",      3, 3), ("iron_pickaxe",   4, 3), ("iron_axe",       4, 3),
    ("iron_shovel",     4, 3), ("iron_sword",     7, 3), ("iron_helmet",    7, 3),
    ("iron_chestplate", 7, 3), ("iron_leggings",  7, 3), ("iron_boots",     7, 3),
    ("bucket",          3, 3), ("shears",         4, 3), ("flint_and_steel",4, 3),
    # Gold tier
    ("gold_ingot",      3, 3), ("gold_sword",     7, 3), ("clock",          6, 3),
    # Diamond tier
    ("diamond_pickaxe", 4, 4), ("diamond_sword",  7, 4), ("diamond_helmet", 7, 4),
    ("diamond_armor",   7, 4),
    # Food
    ("bread",           5, 2), ("cooked_beef",    5, 2), ("cooked_pork",    5, 2),
    ("apple",           5, 1), ("cake",           5, 3),
    # Redstone
    ("redstone",        6, 2), ("torch",          6, 1), ("piston",         6, 3),
    ("lever",           6, 2), ("repeater",       6, 3), ("comparator",     6, 4),
    # Combat
    ("bow",             7, 2), ("arrow",          7, 1), ("shield",         7, 3),
    # Nether
    ("nether_rack",     8, 3), ("glowstone",      8, 3), ("blaze_rod",      8, 4),
    ("nether_star",     8, 5), ("beacon",         8, 5),
    # End / misc
    ("ender_pearl",     9, 4), ("eye_of_ender",   9, 4), ("dragon_egg",     9, 5),
    ("book",           10, 2), ("enchanting_table",10,3), ("anvil",         10, 4),
    ("wool",           10, 1), ("glass",          10, 1), ("paper",         10, 1),
]

# Dependency edges: (ingredient_idx → product_idx)
# Only a representative subset is hardcoded; the GAT propagates structure.
_EDGES_RAW = [
    # Wood chain
    ("log", "planks"), ("planks", "stick"), ("planks", "crafting_table"),
    ("planks", "chest"), ("planks", "fence"), ("planks", "door_wood"),
    ("stick", "wooden_pickaxe"), ("stick", "wooden_axe"),
    ("stick", "wooden_shovel"), ("stick", "wooden_sword"),
    ("stick", "torch"), ("planks", "bed"), ("wool", "bed"),
    # Stone chain
    ("wooden_pickaxe", "cobblestone"), ("cobblestone", "stone_pickaxe"),
    ("cobblestone", "stone_axe"), ("cobblestone", "stone_shovel"),
    ("cobblestone", "stone_sword"), ("cobblestone", "furnace"),
    ("cobblestone", "stone_slab"),
    # Iron chain
    ("stone_pickaxe", "iron_ore"), ("iron_ore", "iron_ingot"),
    ("iron_ingot", "iron_pickaxe"), ("iron_ingot", "iron_axe"),
    ("iron_ingot", "iron_shovel"), ("iron_ingot", "iron_sword"),
    ("iron_ingot", "iron_helmet"), ("iron_ingot", "iron_chestplate"),
    ("iron_ingot", "iron_leggings"), ("iron_ingot", "iron_boots"),
    ("iron_ingot", "bucket"), ("iron_ingot", "shears"),
    ("iron_ingot", "flint_and_steel"),
    # Gold chain
    ("iron_pickaxe", "gold_ore"), ("gold_ore", "gold_ingot"),
    ("gold_ingot", "gold_sword"), ("gold_ingot", "clock"),
    # Diamond chain
    ("iron_pickaxe", "diamond"), ("diamond", "diamond_pickaxe"),
    ("diamond", "diamond_sword"), ("diamond", "diamond_helmet"),
    ("diamond", "diamond_armor"),
    # Redstone
    ("stone_pickaxe", "redstone"), ("redstone", "piston"),
    ("redstone", "lever"), ("redstone", "repeater"),
    ("redstone", "comparator"),
    # Food
    ("crafting_table", "bread"), ("furnace", "cooked_beef"),
    ("furnace", "cooked_pork"),
    # Nether
    ("flint_and_steel", "nether_rack"), ("nether_rack", "glowstone"),
    ("nether_rack", "blaze_rod"), ("nether_star", "beacon"),
    # End
    ("ender_pearl", "eye_of_ender"), ("crafting_table", "book"),
    ("book", "enchanting_table"), ("iron_ingot", "anvil"),
    # Misc
    ("furnace", "glass"), ("crafting_table", "paper"),
    ("book", "enchanting_table"),
]


def _build_graph() -> tuple[int, torch.Tensor, torch.Tensor]:
    """Build item index and edge tensors from the raw definitions."""
    item2idx = {name: i for i, (name, _, _) in enumerate(_ITEMS)}
    num_items = len(_ITEMS)

    # Node feature: [one-hot category (11 dims) + tier (1 dim)]
    feats = []
    for _, cat, tier in _ITEMS:
        oh = [0.0] * 11
        oh[cat] = 1.0
        feats.append(oh + [float(tier) / 5.0])
    node_feats = torch.tensor(feats)  # (N, 12)

    # Edges → (2, E) tensor
    src, dst = [], []
    for a, b in _EDGES_RAW:
        if a in item2idx and b in item2idx:
            src.append(item2idx[a])
            dst.append(item2idx[b])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return num_items, node_feats, edge_index


class GATLiteLayer(nn.Module):
    """
    Single-head Graph Attention layer (simplified for small graphs).

    Args:
        in_dim:  Input node feature dimension.
        out_dim: Output node feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Linear(out_dim, 1, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:           (N, in_dim) node features
            edge_index:  (2, E) src/dst indices

        Returns:
            (N, out_dim) updated node features
        """
        h = self.linear(x)              # (N, out_dim)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Attention score: e_ij = LeakyReLU(a_src[i] + a_dst[j])
        e = F.leaky_relu(
            self.attn_src(h)[src_idx] + self.attn_dst(h)[dst_idx],
            negative_slope=0.2,
        )  # (E, 1)

        # Softmax over incoming edges per destination node
        N = x.size(0)
        attn = torch.zeros(N, N, device=x.device)
        attn[dst_idx, src_idx] = e.squeeze(-1)
        attn = torch.softmax(attn + (attn == 0).float() * -1e9, dim=-1)  # (N, N)
        # Zero out self-attention for non-edge pairs
        attn = attn * (attn > 1e-6).float()

        # Aggregate
        out = attn @ h  # (N, out_dim)
        return self.norm(F.elu(out) + h)  # residual


class CraftingGraph(nn.Module):
    """
    Minecraft crafting graph prior.

    Produces item embeddings via a 2-layer GAT over the crafting DAG.
    These embeddings are used to:
      1. Initialise goal candidates in GoalProposer.
      2. Constrain the agent to propose reasonable goals.

    Args:
        embed_dim:   Output embedding dimension per item.
        hidden_dim:  GAT hidden dimension.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        num_items, node_feats, edge_index = _build_graph()
        self.num_items = num_items

        # Register as buffers so they move with .to(device)
        self.register_buffer("node_feats", node_feats)   # (N, 12)
        self.register_buffer("edge_index", edge_index)   # (2, E)

        feat_dim = node_feats.size(-1)  # 12
        self.gat1 = GATLiteLayer(feat_dim, hidden_dim)
        self.gat2 = GATLiteLayer(hidden_dim, embed_dim)

        # Cache for embeddings (recomputed when parameters change)
        self._cached: torch.Tensor | None = None
        self._cached_device: torch.device | None = None

    def forward(self) -> torch.Tensor:
        """
        Compute and return item embeddings.

        Returns:
            (num_items, embed_dim)
        """
        x = self.gat1(self.node_feats, self.edge_index)
        x = self.gat2(x, self.edge_index)
        return x  # (N, embed_dim)

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        """Return cached item embeddings (reused across calls in inference)."""
        dev = self.node_feats.device
        if self._cached is None or self._cached_device != dev:
            self._cached = self.forward()
            self._cached_device = dev
        return self._cached

    def lookup(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for item indices.

        Args:
            item_ids: (B,) or (B, K) long tensor of item indices.

        Returns:
            (..., embed_dim) item embeddings.
        """
        emb = self.get_embeddings()  # (N, D)
        return emb[item_ids]

    def goal_affinities(
        self,
        goal_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity of proposed goals to all item embeddings.

        Useful for regularising GoalProposer toward known item goals.

        Args:
            goal_candidates: (B, K, embed_dim)

        Returns:
            (B, K, num_items) cosine similarity scores
        """
        emb = self.get_embeddings()  # (N, D)
        gc_norm = F.normalize(goal_candidates, dim=-1)    # (B, K, D)
        emb_norm = F.normalize(emb, dim=-1)               # (N, D)
        return torch.einsum("bkd,nd->bkn", gc_norm, emb_norm)  # (B, K, N)
