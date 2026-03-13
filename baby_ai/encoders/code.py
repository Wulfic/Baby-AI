"""
Code encoder — AST → Graph Neural Network.

Parses source code into an Abstract Syntax Tree, converts to a graph,
and uses a GNN to produce a fixed-size embedding.

Uses torch_geometric for graph operations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class CodeEncoder(nn.Module):
    """
    GNN-based code encoder.

    Takes graph-structured AST representations and produces embeddings.

    Args:
        node_feature_dim: Input dimension of node features.
        hidden_dim: Hidden GNN layer dimension.
        embed_dim: Output embedding dimension.
        num_layers: Number of GCN layers.
    """

    def __init__(
        self,
        node_feature_dim: int = 64,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required for CodeEncoder. "
                "Install with: pip install torch-geometric"
            )

        self.node_embed = nn.Linear(node_feature_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, node_feature_dim) node features for all graphs in batch.
            edge_index: (2, E) edge connectivity.
            batch: (N,) graph membership for each node.

        Returns:
            (B, embed_dim) per-graph embedding.
        """
        # GCN message-passing layers: each convolution aggregates
        # features from neighbouring nodes in the AST graph.
        # Batch normalisation + ReLU activation after each layer.
        # Residual skip connections stabilise training for deeper GNNs
        # (only when input and output dims match, i.e. not the first layer).
        x = F.relu(self.node_embed(x))

        for conv, bn in zip(self.convs, self.bns):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            # Residual connection: only applies when dims match
            # (first layer maps node_embed → hidden_dim, so skip is possible
            # only from the second layer onward)
            if x.shape == residual.shape:
                x = x + residual

        # Global pooling
        x = global_mean_pool(x, batch)  # (B, hidden_dim)
        x = self.head(x)                # (B, embed_dim)
        return x

    @staticmethod
    def make_dummy_graph(
        num_nodes: int = 16,
        node_feature_dim: int = 64,
        num_edges: int = 30,
        batch_size: int = 1,
    ) -> dict:
        """Create a dummy batched graph for testing/profiling.

        Builds ``batch_size`` independent random graphs, concatenates
        their node features and edge indices into a single batched
        representation (as expected by torch_geometric), and returns
        them in a dict ready to be unpacked into ``forward()``.
        """
        all_x, all_edge_index, all_batch = [], [], []
        # offset tracks the running node count so edge indices are
        # adjusted to reference global (batched) node IDs.
        offset = 0
        for b in range(batch_size):
            x = torch.randn(num_nodes, node_feature_dim)
            # Randomly generate source/dest edges (may have duplicates/self-loops)
            src = torch.randint(0, num_nodes, (num_edges,))
            dst = torch.randint(0, num_nodes, (num_edges,))
            # Add offset to convert local node IDs (0..num_nodes-1)
            # to global IDs in the concatenated graph
            edge_index = torch.stack([src, dst]) + offset
            batch_vec = torch.full((num_nodes,), b, dtype=torch.long)

            all_x.append(x)
            all_edge_index.append(edge_index)
            all_batch.append(batch_vec)
            # Next graph's node IDs start after this one
            offset += num_nodes

        return {
            "x": torch.cat(all_x, dim=0),
            "edge_index": torch.cat(all_edge_index, dim=1),
            "batch": torch.cat(all_batch, dim=0),
        }
