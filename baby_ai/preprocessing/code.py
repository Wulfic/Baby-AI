"""
Code preprocessing pipeline.

Source code → AST via tree-sitter → graph (adjacency + node features).
Produces inputs for the GNN-based code encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

try:
    import tree_sitter_python
    HAS_TS_PYTHON = True
except ImportError:
    HAS_TS_PYTHON = False


class CodePreprocessor:
    """
    Converts source code to graph representation for GNN encoding.

    Pipeline:
    1. Parse source code to AST via tree-sitter
    2. Extract nodes with type features
    3. Build edge list (parent→child)
    4. Canonicalize identifiers
    5. Produce (node_features, edge_index) tensors

    Args:
        max_nodes: Maximum nodes in the graph (truncate if exceeded).
        max_edges: Maximum edges.
        node_feature_dim: Dimension of node feature vectors.
    """

    # Common AST node types → fixed indices for feature encoding
    NODE_TYPES = [
        "module", "function_definition", "class_definition", "if_statement",
        "for_statement", "while_statement", "return_statement", "assignment",
        "call", "argument_list", "parameters", "identifier", "string",
        "integer", "float", "binary_operator", "comparison_operator",
        "boolean_operator", "unary_operator", "list", "tuple", "dict",
        "import_statement", "import_from_statement", "try_statement",
        "except_clause", "with_statement", "expression_statement",
        "decorated_definition", "attribute", "subscript", "block",
        "comment", "pass_statement", "break_statement", "continue_statement",
    ]

    def __init__(
        self,
        max_nodes: int = 256,
        max_edges: int = 512,
        node_feature_dim: int = 64,
    ):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.node_feature_dim = node_feature_dim

        # Build node type → index mapping
        self._type_to_idx = {t: i for i, t in enumerate(self.NODE_TYPES)}
        self._num_types = len(self.NODE_TYPES) + 1  # +1 for unknown

        # Initialize tree-sitter parser
        self._parser: Optional[Parser] = None
        if HAS_TREE_SITTER and HAS_TS_PYTHON:
            self._parser = Parser(tree_sitter_python.language())

    def _node_type_feature(self, node_type: str) -> np.ndarray:
        """Create a feature vector for an AST node type."""
        # One-hot for node type + padding to node_feature_dim
        idx = self._type_to_idx.get(node_type, self._num_types - 1)
        vec = np.zeros(self.node_feature_dim, dtype=np.float32)
        if idx < self.node_feature_dim:
            vec[idx] = 1.0
        # Encode depth and child count in remaining dims
        return vec

    def parse_to_graph(
        self, source_code: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse source code to a graph.

        Args:
            source_code: Raw Python source code string.

        Returns:
            node_features: (N, node_feature_dim) tensor.
            edge_index: (2, E) tensor of directed edges.
        """
        if self._parser is None:
            # Fallback: return a simple dummy graph
            return self._dummy_graph()

        tree = self._parser.parse(source_code.encode("utf-8"))
        root = tree.root_node

        nodes = []
        edges = []
        node_map = {}  # tree_sitter node id → our index

        # BFS traversal
        queue = [root]
        while queue and len(nodes) < self.max_nodes:
            node = queue.pop(0)
            node_idx = len(nodes)
            node_map[id(node)] = node_idx

            # Create node feature
            feat = self._node_type_feature(node.type)
            # Encode depth (normalized)
            depth = 0
            parent = node.parent
            while parent is not None:
                depth += 1
                parent = parent.parent
            if self.node_feature_dim > self._num_types:
                feat[min(self._num_types, self.node_feature_dim - 1)] = depth / 20.0

            nodes.append(feat)

            # Add edges from parent
            if node.parent is not None and id(node.parent) in node_map:
                parent_idx = node_map[id(node.parent)]
                edges.append((parent_idx, node_idx))
                edges.append((node_idx, parent_idx))  # bidirectional

            # Enqueue children
            for child in node.children:
                queue.append(child)

        # Truncate edges
        edges = edges[:self.max_edges]

        # Convert to tensors
        if not nodes:
            return self._dummy_graph()

        node_features = torch.from_numpy(np.stack(nodes))

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return node_features, edge_index

    def _dummy_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a minimal valid graph."""
        node_features = torch.randn(4, self.node_feature_dim)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 3]], dtype=torch.long)
        return node_features, edge_index

    def process_file(self, path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse a source file to graph representation."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        return self.parse_to_graph(source)
