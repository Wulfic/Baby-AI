"""Modality encoder subpackage.

Provides specialised neural-network encoders for each input modality:

- **AudioEncoder** — 1D convolutional encoder for log-mel spectrograms.
- **CodeEncoder**  — GNN-based encoder for code abstract syntax trees.
- **VisionEncoder** — Lightweight MobileNetV2-style CNN for image frames.
- **MultimodalFusion** — Gated fusion layer that combines all modality
  embeddings into a single ``fused_dim``-dimensional vector via learned
  per-modality attention weights.
"""
