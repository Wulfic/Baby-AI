"""Preprocessing subpackage — raw data → compact representations.

Submodules
----------
video    : VideoPreprocessor — raw frames → resized, normalised (T,C,H,W) clips.
audio    : AudioPreprocessor — raw waveform → log-mel spectrogram windows.
code     : CodePreprocessor  — source code → AST graph (node features + edge index)
             via tree-sitter for the GNN-based code encoder.
sensors  : SensorPreprocessor — multi-channel numeric readings → online-normalised
             fixed-rate frames.
internet : InternetPreprocessor — sandboxed, rate-limited fetches from curated APIs
             → bag-of-characters embeddings.  Domain allowlist enforced.
"""
