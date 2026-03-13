"""Utility subpackage.

Provides cross-cutting helpers:

- **logging** — Structured logger with file + console output.
- **compression** — LZ4/gzip tensor serialisation (FP16 + frame-compressed).
- **multigpu** — Multi-GPU offline training spawner + checkpoint averaging.
- **profiling** — Parameter counting, GPU memory reports, and end-to-end
  model profiling utilities.
"""
