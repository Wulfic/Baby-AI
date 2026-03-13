"""Replay memory and consolidation.

Modules:

- **replay_buffer** — Prioritised replay buffer backed by an on-disk
  chunk store.  Uses a binary SumTree for O(log n) sampling and
  supports sequential (temporal-order) iteration for offline training.
- **consolidation** — Elastic Weight Consolidation (EWC) manager:
  periodically computes Fisher information over recent replay data
  and adds a quadratic penalty to prevent catastrophic forgetting.
"""
