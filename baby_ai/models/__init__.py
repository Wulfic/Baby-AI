"""Composite model definitions (Student, Teacher).

Baby-AI uses a dual-model architecture:

- **StudentModel** — Compact (10–30 M params) agent optimised for
  real-time inference (<200 ms).  Runs in the inference thread.
- **TeacherModel** — Larger (50–100 M params) agent trained
  asynchronously on the replay buffer.  Periodically distils its
  knowledge into the Student via the DistillThread.

Both inherit from ``BabyAgentBase`` which owns the shared architecture:
  encoders → multimodal fusion → Jamba temporal core → policy head.
"""
