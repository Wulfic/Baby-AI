"""Runtime threads: inference, learner, distillation, orchestrator.

The runtime layer manages concurrency between the three main workloads:

- **InferenceThread** — Runs the Student model on incoming observations
  at <200 ms latency.  Triggers System 2 and System 3 when uncertainty
  is high.
- **LearnerThread** — Trains the Teacher model on prioritised replay
  data in a background loop (gradient steps + EWC consolidation).
- **DistillThread** — Periodically distils Teacher → Student via KL
  divergence + feature matching, then swaps the updated Student into
  the InferenceThread.
- **Orchestrator** — Top-level coordinator: builds all models, threads,
  and the replay buffer; handles checkpointing and system stats.
"""
