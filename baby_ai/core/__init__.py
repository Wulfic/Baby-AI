"""Core temporal integration and heads.

Key components:

- **JambaCore** — Mamba-2 SSM + Mixture-of-Experts temporal backbone.
  Maintains O(1)-per-step hidden state with infinite context caching.
- **DiffusionPolicyHead** — DDIM-based continuous action generation.
- **FlowMatchingPolicyHead** — ODE-based continuous action generation
  (Phase C upgrade; fewer inference steps than diffusion).
- **LatentWorldModel** — JEPA/RSSM-style latent dynamics model for
  curiosity reward and System 2 planning.
- **LatentMCTS** — System 2 “thinking”: batched latent-space Monte Carlo
  Tree Search triggered by high uncertainty.
- **GoalProposer / SubgoalPlanner / GoalConditioner / GoalMonitor** —
  System 3 hierarchical goal planning via FiLM-conditioned subgoals.
- **CommunicationHead** — Small GRU decoder for agent utterances.
- **ActionTokenizer** — VQ-BeT residual vector quantiser for discretising
  continuous actions (Phase E, optional).
"""
