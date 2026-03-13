"""Learning: rewards, intrinsic motivation, distillation, continual learning.

Modules:

- **rewards** — RewardComposer: z-score normalisation + dynamic weight
  composition for multi-channel extrinsic/intrinsic rewards.
- **intrinsic** — JEPACuriosity: thin wrapper around the LatentWorldModel
  that surfaces latent prediction error as a curiosity signal.
- **distillation** — DistillationManager: KL + feature-matching transfer
  from Teacher to Student with progressive curriculum.
- **rebel** — REBELLoss: preference-based RL via Bradley-Terry loss
  (Phase D).
- **augmentations** — Asymmetric data augmentation for distillation.
- **consolidation** — EWC (Elastic Weight Consolidation) to prevent
  catastrophic forgetting.
- **item_rewards / item_reward_data** — Per-Minecraft-item reward tables.
"""
