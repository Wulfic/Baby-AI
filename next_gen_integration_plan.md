# Baby-AI: Next-Generation Architecture Integration Plan

This document outlines the strategic integration of four bleeding-edge paradigms (Jamba, JEPA/RSSM World Models, Diffusion Policies, and System 2 Test-Time Compute) into the Baby-AI architecture. 

A primary constraint is maintaining the **Student Model's <200ms inference budget** on an RTX 2080 Ti and its **10-30M parameter size**, while allowing the **Teacher Model (50-100M)** to scale computationally.

---

## 🚀 Phase 1: Jamba-based Temporal Core (MoE + Mamba)
**Goal:** Replace the bottlenecked `nn.GRU` in `core/temporal.py` with a lightweight Jamba architecture (Mamba + Mixture of Experts) to achieve infinite context caching with $O(1)$ fast inference.

### Hardware Feasibility & Sizing constraints:
*   **Student Model:** We will implement a *Micro-Jamba*. Total parameters ~25M, but using **Top-1 Routing across 4 Experts**. 
    *   *Why this works:* Even with 25M total parameters, an MoE with 4 experts and Top-1 routing executes only ~1/4th of the FFN parameters per token. This keeps active inference parameters around **10-12M**, easily fitting the <200ms latency budget.
*   **Teacher Model:** Uses an *Iterative Jamba* with **8 Experts, Top-2 Routing**, pushing total parameters to ~80-90M but keeping training operations sparse and fast.

### File Modifications:
1.  **`baby_ai/core/temporal.py`:** 
    *   Deprecate `TemporalCore` (GRU).
    *   Implement `JambaCore`: interleaves Mamba-2 blocks with sparse MoE MLP layers.
2.  **`baby_ai/config.py`:**
    *   Add MoE hyper-parameters (`num_experts`, `top_k_routing`, `mamba_d_state`).

---

## 🌍 Phase 2: Latent World Modeling (V-JEPA / DreamerV3)
**Goal:** Upgrade `PredictiveHead` from standard ICM (predicting pixel/noisy states) to a Joint-Embedding Predictive Architecture (JEPA) or Recurrent State-Space Model (RSSM).

### Design Strategy:
*   Instead of `predict_next_state` trying to reconstruct the exact `fused_dim` tensor, the model will transition to predicting *dynamics in the latent space*. 
*   This module will borrow the `JambaCore` temporal state to predict future latent representations without needing to decode them back into audio/vision.
*   Curiosity (Intrinsic Reward) becomes the error between *predicted latent transition* and *actual latent transition*, filtering out stochastic noise (like Minecraft weather or random entity movement).

### File Modifications:
1.  **`baby_ai/core/predictive.py`:** 
    *   Refactor `PredictiveHead` into `LatentWorldModel`.
    *   Drop the old `inverse_model`; rely purely on forward latent dynamics (RSSM style).
2.  **`baby_ai/learning/intrinsic.py`:**
    *   Update the curiosity signal calculation to compute distance in the JEPA latent space rather than standard ICM MSE.

---

## 🎯 Phase 3: Continuous Action Generation (Diffusion Policy)
**Goal:** Replace the single-shot discrete `PolicyHead` with a lightweight Conditional Denoising Diffusion Probabilistic Model (DDPM/DDIM).

### Hardware Feasibility:
*   Diffusion historically requires many forward passes (slow). To hit the <200ms budget, we will use a **1D Continuous U-Net or MLP Diffusion network** with an ultra-fast solver (e.g., DPM-Solver or DDIM).
*   We constrain the sampling to **3-5 refinement steps** for the Student Model during real-time play, while the Teacher can train on 20+ steps.

### Design Strategy:
*   The system will handle compound actions natively (e.g., `Pitch: 12.5`, `Yaw: -5.0`, `Attack: True`, `Forward: True`) by generating a unified continuous latent matrix and denosing it over 4 steps conditioned on the `JambaCore` hidden state.

### File Modifications:
1.  **`baby_ai/core/policy.py`:**
    *   Create `DiffusionPolicyHead`.
    *   Implement DDIM sampling loop for the fast Student inference pass.
    *   Convert `action_dim` from discrete categorical logit outputs to continuous bounded parameters.

---

## 🧠 Phase 4: System 2 Thinking (Test-Time Search & Planning)
**Goal:** Allow the agent to pause and simulate outcomes in its "mind" using the Phase 2 World Model before committing to actions. This pause should also pause the world and display a message like "Thinking..." to the user in the game. As such the mod will need to interface with Minecraft's tick system to freeze the world state during System 2 compute.

### Hardware Feasibility:
*   Running brute-force search continuously will violate the 200ms budget. Therefore, System 2 compute must be **Threshold-Triggered**.
*   We use the entropy/uncertainty from the Diffusion Policy or World Model. If uncertainty spikes (e.g., encountering a hostile mob or attempting a complex crafting recipe), the agent allocates ~150ms to do a localized Latent Monte Carlo Tree Search (MCTS) utilizing the Jamba state.

### Design Strategy:
*   **Trigger Mechanism:** Check prediction error variance. If > threshold, suspend System 1.
*   **Latent Rollout:** Use the Phase 2 `LatentWorldModel` to unroll $T=5$ timesteps into the future for 8 possible action trajectories simultaneously (batched tensor ops).
*   **Evaluation:** Score the latent trajectories based on expected reward and pick the best initial action.

### File Modifications:
1.  **`baby_ai/runtime/inference_thread.py`:** 
    *   Add the uncertainty trigger condition before calling the standard `student.act()`.
2.  **`baby_ai/core/planner.py` (New File):** 
    *   Implement batched latent rollout (`LatentMCTS`).
    *   Bridge the `DiffusionPolicy` action generation with `LatentWorldModel` predictions.

