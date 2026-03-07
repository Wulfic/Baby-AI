### Overview
**Goal**: Build a **child‑like, continuously learning multimodal agent** that ingests raw video, audio, code, sensors, and curated internet signals, communicates interactively, and **supports low‑latency inference while learning** on hardware: **32 GB RAM, 11 GB RTX 2080 Ti, Intel i7‑11700K, ≤1 TB storage**.  
**Core pattern**: **Dual model** — a small low‑latency **Student** for inference and interaction, and a larger **Teacher** that learns continuously in the background and periodically distills improvements into the Student.

---

### Architecture and Components
**High level**: modular encoders per modality → compact continuous core → policy/communication head → prioritized replay and intrinsic reward module. Teacher is a larger, trainable copy; Student is a compact, fast inference copy updated by distillation.

| **Component** | **Role** | **Recommended** |
|---|---:|---|
| **Vision encoder** | Frame → embedding | Tiny CNN (MobileNetV2 style) 160–224 px |
| **Audio encoder** | Wave → embedding | 1D conv → log‑mel front end |
| **Code encoder** | Source → embedding | AST → Graph Neural Net |
| **Core learner** | Temporal integration | Small GRU or reservoir RNN |
| **Policy and communication** | Actions and utterances | Small MLP + lightweight seq generator |
| **Replay memory** | Continual rehearsal | Prioritized compressed replay |
| **Intrinsic module** | Curiosity and learning progress | ICM + learning progress estimator |
| **Distillation** | Teacher → Student updates | KL + feature matching on prioritized batches |

**Design targets**
- **Student size**: **10–30M parameters**, inference latency target **<200 ms**.
- **Teacher size**: **50–100M parameters**, asynchronous training.
- **Precision**: mixed precision FP16 on GPU; CPU for preprocessing.

---

### Data Preprocessing and Compact Representations
**Video**
- Sample **3–6 fps**; resize to **160–224 px**; optionally grayscale.
- Use short clips (0.5–2 s); optionally compute optical flow or event frames.
- Store compressed FP16 tensors.

**Audio**
- Downmix to mono; compute **log‑mel spectrograms** with 20–50 ms windows.
- Use short context windows (0.5–2 s).

**Code**
- Parse to **AST**, canonicalize identifiers, convert to graph adjacency + node features.

**Sensors and Internet**
- Normalize and timestamp sensor streams; quantize to fixed‑rate frames.
- Limit internet to curated APIs; fetch small snippets and metadata only.

**Storage**
- Keep replay compressed; cap on‑disk replay to **2–8 GB** depending on storage.

---

### Continuous Learning Loop Reward and Distillation
**Loop summary**
1. **Ingest** raw streams into modality buffers.
2. **Preprocess** to compact representations.
3. **Encode** each modality → compact state vector.
4. **Student inference**: immediate action/utterance from Student.
5. **Execute** action; observe response and next state.
6. **Compute rewards**: intrinsic + extrinsic + social.
7. **Store** transition in prioritized compressed replay.
8. **Teacher training**: asynchronous small gradient steps on Teacher.
9. **Distill** Teacher → Student periodically.
10. **Consolidate** via offline rehearsal and EWC.

**Reward composition**
- **Intrinsic reward**: **learning progress** (reduction in prediction error) + curiosity (ICM prediction error).
- **Communication reward**: positive when human/environment response reduces uncertainty or completes a task.
- **Extrinsic reward**: sparse task rewards when available.
- **Safety penalty**: negative for unsafe or disallowed actions.
- **Normalization**: normalize channels; anneal intrinsic weight from high to low over time.

**Distillation strategy**
- Distill on **prioritized minibatches** (high learning progress / novelty).
- Loss = **KL(Teacher logits || Student logits)** + encoder feature matching.
- Use **small learning rate** and **layer freezing** to keep Student stable.
- Atomic weight swap: prepare new Student weights in memory and swap pointers to avoid blocking inference.

---

### Resource Optimization and Latency Guarantees
**Resource budget**

| **Item** | **Target** |
|---|---:|
| **Student params** | 10–30M |
| **Teacher params** | 50–100M |
| **Replay on disk** | 2–8 GB compressed |
| **RAM usage** | ≤ 32 GB |
| **GPU memory** | ≤ 11 GB (FP16, micro‑batches) |
| **Storage** | ≤ 1 TB total |

**Optimizations**
- **Mixed precision** (AMP FP16) for training and inference.
- **Micro‑batches** and **gradient accumulation** to fit GPU memory.
- **Structured pruning** and **8‑bit quantization** for Student inference.
- **Asynchronous pipelines**: separate inference, learner, and distillation threads.
- **Compressed prioritized replay**: store FP16 + gzip; load prioritized samples into RAM for consolidation.
- **Preprocessing offload**: CPU handles heavy preprocessing; GPU used for encoder forward/backward only.
- **Latency safeguards**: keep Student graph static; avoid dynamic ops during inference; use lockless model swap.

---

### Implementation Roadmap Copy Paste Checklist
**Environment**
- [ ] Install PyTorch with AMP, PyTorch Geometric, audio libs, profiling tools.

**Data and preprocessors**
- [ ] Implement modality buffers and asynchronous preprocessors for video, audio, code, sensors, internet.
- [ ] Implement compact representation pipelines and compression for replay.

**Models**
- [ ] Implement **Student**: tiny vision CNN, audio conv, AST→GNN, GRU core, policy/comm head.
- [ ] Implement **Teacher**: same architecture scaled up.
- [ ] Add predictive head for next‑state prediction.

**Memory and intrinsic**
- [ ] Implement prioritized compressed replay buffer with metadata.
- [ ] Implement ICM curiosity module and learning progress estimator.

**Concurrency and distillation**
- [ ] Build **Inference thread** serving Student (nonblocking, target <200 ms).
- [ ] Build **Learner thread** training Teacher and ingesting replay.
- [ ] Implement **Distillation thread**: periodic Teacher → Student updates with atomic swap.

**Continual learning**
- [ ] Implement consolidation: rehearsal + EWC.
- [ ] Implement checkpointing and replay pruning.

**Safety and interfaces**
- [ ] Add sandboxed internet access and action filters.
- [ ] Build human feedback UI for binary/graded feedback.

**Profiling and optimization**
- [ ] Profile GPU/CPU memory and latency.
- [ ] Prune and quantize Student; reprofile.

**Evaluation**
- [ ] Create probes for perception, communication, continual learning, and latency.

---

### Minimal Hyperparameters and Operational Rules
- **Learning rates**: encoder/core \(1\mathrm{e}{-4}\); policy \(3\mathrm{e}{-4}\); distillation \(1\mathrm{e}{-5}\).  
- **Batch size**: 4–16 micro‑batches.  
- **Replay size**: 50k compressed transitions (adjust to disk cap).  
- **Distill cadence**: every 1k Teacher steps.  
- **Intrinsic weight**: start **1.0**, decay to **0.1** over weeks.  
- **Consolidation cadence**: every 1k–5k steps.  
- **Target inference latency**: **<200 ms**.

**Operational rules**
- **Human in loop** for high‑risk actions.
- **Sandbox internet** to curated APIs only.
- **Data minimization**: avoid storing raw human data; store embeddings or anonymized metadata.
- **Monitoring**: log reward anomalies and unsafe behavior; pause learning if thresholds exceeded.

---

### Final Copy Paste Quick Start
1. **Set up environment** with PyTorch and AMP.  
2. **Create Student**: tiny encoders + GRU + policy; export for inference.  
3. **Create Teacher**: larger copy for training.  
4. **Start threads**: Inference thread serving Student; Learner thread training Teacher and writing prioritized transitions to replay.  
5. **Every N steps** run distillation: sample prioritized batch → compute distillation loss → update Student weights atomically.  
6. **Run consolidation** periodically: rehearsal + EWC.  
7. **Profile and prune** Student until inference latency and memory targets met.  
8. **Iterate** on reward weights, distillation cadence, and evaluation probes.

---

If you want, I can now **generate a compact PyTorch skeleton** (student/teacher threads, replay buffer, distillation step) sized for your GPU that you can paste and run locally.