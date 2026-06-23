# Baby-AI

Baby-AI is a self-evolving multimodal agent that learns to play **Minecraft** in real time, directly from screen pixels, game audio, and mod-streamed game events — no `python-minecraft` API, no bots, no replay datasets. It watches the screen and drives the mouse/keyboard the same way a human does, while a continuous Student/Teacher distillation loop trains it online as it plays.

A companion **Fabric mod** runs inside Minecraft itself and streams ground-truth events (block break/place, crafting, item pickups, death, etc.) to the Python process over a local TCP socket, and accepts virtual input (key/mouse/GUI actions) so the agent can act without hijacking your real cursor.

## What it actually does

- **Plays the game live.** `python main.py --minecraft` launches (or attaches to) Minecraft, captures the game window, and lets a small "Student" neural network choose actions every step (move, look, jump, attack, place/break blocks, open inventory, craft, etc.).
- **Learns continuously, not offline.** Every step is pushed into a prioritized replay buffer. A background "Teacher" model (larger, async) trains on that replay, and periodically distills its knowledge back down into the fast Student via KL + feature distillation — so the agent that's actually playing keeps getting better without ever stopping.
- **Multi-channel reward shaping.** Reward isn't a single scalar — there are ~25 independent reward channels (mining, crafting, combat, exploration, survival, item pickups, idle/spam penalties, intrinsic curiosity, etc.), each with its own toggle and weight, live-tunable from a GUI while the agent is playing.
- **Two systems of "thinking".** A fast reactive policy handles most steps; when uncertainty is high, the model triggers short-horizon latent rollout planning ("System 2"), and a separate hierarchical goal planner ("System 3") proposes and tracks long-horizon subgoals (e.g. "get wood" → "craft pickaxe" → "mine stone").
- **Live control panel.** A tkinter GUI lets you pause/stop the agent, toggle which reward channels and which input keys/buttons the AI is allowed to use, tune reward weights with sliders, set a "home" location, and watch live reward/step stats — all while training continues on a background thread.
- **Imitation learning.** You can take over the controls yourself (or run with `--offline`) to record human demonstrations, which get a priority boost in replay so the agent imitates them.

## Tech stack

### Python side (`baby_ai/`)
- **PyTorch** (CUDA build) — all models, training loops, mixed precision (AMP), `torch.compile` for the Student's inference path.
- **Custom Jamba-style temporal core** — Mamba‑2 selective state-space blocks (chunked SSD scan) interleaved with sparse Mixture-of-Experts FFN layers, giving O(1)-per-step inference with long effective context.
- **Flow Matching / Diffusion policy heads** — continuous 23-dim action generation (DDIM/Euler sampling), selectable per model.
- **REBEL** (regression-to-relative-reward RL) for online policy improvement from paired preferences, with an entropy bonus to avoid policy collapse.
- **Grounded Successor Features** — a decomposed value head ψ(s) ∈ ℝ^C predicting per-reward-channel expected discounted return, recombined as V(s) = ψ(s)·w so reward-weight changes re-value the policy zero-shot.
- **JEPA-style intrinsic curiosity** module for exploration bonus, decayed over training.
- **EWC + rehearsal consolidation** to reduce catastrophic forgetting across the continuous training stream.
- **Encoders:** CNN/ViT-ish vision encoder, log-mel audio encoder (librosa/`AudioPreprocessor`), code/graph encoder, sensor encoder, optional Slot Attention object-centric vision and Titans-style episodic K-V memory.
- **Screen/input I/O:** `mss` (screen capture), `opencv-python` (frame processing), `sounddevice` (audio capture), Win32 `ctypes` calls for cursor clipping/safety.
- **GUI:** `tkinter`/`ttk` control panel, threaded and DPI-aware, with persistent JSON settings (`baby_ai_settings.json`).
- **Config:** dataclass-based (`baby_ai/config.py`) with YAML save/load and `.env`-driven environment overrides (`python-dotenv`).
- **Networking:** `aiohttp`-based TCP client/event loop talking to the Fabric mod's socket server (`mod_bridge.py`).

### Minecraft side (`baby_ai_mod/`)
- **Java 21**, **Fabric Loader/API**, targeting **Minecraft 1.21.11**.
- **Fabric Mixins** injecting into Minecraft's input, screen, crafting, item-pickup, and death-screen classes to capture ground-truth events and to accept virtual mouse/keyboard/GUI actions without needing OS-level input injection (GLFW ignores Win32 `PostMessage`, so input/GUI interaction is routed through the mod itself).
- **Gradle** (`gradlew`) build producing the mod jar, auto-installed into `.minecraft/mods` by `setup.ps1`.
- A lightweight **TCP event bridge** (`EventBridge.java`) streaming JSON-ish events to Python and a `HomeManager`/custom commands for syncing the agent's "home" teleport point between the mod and Python.

## Project layout

```
main.py                      Entry point / CLI (run, demo, profile, offline train, minecraft play)
baby_ai_settings.json        Persisted GUI settings (reward toggles/weights, AI control toggles, home coords)
.env / .env.example          Minecraft paths, world/player identity, storage location
setup.ps1                    One-shot setup: builds the Fabric mod, creates the venv, installs deps

baby_ai/
  config.py                  All hyperparameters/paths as dataclasses (Student, Teacher, training, runtime, Minecraft)
  models/                    Student (10–30M params) and Teacher (50–100M params) networks
  encoders/                  Vision, audio, code, multimodal fusion, slot attention
  core/                      Policy, temporal core, planner, goals, predictive model, action tokenizer, communication
  learning/                  Rewards, reward channels, distillation, intrinsic curiosity, REBEL, successor features, crafting graph
  memory/                    Prioritized replay buffer, EWC/rehearsal consolidation
  runtime/                   Orchestrator + inference/learner/distill threads
  environments/minecraft/    Screen capture env, virtual input driver, Fabric-mod TCP bridge client
  safety/                    Action filtering / sandboxing
  ui/                        tkinter control panel, reward toggle/weight state, settings persistence
  utils/                     Logging, profiling, Muon optimizer, multi-GPU helpers, compression

baby_ai_mod/                 Fabric mod (Java) — Gradle project, mixins, TCP event bridge
```

## Setup

**Prerequisites:** Windows, an NVIDIA GPU (CUDA), Minecraft Java Edition 1.21.11 with a pre-created world, JDK 21 (auto-installed via winget if missing).

```powershell
# 1. Configure environment
copy .env.example .env
# edit .env: MC_DIR, MC_WORLD_NAME, MC_PLAYER_NAME, MC_PLAYER_UUID, BABY_AI_STORAGE, etc.

# 2. Build the Fabric mod, install it into .minecraft/mods, and create the Python venv
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup.ps1` builds `baby_ai_mod` with Gradle, copies the resulting jar into your `.minecraft/mods` folder, creates `.venv`, installs the CUDA build of PyTorch, then `requirements.txt`.

## Usage

```powershell
.venv\Scripts\python.exe main.py --minecraft       # launch/attach Minecraft and play live, training continuously
.venv\Scripts\python.exe main.py --minecraft --checkpoint path\to\ckpt.pt   # resume from a checkpoint

.venv\Scripts\python.exe main.py --demo            # quick sanity-check run with dummy multimodal data
.venv\Scripts\python.exe main.py --profile         # report model sizes, params, and inference latency

.venv\Scripts\python.exe main.py --offline --epochs 20          # replay recorded imitation data offline
.venv\Scripts\python.exe main.py --offline --multi-gpu          # same, sharded across all matching GPUs

.venv\Scripts\python.exe main.py --config my_config.yaml        # override default hyperparameters
```

While `--minecraft` is running, a control panel window lets you pause/stop the agent, flip reward channels and allowed input keys on/off, drag reward-weight sliders, and set the agent's home location — changes apply live without restarting.

Checkpoints, replay data, logs, and TensorBoard traces are written under `BABY_AI_STORAGE` (configured in `.env`), separate from the fast local SSD checkout.
