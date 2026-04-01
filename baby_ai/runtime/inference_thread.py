"""
Inference thread — serves the Student model for real-time action selection.

Runs on a dedicated thread with the Student model pinned to GPU.
Handles incoming observation requests and returns actions within
the latency target (<200ms).

Integrates System 2 test-time compute: when the uncertainty estimator
detects a novel or difficult situation, the agent pauses and runs
Latent MCTS planning before committing to an action.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional

import torch

from baby_ai.core.planner import LatentMCTS, UncertaintyEstimator
from baby_ai.config import System2Config, System3Config, RuntimeConfig
from baby_ai.core.goals import GoalProposer, SubgoalPlanner, GoalMonitor
from baby_ai.utils.logging import get_logger, LatencyTracker

log = get_logger("inference", log_file="inference.log")


class InferenceRequest:
    """Container for an inference request."""

    def __init__(self, observation: Dict[str, Any], request_id: int = 0):
        self.observation = observation
        self.request_id = request_id
        self.result: Optional[Dict[str, Any]] = None
        self.event = threading.Event()


class InferenceThread:
    """
    Dedicated inference thread serving the Student model.

    The Student model is kept in eval mode and processes observations
    as they arrive via a queue. Results are returned through events.

    Features:
    - Event-driven: callers submit requests via a queue and block-
      wait on per-request ``threading.Event`` objects
    - Latency tracking: measures per-step inference time
    - Hidden state management: maintains Jamba hidden per "session"
    - Atomic model swap: accepts new weights without stopping

    Args:
        student: The Student model to serve.
        device: Device to run inference on.
        target_latency_ms: Target per-step latency.
        queue_size: Maximum pending requests.
    """

    def __init__(
        self,
        student: torch.nn.Module,
        device: str = "cuda",
        target_latency_ms: float = 200.0,
        queue_size: int = 64,
        system2_config: System2Config | None = None,
        system3_config: System3Config | None = None,
        mod_bridge=None,
        swap_lock: threading.Lock | None = None,
        runtime_config: RuntimeConfig | None = None,
    ):
        self._runtime_config = runtime_config or RuntimeConfig()
        self._original_student = student  # keep un-compiled reference
        self.student = student
        self.device = device
        self.target_latency_ms = target_latency_ms

        self._queue: queue.Queue[Optional[InferenceRequest]] = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._hidden = None  # JambaState (Jamba temporal core)
        self._latency = LatencyTracker("inference")
        self._step = 0
        self._swap_lock = swap_lock or threading.Lock()

        # ── System 2: test-time planning ──
        self._s2_config = system2_config or System2Config()
        self._uncertainty = UncertaintyEstimator(
            threshold=self._s2_config.uncertainty_threshold,
            warmup_steps=self._s2_config.warmup_steps,
            cooldown_steps=self._s2_config.cooldown_steps,
        )
        self._planner: LatentMCTS | None = None
        self._mod_bridge = mod_bridge
        self._s2_trigger_count = 0

        # Lazily initialise planner once student model is available
        if self._s2_config.enabled and hasattr(student, 'predictive') and hasattr(student, 'policy'):
            self._planner = LatentMCTS(
                world_model=student.predictive,
                policy=student.policy,
                num_trajectories=self._s2_config.num_trajectories,
                horizon=self._s2_config.planning_horizon,
                discount=self._s2_config.discount,
                budget_ms=self._s2_config.planning_budget_ms,
            )

        # ── System 3: hierarchical goal planning ──
        self._s3_config = system3_config or System3Config(enabled=False)
        self._active_goal: torch.Tensor | None = None      # (1, goal_dim) current subgoal
        self._subgoal_queue: list[torch.Tensor] = []        # remaining subgoals
        self._subgoal_idx: int = 0                          # index in current plan
        self._s3_trigger_count: int = 0
        self._s3_step: int = 0                              # steps since system 3 active
        self._goal_monitor: GoalMonitor | None = None
        self._goal_proposer: GoalProposer | None = None
        self._subgoal_planner: SubgoalPlanner | None = None

        if self._s3_config.enabled:
            state_dim = getattr(student, 'hidden_dim', 512)
            goal_dim = self._s3_config.goal_dim
            self._goal_proposer = GoalProposer(
                state_dim=state_dim,
                goal_dim=goal_dim,
                num_candidates=self._s3_config.num_goal_candidates,
                hidden_dim=self._s3_config.proposer_hidden_dim,
            ).to(device).eval()
            self._subgoal_planner = SubgoalPlanner(
                state_dim=state_dim,
                goal_dim=goal_dim,
                max_subgoals=self._s3_config.max_subgoals,
                num_layers=self._s3_config.planner_layers,
                num_heads=self._s3_config.planner_heads,
            ).to(device).eval()
            self._goal_monitor = GoalMonitor(
                achieve_threshold=self._s3_config.achieve_threshold,
                patience_steps=self._s3_config.patience_steps,
                min_replan_interval=self._s3_config.min_replan_interval,
            )
            log.info(
                "System 3 initialised: goal_dim=%d, num_candidates=%d, max_subgoals=%d",
                goal_dim, self._s3_config.num_goal_candidates, self._s3_config.max_subgoals,
            )

    def start(self) -> None:
        """Start the inference thread."""
        if self._running:
            return
        self._running = True
        self.student.to(self.device).eval()

        # ── Phase A: torch.compile() ──
        if self._runtime_config.compile_student and hasattr(torch, 'compile'):
            compile_mode = self._runtime_config.compile_mode
            log.info("Compiling Student model with torch.compile(mode='%s')...", compile_mode)
            self.student = torch.compile(
                self.student, mode=compile_mode, fullgraph=False,
            )
            log.info("torch.compile wrapper applied.")

        # CUDA warmup — first forward pass compiles kernels / triggers graph capture
        self._warmup()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="InferenceThread")
        self._thread.start()
        log.info("Inference thread started on %s", self.device)

    @torch.no_grad()
    def _warmup(self) -> None:
        """Run a dummy forward pass to warm up CUDA kernels."""
        dummy = torch.randn(1, 3, 360, 640, device=self.device)
        try:
            self.student.act(vision=dummy)
        except Exception:
            pass  # shape mismatches are fine — just warming up CUDA
        if self.device == "cuda":
            torch.cuda.synchronize()

    def stop(self) -> None:
        """Stop the inference thread gracefully."""
        self._running = False
        self._queue.put(None)  # sentinel
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        log.info("Inference thread stopped after %d steps.", self._step)

    def submit(self, observation: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        """
        Submit an observation and wait for the inference result.

        Args:
            observation: Dict with tensor inputs (vision, audio, etc.).
            timeout: Max seconds to wait for result.

        Returns:
            Dict with action, value, utterance, etc.
        """
        req = InferenceRequest(observation, request_id=self._step)
        self._queue.put(req, timeout=timeout)
        req.event.wait(timeout=timeout)
        if req.result is None:
            raise TimeoutError("Inference did not complete in time.")
        return req.result

    def _loop(self) -> None:
        """Main inference loop — processes requests from the queue."""
        while self._running:
            try:
                req = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if req is None:  # shutdown sentinel
                break

            try:
                with self._latency:
                    result = self._infer(req.observation)
                req.result = result

                if not self._latency.check(self.target_latency_ms):
                    log.warning(
                        "Latency %.1f ms > target %.1f ms at step %d",
                        self._latency.last_ms, self.target_latency_ms, self._step,
                    )
            except Exception as e:
                log.error("Inference error at step %d: %s", self._step, e)
                req.result = {"error": str(e)}
            finally:
                req.event.set()
                self._step += 1

    @torch.no_grad()
    def _infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Run Student inference, potentially triggering System 2/3 planning."""
        # Move tensors to device
        inputs = {}
        for k, v in observation.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
            else:
                inputs[k] = v

        # ── System 3: manage active goal ──
        goal_for_model = None
        if self._s3_config.enabled and self._active_goal is not None:
            goal_for_model = self._active_goal.to(self.device)

        with self._swap_lock:
            result = self.student.act(hidden=self._hidden, goal=goal_for_model, **inputs)

        # Update hidden state
        self._hidden = result["hidden"].detach()

        # ── System 3: goal monitoring and planning ──
        s3_replanned = False
        if (
            self._s3_config.enabled
            and self._goal_monitor is not None
        ):
            self._s3_step += 1
            core_state = result["core_state"]

            # Check if player died (trigger full replan)
            player_died = observation.get("player_died", False)
            if player_died and self._s3_config.replan_on_death:
                log.info("System 3: player died — full replan")
                self._trigger_system3(core_state, reason="death")
                s3_replanned = True
            elif self._active_goal is not None:
                # Monitor progress toward current subgoal
                # Pass goal_proj so the monitor can project core_state
                # into goal space for meaningful cosine similarity.
                _gp = self._goal_proposer.goal_proj if self._goal_proposer is not None else None
                status = self._goal_monitor.step(core_state, self._active_goal, goal_proj=_gp)
                if status == "advance":
                    log.info(
                        "System 3: subgoal %d/%d achieved, advancing",
                        self._subgoal_idx, len(self._subgoal_queue) + self._subgoal_idx,
                    )
                    self._advance_subgoal(core_state)
                elif status == "replan":
                    log.info(
                        "System 3: stuck on subgoal %d — replanning",
                        self._subgoal_idx,
                    )
                    self._trigger_system3(core_state, reason="stuck")
                    s3_replanned = True
            elif self._s3_step >= self._s3_config.warmup_steps:
                # No active goal: propose one right away or periodically
                if self._subgoal_idx == 0 or self._s3_step % self._s3_config.propose_every_n == 0:
                    log.info("System 3: no active goal — proposing new plan")
                    self._trigger_system3(core_state, reason="no_goal")
                    s3_replanned = True

        # ── System 2 uncertainty check ──
        used_planning = False
        if (
            self._s2_config.enabled
            and self._planner is not None
        ):
            u = self._uncertainty.update(
                value=result.get("value"),
                log_prob=result.get("log_prob"),
            )

            if self._uncertainty.should_plan:
                used_planning = True
                self._s2_trigger_count += 1
                s2_t_start = time.perf_counter()
                log.info(
                    "System 2 triggered at step %d (uncertainty=%.3f > %.3f)",
                    self._step, u, self._s2_config.uncertainty_threshold,
                )

                # Request game pause via mod bridge (if available)
                s2_paused_game = False
                if self._s2_config.pause_game and self._mod_bridge is not None:
                    self._send_pause(True)
                    s2_paused_game = True

                # ── Multi-round MCTS deliberation ──
                n_rounds = max(1, self._s2_config.deliberation_rounds)
                best_ev = float('-inf')
                best_plan: dict | None = None

                goal_for_plan = (
                    self._active_goal.to(self.device)
                    if self._active_goal is not None else None
                )

                for r in range(n_rounds):
                    plan_result = self._planner.plan(
                        core_state=result["core_state"],
                        goal=goal_for_plan,
                    )
                    ev = plan_result["expected_value"].item()
                    log.debug(
                        "  System 2 round %d/%d: expected_v=%.3f, %.1f ms",
                        r + 1, n_rounds, ev, plan_result["planning_ms"],
                    )
                    if ev > best_ev:
                        best_ev = ev
                        best_plan = plan_result

                # Override the System 1 action with the best planned action
                result["action"] = best_plan["best_action"].squeeze(0)
                result["planning_ms"] = best_plan["planning_ms"]
                result["system2_triggered"] = True

                s2_elapsed_ms = (time.perf_counter() - s2_t_start) * 1000

                # ── Enforce minimum deliberation floor ──
                s2_min_ms = self._s2_config.min_deliberation_ms
                if s2_elapsed_ms < s2_min_ms:
                    remaining_s = (s2_min_ms - s2_elapsed_ms) / 1000.0
                    log.info(
                        "  System 2 deliberated in %.1f ms, holding pause for "
                        "%.0f ms more (min_deliberation_ms=%.0f)",
                        s2_elapsed_ms, remaining_s * 1000, s2_min_ms,
                    )
                    time.sleep(remaining_s)
                    s2_elapsed_ms = (time.perf_counter() - s2_t_start) * 1000

                log.info(
                    "  System 2 done: %d rounds, %.1f ms total, best expected_v=%.3f",
                    n_rounds, s2_elapsed_ms, best_ev,
                )

                # Mark triggered so cooldown starts
                self._uncertainty.mark_triggered()

                # Resume game
                if s2_paused_game:
                    self._send_pause(False)

        # Move results back to CPU for downstream use
        cpu_result = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                cpu_result[k] = v.cpu()
            else:
                cpu_result[k] = v

        cpu_result["latency_ms"] = self._latency.last_ms
        cpu_result["system2_triggered"] = used_planning
        cpu_result["system3_replanned"] = s3_replanned
        cpu_result["uncertainty"] = self._uncertainty.current
        # Expose the active goal to the main loop for transitions
        if self._active_goal is not None:
            cpu_result["goal_embedding"] = self._active_goal.cpu()
        return cpu_result

    def _send_pause(self, paused: bool) -> None:
        """Send pause/resume to the mod bridge and wait for ack."""
        try:
            if self._mod_bridge is not None and hasattr(self._mod_bridge, 'send_pause'):
                timeout = self._s2_config.pause_settle_ms / 1000.0 + 0.3
                success = self._mod_bridge.send_pause(
                    paused, reason="system2_planning", timeout=timeout,
                )
                cmd = "pause" if paused else "resume"
                log.info(
                    "System 2 %s %s",
                    cmd, "confirmed" if success else "unconfirmed (timeout)",
                )
        except Exception as e:
            log.warning("Could not send pause command: %s", e)

    # ── System 3: goal planning ──────────────────────────────

    @torch.no_grad()
    def _trigger_system3(self, core_state: torch.Tensor, reason: str = "unknown") -> None:
        """
        Full System 3 replan: propose a high-level goal, decompose
        into subgoals, and populate the subgoal queue.

        The game is PAUSED for the entire duration — System 3 is
        allowed to take as long as it needs.  This is the whole point
        of the pause: thoughtful, unhurried deliberation.

        Iterative refinement: runs multiple rounds of goal proposal
        and subgoal planning, keeping the best plan across rounds.
        A minimum deliberation floor ensures the game visibly pauses.
        """
        if self._goal_proposer is None or self._subgoal_planner is None:
            return

        self._s3_trigger_count += 1
        t_start = time.perf_counter()

        # ─ Pause game so the world freezes while we think ─
        paused_game = False
        if self._s3_config.pause_game_on_replan and self._mod_bridge is not None:
            self._send_pause_s3(True)
            paused_game = True

        device = core_state.device
        state = core_state.detach()  # (1, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # ── Iterative refinement: run multiple rounds, keep best plan ──
        n_rounds = max(1, self._s3_config.deliberation_rounds)
        best_score = float('-inf')
        best_goal = None
        best_subgoals = None
        best_done_probs = None
        best_round_info = (0, 0, 0.0)  # (round, best_idx, score)

        for r in range(n_rounds):
            # ─ 1. Propose candidate goals ─
            proposal = self._goal_proposer(state)
            goals = proposal["goals"]    # (1, K, goal_dim)
            scores = proposal["scores"]  # (1, K)

            # Pick the highest-scoring goal this round
            round_best_idx = scores[0].argmax()
            round_score = scores[0, round_best_idx].item()
            round_goal = goals[0, round_best_idx].unsqueeze(0)  # (1, goal_dim)

            # ─ 2. Decompose into subgoal sequence ─
            plan_out = self._subgoal_planner(state, round_goal)

            log.debug(
                "System 3 deliberation round %d/%d: chose goal #%d (score=%.3f)",
                r + 1, n_rounds, round_best_idx.item(), round_score,
            )

            # Keep this plan if it beats the previous best
            if round_score > best_score:
                best_score = round_score
                best_goal = round_goal
                best_subgoals = plan_out["subgoals"]
                best_done_probs = plan_out["done_probs"]
                best_round_info = (r + 1, round_best_idx.item(), round_score)

        log.info(
            "System 3 [%s]: %d deliberation rounds, best plan from round %d "
            "(goal #%d, score=%.3f)",
            reason, n_rounds, best_round_info[0],
            best_round_info[1], best_round_info[2],
        )

        # ── Build final subgoal queue from best plan ──
        subgoals = best_subgoals    # (1, T, goal_dim)
        done_probs = best_done_probs  # (1, T)

        # Trim at the first step where done_prob > 0.5, or keep all
        done_mask = done_probs[0] > 0.5
        if done_mask.any():
            T_end = done_mask.nonzero(as_tuple=True)[0][0].item() + 1
        else:
            T_end = subgoals.size(1)
        T_end = max(T_end, 1)  # at least one subgoal

        # Build the subgoal queue
        self._subgoal_queue = [
            subgoals[0, t].unsqueeze(0)  # (1, goal_dim)
            for t in range(T_end)
        ]
        self._subgoal_idx = 0
        self._active_goal = self._subgoal_queue[0]

        # Reset the monitor for the fresh plan
        if self._goal_monitor is not None:
            self._goal_monitor.reset_full()

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # ── Enforce minimum deliberation floor ──
        # If the neural net rounds completed too quickly, hold the
        # pause so the game visibly freezes and the agent is clearly
        # "thinking".  The remaining budget is genuine idle wait.
        min_ms = self._s3_config.min_deliberation_ms
        if elapsed_ms < min_ms:
            remaining_s = (min_ms - elapsed_ms) / 1000.0
            log.info(
                "System 3 deliberated in %.1f ms, holding pause for %.0f ms more "
                "(min_deliberation_ms=%.0f)",
                elapsed_ms, remaining_s * 1000, min_ms,
            )
            time.sleep(remaining_s)
            elapsed_ms = (time.perf_counter() - t_start) * 1000

        log.info(
            "System 3 replanned: %d subgoals, %.1f ms total (%d rounds, game frozen)",
            len(self._subgoal_queue), elapsed_ms, n_rounds,
        )

        # Resume game — thinking is done
        if paused_game:
            self._send_pause_s3(False)

    def _advance_subgoal(self, core_state: torch.Tensor) -> None:
        """Move to the next subgoal, or trigger full replan if done."""
        self._subgoal_idx += 1
        if self._subgoal_idx < len(self._subgoal_queue):
            self._active_goal = self._subgoal_queue[self._subgoal_idx]
            if self._goal_monitor is not None:
                self._goal_monitor.reset()  # reset patience for new subgoal
            log.info(
                "System 3: advanced to subgoal %d/%d",
                self._subgoal_idx, len(self._subgoal_queue),
            )
        else:
            # All subgoals completed — celebrate then replan
            log.info(
                "System 3: all %d subgoals completed — replanning",
                len(self._subgoal_queue),
            )
            self._trigger_system3(core_state, reason="plan_complete")

    def _send_pause_s3(self, paused: bool) -> None:
        """Send pause/resume for System 3 replanning and wait for ack."""
        try:
            if self._mod_bridge is not None and hasattr(self._mod_bridge, 'send_pause'):
                timeout = self._s3_config.pause_settle_ms / 1000.0 + 0.3
                success = self._mod_bridge.send_pause(
                    paused, reason="system3_planning", timeout=timeout,
                )
                cmd = "pause" if paused else "resume"
                log.info(
                    "System 3 %s %s",
                    cmd, "confirmed" if success else "unconfirmed (timeout)",
                )
        except Exception as e:
            log.warning("Could not send System 3 pause command: %s", e)

    def set_mod_bridge(self, mod_bridge) -> None:
        """Attach the mod bridge after construction.

        The env creates the ModBridge, but the InferenceThread is
        built before the env exists.  Call this once the env is ready
        so System 2 planning can freeze/resume the game.
        """
        self._mod_bridge = mod_bridge
        log.info("ModBridge attached to InferenceThread (pause_game=%s)",
                 self._s2_config.pause_game)

    def reset_hidden(self) -> None:
        """Reset the Jamba hidden state (e.g., start of new episode)."""
        self._hidden = None

    def reset_system3(self) -> None:
        """Reset all System 3 goal state (e.g., start of new episode)."""
        self._active_goal = None
        self._subgoal_queue = []
        self._subgoal_idx = 0
        self._s3_step = 0
        if self._goal_monitor is not None:
            self._goal_monitor.reset_full()
        log.info("System 3 state reset (episode boundary)")

    def swap_model(self, new_state_dict: dict) -> None:
        """Atomically swap Student weights (called by distillation thread)."""
        with self._swap_lock:
            # If model is compiled, load into the original module
            target = getattr(self.student, '_orig_mod', self.student)
            target.load_state_dict(new_state_dict)
        log.info("Model weights swapped at inference step %d", self._step)

    @property
    def stats(self) -> dict:
        return {
            "step": self._step,
            "last_latency_ms": self._latency.last_ms,
            "queue_size": self._queue.qsize(),
            "running": self._running,
            "system2_trigger_count": self._s2_trigger_count,
            "system3_trigger_count": self._s3_trigger_count,
            "uncertainty": self._uncertainty.current,
            "active_goal": self._active_goal is not None,
            "subgoal_queue_len": len(self._subgoal_queue),
            "subgoal_idx": self._subgoal_idx,
        }

    def system3_state_dict(self) -> dict:
        """Return state dicts for System 3 modules (for checkpointing)."""
        sd = {}
        if self._goal_proposer is not None:
            sd["goal_proposer"] = self._goal_proposer.state_dict()
        if self._subgoal_planner is not None:
            sd["subgoal_planner"] = self._subgoal_planner.state_dict()
        return sd

    def load_system3_state_dict(self, sd: dict) -> None:
        """Load System 3 module weights from a checkpoint."""
        if self._goal_proposer is not None and "goal_proposer" in sd:
            self._goal_proposer.load_state_dict(sd["goal_proposer"])
            log.info("System 3 GoalProposer weights loaded")
        if self._subgoal_planner is not None and "subgoal_planner" in sd:
            self._subgoal_planner.load_state_dict(sd["subgoal_planner"])
            log.info("System 3 SubgoalPlanner weights loaded")
