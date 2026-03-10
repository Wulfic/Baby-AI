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
from baby_ai.config import System2Config
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
    - Non-blocking: callers submit requests and wait on events
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
        mod_bridge=None,
    ):
        self.student = student
        self.device = device
        self.target_latency_ms = target_latency_ms

        self._queue: queue.Queue[Optional[InferenceRequest]] = queue.Queue(maxsize=queue_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._hidden = None  # JambaState (Jamba temporal core)
        self._latency = LatencyTracker("inference")
        self._step = 0
        self._swap_lock = threading.Lock()

        # ── System 2: test-time planning ──
        self._s2_config = system2_config or System2Config()
        self._uncertainty = UncertaintyEstimator(
            threshold=self._s2_config.uncertainty_threshold,
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

    def start(self) -> None:
        """Start the inference thread."""
        if self._running:
            return
        self._running = True
        self.student.to(self.device).eval()
        # CUDA warmup — first forward pass compiles kernels
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
        """Run Student inference, potentially triggering System 2 planning."""
        # Move tensors to device
        inputs = {}
        for k, v in observation.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
            else:
                inputs[k] = v

        with self._swap_lock:
            result = self.student.act(hidden=self._hidden, **inputs)

        # Update hidden state
        self._hidden = result["hidden"].detach()

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
                log.info(
                    "System 2 triggered at step %d (uncertainty=%.3f > %.3f)",
                    self._step, u, self._s2_config.uncertainty_threshold,
                )

                # Request game pause via mod bridge (if available)
                if self._s2_config.pause_game and self._mod_bridge is not None:
                    self._send_pause(True)

                # Run latent MCTS planning
                plan_result = self._planner.plan(
                    core_state=result["core_state"],
                )

                # Override the System 1 action with the planned action
                result["action"] = plan_result["best_action"].squeeze(0)
                result["planning_ms"] = plan_result["planning_ms"]
                result["system2_triggered"] = True

                log.info(
                    "  Planning: %.1f ms, %d trajectories, expected_v=%.3f",
                    plan_result["planning_ms"],
                    plan_result["num_trajectories"],
                    plan_result["expected_value"].item(),
                )

                # Resume game
                if self._s2_config.pause_game and self._mod_bridge is not None:
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
        cpu_result["uncertainty"] = self._uncertainty.current
        return cpu_result

    def _send_pause(self, paused: bool) -> None:
        """Send pause/resume command to the Minecraft mod bridge."""
        try:
            if hasattr(self._mod_bridge, 'send_command'):
                cmd = "pause" if paused else "resume"
                self._mod_bridge.send_command({"command": cmd, "reason": "system2_planning"})
        except Exception as e:
            log.debug("Could not send pause command: %s", e)

    def reset_hidden(self) -> None:
        """Reset the Jamba hidden state (e.g., start of new episode)."""
        self._hidden = None

    def swap_model(self, new_state_dict: dict) -> None:
        """Atomically swap Student weights (called by distillation thread)."""
        with self._swap_lock:
            self.student.load_state_dict(new_state_dict)
        log.info("Model weights swapped at inference step %d", self._step)

    @property
    def stats(self) -> dict:
        return {
            "step": self._step,
            "last_latency_ms": self._latency.last_ms,
            "queue_size": self._queue.qsize(),
            "running": self._running,
            "system2_trigger_count": self._s2_trigger_count,
            "uncertainty": self._uncertainty.current,
        }
