"""
Creative-workflow trackers for the Minecraft environment.

Extracted from ``screen_analyzer.py`` to keep individual modules
under ~800 lines.  Contains:

* :class:`BuildingStreakTracker` — rewards consecutive block placements.
* :class:`CreativeSequenceTracker` — state-machine rewarding the
  gather → craft → build pipeline.
"""

from __future__ import annotations


class BuildingStreakTracker:
    """
    Tracks consecutive block-placement events to reward sustained
    building activity.

    When the agent places multiple blocks in a short time window,
    the reward scales super-linearly — encouraging the agent to
    build structures rather than placing a single block and moving on.

    Args:
        decay_steps: Number of steps without placement before the
            streak resets to zero.
        max_streak: Streak cap for reward calculation.
    """

    def __init__(self, decay_steps: int = 15, max_streak: int = 50):
        self._decay_steps = decay_steps
        self._max_streak = max_streak
        self._streak: int = 0
        self._steps_since_placement: int = 0
        self._total_placements: int = 0

    def reset(self) -> None:
        self._streak = 0
        self._steps_since_placement = 0
        self._total_placements = 0

    def update(self, placed: bool) -> dict:
        """
        Update the streak with the latest placement result.

        Args:
            placed: True if a block was placed this step.

        Returns:
            dict with:
                streak (int): Current consecutive-placement streak.
                streak_bonus (float): 0.0-1.0 scaled bonus.
                total_placements (int): Lifetime placement count.
        """
        if placed:
            self._streak = min(self._streak + 1, self._max_streak)
            self._steps_since_placement = 0
            self._total_placements += 1
        else:
            self._steps_since_placement += 1
            if self._steps_since_placement > self._decay_steps:
                self._streak = max(0, self._streak - 1)

        # Super-linear bonus: sqrt(streak) / sqrt(max_streak)
        # Placing 4 blocks in a row -> 2x the single-block reward
        # Placing 25 blocks -> 5x the single-block reward
        streak_bonus = 0.0
        if self._streak > 0:
            streak_bonus = min(
                (self._streak ** 0.5) / (self._max_streak ** 0.5),
                1.0,
            )

        return {
            "streak": self._streak,
            "streak_bonus": streak_bonus,
            "total_placements": self._total_placements,
        }


class CreativeSequenceTracker:
    """
    Tracks the gather -> craft -> build workflow and rewards completing
    each stage of the creative pipeline.

    State machine:
    - IDLE -> GATHERING : agent breaks blocks / picks up items
    - GATHERING -> CRAFTING : agent opens crafting UI
    - CRAFTING -> BUILDING : agent places blocks after crafting
    - BUILDING -> IDLE : streak ends, cycle can restart

    Completing a full cycle (gather + craft + build) yields a large
    bonus.  Partial progress is also rewarded at lower rates.

    Args:
        gather_threshold: Minimum gather score to advance from IDLE.
        craft_threshold: Minimum craft score to advance.
        build_threshold: Minimum build score to complete the cycle.
    """

    # States
    IDLE = 0
    GATHERING = 1
    CRAFTING = 2
    BUILDING = 3

    def __init__(
        self,
        gather_threshold: float = 0.3,
        craft_threshold: float = 0.2,
        build_threshold: float = 0.3,
        timeout_steps: int = 500,
    ):
        self._gather_thresh = gather_threshold
        self._craft_thresh = craft_threshold
        self._build_thresh = build_threshold
        self._timeout = timeout_steps

        self._state: int = self.IDLE
        self._steps_in_state: int = 0
        self._gather_accum: float = 0.0
        self._gather_milestone_hit: bool = False
        self._cycles_completed: int = 0

    def reset(self) -> None:
        self._state = self.IDLE
        self._steps_in_state = 0
        self._gather_accum = 0.0
        self._gather_milestone_hit = False
        self._cycles_completed = 0

    def update(
        self,
        block_break: float,
        item_pickup: float,
        craft_score: float,
        block_place: float,
    ) -> dict:
        """
        Update the state machine with this step's signals.

        Args:
            block_break: Block break reward (0-1).
            item_pickup: Item pickup reward (0-1).
            craft_score: Crafting event score (0-1).
            block_place: Block placement score (0-1).

        Returns:
            dict with:
                state (int): Current state (0-3).
                state_name (str): Human-readable state name.
                stage_reward (float): Reward for this step's
                    progress through the creative pipeline.
                cycle_bonus (float): Large bonus on cycle completion.
                cycles (int): Total completed cycles.
        """
        self._steps_in_state += 1
        stage_reward = 0.0
        cycle_bonus = 0.0

        # Timeout — if stuck in any state too long, reset
        if self._steps_in_state > self._timeout:
            self._state = self.IDLE
            self._steps_in_state = 0
            self._gather_accum = 0.0

        if self._state == self.IDLE:
            # Transition to GATHERING on any resource acquisition
            gather_signal = block_break + item_pickup
            if gather_signal > 0.1:
                self._state = self.GATHERING
                self._steps_in_state = 0
                self._gather_accum = gather_signal
                stage_reward = 0.05  # one-time reward for starting

        elif self._state == self.GATHERING:
            # Only reward ACTIVE gathering — not just being in state
            gather_signal = block_break + item_pickup
            self._gather_accum += gather_signal

            # Reward proportional to actual gathering this step
            if gather_signal > 0:
                stage_reward = min(gather_signal * 0.15, 0.1)

            # Milestone: accumulated enough to advance
            if (self._gather_accum > self._gather_thresh
                    and not getattr(self, '_gather_milestone_hit', False)):
                stage_reward += 0.1  # one-time milestone bonus
                self._gather_milestone_hit = True

            # Transition to CRAFTING on UI open or crafting event
            if craft_score > self._craft_thresh:
                self._state = self.CRAFTING
                self._steps_in_state = 0
                self._gather_milestone_hit = False
                stage_reward = 0.3  # big reward for reaching crafting

        elif self._state == self.CRAFTING:
            # Only reward ACTUAL crafting, not sitting in crafting state
            if craft_score > 0:
                stage_reward = 0.4  # reward for actually crafting

            # Transition to BUILDING when agent places blocks
            if block_place > self._build_thresh:
                self._state = self.BUILDING
                self._steps_in_state = 0
                stage_reward = 0.5  # big reward for build after craft

        elif self._state == self.BUILDING:
            if block_place > 0:
                stage_reward = 0.2  # reward for continued building

            # Cycle complete after some building
            if self._steps_in_state > 10 and block_place < 0.1:
                # Building phase ended — cycle complete
                cycle_bonus = 1.0
                self._cycles_completed += 1
                self._state = self.IDLE
                self._steps_in_state = 0
                self._gather_accum = 0.0

        # Allow jumping directly to states from any position
        # (e.g., agent finds crafting table while in IDLE)
        if self._state == self.IDLE and craft_score > self._craft_thresh:
            self._state = self.CRAFTING
            self._steps_in_state = 0
            stage_reward = max(stage_reward, 0.3)

        state_names = {0: "idle", 1: "gathering", 2: "crafting", 3: "building"}

        return {
            "state": self._state,
            "state_name": state_names[self._state],
            "stage_reward": stage_reward,
            "cycle_bonus": cycle_bonus,
            "cycles": self._cycles_completed,
        }
