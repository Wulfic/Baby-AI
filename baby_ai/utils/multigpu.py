"""
Multi-GPU utilities for Baby-AI.

Strategy: **Epoch-Split + Checkpoint Averaging**
    The replay data is temporal (sequential), so each GPU trains on a
    disjoint slice of epochs starting from the **same checkpoint**.
    With 10 epochs and 2 GPUs:
        - GPU 0 trains epochs 1–5
        - GPU 1 trains epochs 6–10
    Both workers run in parallel.  When all workers finish, their
    checkpoints are element-wise averaged into a single merged file.
    This cuts wall-time roughly by ``1 / n_gpus`` while still
    covering all requested epochs.

Requirements:
    - All GPUs must be from the same vendor (NVIDIA) so driver
      semantics are consistent.
    - All GPUs must have the same total VRAM so batch sizes and
      memory-budget assumptions hold equally.

Typical usage::

    python main.py --offline --epochs 10 --multi-gpu

This auto-detects eligible GPUs, splits the epochs across them,
spawns one worker per GPU, waits for all workers to finish, averages
their checkpoints, and saves the merged result as
``checkpoint_offline_final.pt``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import torch

log = logging.getLogger("multigpu")


# ── GPU validation ──────────────────────────────────────────────


def detect_gpus() -> list[dict]:
    """Return a list of dicts describing each CUDA GPU.

    Each dict has:
        index (int) – CUDA device ordinal
        name  (str) – e.g. "NVIDIA GeForce RTX 2080 Ti"
        vram_mb (int) – total VRAM in MiB
    """
    if not torch.cuda.is_available():
        return []
    gpus: list[dict] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "vram_mb": props.total_memory // (1024 * 1024),
        })
    return gpus


def validate_multi_gpu(gpus: list[dict] | None = None) -> list[dict]:
    """Check that ≥2 GPUs exist and all share the same vendor + VRAM.

    Returns the list of eligible GPUs on success.
    Raises ``RuntimeError`` with a clear message on failure.
    """
    if gpus is None:
        gpus = detect_gpus()

    if len(gpus) < 2:
        raise RuntimeError(
            f"Multi-GPU requires at least 2 CUDA GPUs, but found {len(gpus)}.  "
            "Run without --multi-gpu to train on a single GPU."
        )

    # ── Same vendor check (compare GPU name prefix) ─────────────
    # All supported GPUs are NVIDIA (PyTorch CUDA), so we compare the
    # full product name to ensure the same model family.  E.g. two
    # RTX 2080 Ti = OK; one RTX 2080 Ti + one RTX 3090 = rejected
    # because memory budgets and compute throughputs differ enough to
    # make checkpoint averaging less reliable.
    #
    # The user specifically requested: "multi gpu will only work if
    # vendor and vram amount are the same."
    reference_name = gpus[0]["name"]
    reference_vram = gpus[0]["vram_mb"]

    # Allow a small tolerance for VRAM (some drivers report slightly
    # different totals for the same physical card).
    _VRAM_TOLERANCE_MB = 256  # ~0.25 GB

    mismatches: list[str] = []
    for g in gpus[1:]:
        if g["name"] != reference_name:
            mismatches.append(
                f"  GPU {g['index']}: {g['name']}  (expected {reference_name})"
            )
        if abs(g["vram_mb"] - reference_vram) > _VRAM_TOLERANCE_MB:
            mismatches.append(
                f"  GPU {g['index']}: {g['vram_mb']} MiB VRAM"
                f"  (expected ~{reference_vram} MiB)"
            )
    if mismatches:
        raise RuntimeError(
            "Multi-GPU requires all GPUs to be the same model with "
            "matching VRAM.\n" + "\n".join(mismatches)
        )

    return gpus


# ── Checkpoint averaging ────────────────────────────────────────


def average_checkpoints(
    paths: Sequence[Path | str],
    output_path: Path | str,
) -> Path:
    """Load checkpoints from *paths*, average their state-dicts, and
    save the merged result to *output_path*.

    Only tensor values in the ``*_state_dict`` keys are averaged.
    Scalar metadata (step counters, config) is taken from the first
    checkpoint.

    Returns the output path.
    """
    paths = [Path(p) for p in paths]
    output_path = Path(output_path)

    if len(paths) < 2:
        raise ValueError("Need at least 2 checkpoints to average.")

    log.info("Averaging %d checkpoints → %s", len(paths), output_path)

    # Load all checkpoints to CPU
    checkpoints = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        checkpoints.append(torch.load(p, map_location="cpu", weights_only=False))
        log.info("  Loaded: %s", p)

    # Use first checkpoint as the base (keeps metadata, config, etc.)
    merged = checkpoints[0]
    n = len(checkpoints)

    # Keys whose values are state_dicts (OrderedDicts of tensors)
    _STATE_DICT_KEYS = [
        "student_state_dict",
        "teacher_state_dict",
        "curiosity_state_dict",
        "curiosity_proj_state_dict",
        "system3_state_dict",
    ]

    for sd_key in _STATE_DICT_KEYS:
        if sd_key not in merged:
            continue

        base_sd = merged[sd_key]
        # Only average keys that are tensors and exist in ALL checkpoints
        tensor_keys = [
            k for k, v in base_sd.items()
            if isinstance(v, torch.Tensor)
        ]

        for k in tensor_keys:
            # Check this key exists in all checkpoints with matching shape
            tensors = []
            for ckpt in checkpoints:
                sd = ckpt.get(sd_key, {})
                if k in sd and sd[k].shape == base_sd[k].shape:
                    tensors.append(sd[k].float())  # upcast to float32 for safe averaging
                else:
                    break  # shape mismatch or missing — skip this key
            else:
                # All checkpoints have this key with matching shape — average
                averaged = torch.stack(tensors).mean(dim=0)
                # Cast back to original dtype
                base_sd[k] = averaged.to(dtype=checkpoints[0][sd_key][k].dtype)

        merged[sd_key] = base_sd
        log.info("  Averaged %d tensors in '%s'", len(tensor_keys), sd_key)

    # Average scalar step counters (take the mean, round to int)
    step_keys = ["learner_step", "distill_count", "reward_composer_step"]
    for sk in step_keys:
        values = [ckpt.get(sk, 0) for ckpt in checkpoints]
        if all(isinstance(v, (int, float)) for v in values):
            merged[sk] = int(sum(values) / n)

    # Save merged checkpoint (atomic write)
    tmp_path = output_path.with_suffix(".pt.tmp")
    torch.save(merged, tmp_path)
    for _attempt in range(5):
        try:
            tmp_path.replace(output_path)
            break
        except (PermissionError, OSError):
            import time
            time.sleep(0.3 * (2 ** _attempt))

    log.info("Merged checkpoint saved: %s", output_path)
    return output_path


# ── Multi-GPU worker spawning ──────────────────────────────────


def run_multi_gpu_offline(
    epochs: int = 10,
    checkpoint_path: str | None = None,
    config_path: str | None = None,
) -> None:
    """Orchestrate multi-GPU offline training.

    1. Validate that all GPUs are eligible (same model + VRAM).
    2. Spawn one ``main.py --offline`` subprocess per GPU, each
       pinned to a single device via ``CUDA_VISIBLE_DEVICES``.
       All workers start from the **same checkpoint** and train
       ALL epochs independently on the full temporal replay data.
    3. Wait for all workers to finish (logs streamed concurrently).
    4. Average the per-GPU checkpoints into a final merged file.
    """
    import threading as _thr
    from baby_ai.config import CHECKPOINT_DIR
    from baby_ai.utils.logging import get_logger
    mgpu_log = get_logger("multigpu", log_file="multigpu.log")

    # ── Step 1: Validate GPUs ──────────────────────────────────
    gpus = detect_gpus()
    eligible = validate_multi_gpu(gpus)
    n_gpus = len(eligible)

    mgpu_log.info("=" * 60)
    mgpu_log.info("MULTI-GPU OFFLINE TRAINING  (%d GPUs, %d epochs each)", n_gpus, epochs)
    mgpu_log.info("=" * 60)
    for g in eligible:
        mgpu_log.info("  GPU %d: %s  (%d MiB)",
                       g["index"], g["name"], g["vram_mb"])

    # ── Step 2: Spawn workers ──────────────────────────────────
    # Each worker runs ALL epochs independently from the same
    # starting checkpoint.  CUDA_VISIBLE_DEVICES pins each worker
    # to exactly one physical GPU (the worker always sees device 0).
    python_exe = sys.executable
    script = str(Path(__file__).resolve().parent.parent.parent / "main.py")

    # Inherit current env (preserves BABY_AI_STORAGE, etc.)
    base_env = os.environ.copy()

    workers: list[subprocess.Popen] = []
    worker_tags: list[str] = []

    for rank, gpu in enumerate(eligible):
        tag = f"offline_final_gpu{rank}"
        worker_tags.append(tag)

        cmd = [
            python_exe, "-X", "utf8", script,
            "--offline",
            "--epochs", str(epochs),
            "--gpu-rank", str(rank),
            "--gpu-total", str(n_gpus),
        ]
        if checkpoint_path:
            cmd.extend(["--checkpoint", checkpoint_path])
        if config_path:
            cmd.extend(["--config", config_path])

        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])

        mgpu_log.info("Spawning worker rank=%d on GPU %d (CUDA_VISIBLE_DEVICES=%s)",
                       rank, gpu["index"], gpu["index"])
        mgpu_log.info("  cmd: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        workers.append(proc)

    # ── Step 3: Wait for all workers ───────────────────────────
    # Stream stdout from ALL workers concurrently using reader
    # threads so both GPUs' progress is interleaved in real-time.
    mgpu_log.info("Waiting for %d workers to complete...", n_gpus)

    def _stream_worker(rank: int, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                mgpu_log.info("[GPU %d] %s", rank, line)

    readers: list[_thr.Thread] = []
    for rank, proc in enumerate(workers):
        t = _thr.Thread(target=_stream_worker, args=(rank, proc), daemon=True)
        t.start()
        readers.append(t)

    # Wait for all readers to finish (they exit when stdout closes)
    for t in readers:
        t.join()

    # Collect exit codes
    failures: list[int] = []
    for rank, proc in enumerate(workers):
        proc.wait()
        if proc.returncode != 0:
            mgpu_log.error("Worker rank=%d exited with code %d", rank, proc.returncode)
            failures.append(rank)
        else:
            mgpu_log.info("Worker rank=%d completed successfully.", rank)

    if failures:
        mgpu_log.error(
            "Multi-GPU training FAILED: %d / %d workers had errors (ranks: %s). "
            "Skipping checkpoint averaging.",
            len(failures), n_gpus, failures,
        )
        return

    # ── Step 4: Average checkpoints ────────────────────────────
    ckpt_paths = [
        CHECKPOINT_DIR / f"checkpoint_{tag}.pt"
        for tag in worker_tags
    ]
    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        mgpu_log.error(
            "Cannot average — missing checkpoint(s): %s",
            [str(p) for p in missing],
        )
        return

    merged_path = CHECKPOINT_DIR / "checkpoint_offline_final.pt"
    average_checkpoints(ckpt_paths, merged_path)

    # Also save as "latest" for easy resumption
    import shutil
    latest_path = CHECKPOINT_DIR / "checkpoint_latest.pt"
    shutil.copy2(merged_path, latest_path)

    # Clean up per-GPU intermediates so they can't be confused
    # with the real merged checkpoint.
    for p in ckpt_paths:
        try:
            p.unlink()
            mgpu_log.info("  Removed intermediate: %s", p.name)
        except OSError:
            pass

    mgpu_log.info("=" * 60)
    mgpu_log.info("MULTI-GPU TRAINING COMPLETE")
    mgpu_log.info("  Workers          : %d", n_gpus)
    mgpu_log.info("  Epochs per GPU   : %d", epochs)
    mgpu_log.info("  Merged ckpt      : %s", merged_path)
    mgpu_log.info("  Latest ckpt      : %s", latest_path)
    mgpu_log.info("=" * 60)
