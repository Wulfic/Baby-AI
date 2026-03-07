"""
GPU/CPU profiling utilities.

Measures memory occupancy, parameter counts, and inference latency.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import psutil

from baby_ai.utils.logging import get_logger

log = get_logger("profiler")


def count_parameters(model: torch.nn.Module) -> int:
    """Return total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    """Approximate model size in MB (all params, FP32)."""
    return count_parameters(model) * 4 / (1024 ** 2)


def gpu_memory_report() -> dict:
    """Return GPU memory stats in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "total_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "total_mb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
    }


def ram_usage_mb() -> float:
    """Return current process RSS in MB."""
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 2)


def profile_inference(
    model: torch.nn.Module,
    dummy_inputs: dict[str, torch.Tensor],
    warmup: int = 3,
    repeats: int = 10,
    device: str = "cuda",
) -> dict:
    """
    Profile a model's forward pass.

    Returns:
        dict with keys: mean_ms, min_ms, max_ms, gpu_mem_mb
    """
    model = model.to(device).eval()
    inputs = {k: v.to(device) for k, v in dummy_inputs.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mem = gpu_memory_report()
    result = {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "gpu_allocated_mb": mem["allocated_mb"],
    }
    log.info("Inference profile: %s", result)
    return result


def full_system_report(student: Optional[torch.nn.Module] = None, teacher: Optional[torch.nn.Module] = None) -> dict:
    """Print a summary of system resource usage."""
    report = {
        "ram_mb": ram_usage_mb(),
        "gpu": gpu_memory_report(),
    }
    if student is not None:
        report["student_params"] = count_parameters(student)
        report["student_size_mb"] = model_size_mb(student)
    if teacher is not None:
        report["teacher_params"] = count_parameters(teacher)
        report["teacher_size_mb"] = model_size_mb(teacher)
    log.info("System report: %s", report)
    return report
