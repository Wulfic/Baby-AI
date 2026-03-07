"""
Compression utilities for replay storage.

Uses LZ4 for fast compression of FP16 tensors,
with optional gzip fallback for higher ratios.
"""

from __future__ import annotations

import io
import gzip
from typing import Dict, Any

import torch
import numpy as np

try:
    import lz4.frame as lz4f
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes via numpy (FP16 to save space)."""
    arr = t.detach().cpu().half().numpy()
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def bytes_to_tensor(data: bytes, device: str = "cpu") -> torch.Tensor:
    """Deserialize bytes back to a float32 tensor."""
    buf = io.BytesIO(data)
    arr = np.load(buf)
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def compress(data: bytes, method: str = "lz4") -> bytes:
    """Compress raw bytes. Prefers LZ4 for speed; falls back to gzip."""
    if method == "lz4" and HAS_LZ4:
        return lz4f.compress(data)
    return gzip.compress(data, compresslevel=1)


def decompress(data: bytes, method: str = "lz4") -> bytes:
    """Decompress bytes."""
    if method == "lz4" and HAS_LZ4:
        return lz4f.decompress(data)
    return gzip.decompress(data)


def compress_tensor(t: torch.Tensor, method: str = "lz4") -> bytes:
    """Serialize + compress a tensor."""
    return compress(tensor_to_bytes(t), method=method)


def decompress_tensor(data: bytes, method: str = "lz4", device: str = "cpu") -> torch.Tensor:
    """Decompress + deserialize a tensor."""
    return bytes_to_tensor(decompress(data, method=method), device=device)


def compress_transition(transition: Dict[str, Any], method: str = "lz4") -> bytes:
    """
    Compress an entire transition dict.

    Tensors are FP16-compressed; scalars are kept as-is.
    Returns a single compressed blob.
    """
    import pickle

    serializable = {}
    for k, v in transition.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = ("tensor", tensor_to_bytes(v))
        else:
            serializable[k] = ("scalar", v)

    raw = pickle.dumps(serializable)
    return compress(raw, method=method)


def decompress_transition(data: bytes, method: str = "lz4", device: str = "cpu") -> Dict[str, Any]:
    """Decompress a transition blob back to a dict."""
    import pickle

    raw = decompress(data, method=method)
    serializable = pickle.loads(raw)

    transition = {}
    for k, (typ, val) in serializable.items():
        if typ == "tensor":
            transition[k] = bytes_to_tensor(val, device=device)
        else:
            transition[k] = val
    return transition
