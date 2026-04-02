"""
Filter replay chunks to keep ONLY user-recorded demos (is_demo == 1.0).

Reads every chunk file, decompresses each transition, checks the
``is_demo`` flag, and rewrites compacted chunk files containing
only demo transitions.  AI-generated transitions are discarded.

Usage:
    python scripts/filter_demos.py            # dry-run (shows stats)
    python scripts/filter_demos.py --apply    # actually rewrite files
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baby_ai.config import REPLAY_DIR
from baby_ai.utils.compression import decompress_transition, compress_transition

CHUNK_SIZE = 500
HEADER_SLOT_BYTES = 8


def header_size() -> int:
    return 4 + CHUNK_SIZE * HEADER_SLOT_BYTES


def read_header(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        return [(0, 0)] * CHUNK_SIZE
    with open(path, "rb") as f:
        raw = f.read(header_size())
    if len(raw) < header_size():
        return [(0, 0)] * CHUNK_SIZE
    entries = []
    for i in range(CHUNK_SIZE):
        base = 4 + i * HEADER_SLOT_BYTES
        off, ln = struct.unpack_from("<II", raw, base)
        entries.append((off, ln))
    return entries


def write_chunk(path: Path, blobs: list[bytes]) -> None:
    """Write a fresh chunk file containing exactly *blobs* transitions."""
    entries: list[tuple[int, int]] = []
    offset = 0
    for b in blobs:
        entries.append((offset, len(b)))
        offset += len(b)
    # Pad remaining slots
    while len(entries) < CHUNK_SIZE:
        entries.append((0, 0))

    with open(path, "wb") as f:
        f.write(struct.pack("<I", CHUNK_SIZE))
        for off, ln in entries:
            f.write(struct.pack("<II", off, ln))
        for b in blobs:
            f.write(b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter replay chunks to demos only.")
    parser.add_argument("--apply", action="store_true", help="Actually rewrite files (default is dry-run).")
    args = parser.parse_args()

    chunks_dir = REPLAY_DIR / "ai"
    if not chunks_dir.exists():
        # Fallback to legacy path
        chunks_dir = REPLAY_DIR / "chunks"
    if not chunks_dir.exists():
        print(f"No replay directory found at {REPLAY_DIR / 'ai'}")
        return

    chunk_files = sorted(chunks_dir.glob("chunk_*.bin"))
    print(f"Found {len(chunk_files)} chunk files in {chunks_dir}")
    print()

    total_transitions = 0
    total_demos = 0
    total_ai = 0
    total_errors = 0

    # Collect all demo blobs for repacking
    demo_blobs: list[bytes] = []

    for cf in chunk_files:
        header = read_header(cf)
        hsz = header_size()

        with open(cf, "rb") as f:
            f.seek(hsz)
            blob_region = f.read()

        chunk_demos = 0
        chunk_ai = 0
        chunk_err = 0

        for slot, (off, ln) in enumerate(header):
            if ln == 0:
                continue
            total_transitions += 1
            blob = blob_region[off : off + ln]
            try:
                transition = decompress_transition(blob, device="cpu")
                is_demo_val = 0.0
                if "is_demo" in transition:
                    t = transition["is_demo"]
                    is_demo_val = float(t.item()) if hasattr(t, "item") else float(t)

                if is_demo_val > 0.5:
                    chunk_demos += 1
                    total_demos += 1
                    demo_blobs.append(blob)  # keep original compressed blob
                else:
                    chunk_ai += 1
                    total_ai += 1
            except Exception as e:
                chunk_err += 1
                total_errors += 1

        status = f"  {cf.name}: {chunk_demos} demos, {chunk_ai} AI, {chunk_err} errors"
        if chunk_demos > 0:
            status += "  <<<  KEEPER"
        print(status)

    print()
    print("=" * 60)
    print(f"Total transitions scanned: {total_transitions}")
    print(f"  Demos (is_demo=1):       {total_demos}")
    print(f"  AI-generated:            {total_ai}")
    print(f"  Errors:                  {total_errors}")
    print(f"  Demo blobs to keep:      {len(demo_blobs)}")

    if not args.apply:
        print()
        print("DRY RUN — no files modified. Run with --apply to rewrite.")
        return

    if total_demos == 0:
        print()
        print("No demo transitions found. Nothing to keep.")
        response = input("Delete ALL chunk files? [y/N]: ").strip().lower()
        if response == "y":
            for cf in chunk_files:
                cf.unlink()
            print(f"Deleted {len(chunk_files)} chunk files.")
        return

    # Repack demo blobs into new chunk files
    print()
    print("Repacking demo transitions...")

    # First, remove all existing chunk files
    for cf in chunk_files:
        cf.unlink()

    # Write demo blobs into fresh sequential chunks
    num_new_chunks = (len(demo_blobs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_id in range(num_new_chunks):
        start = chunk_id * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(demo_blobs))
        batch = demo_blobs[start:end]
        path = chunks_dir / f"chunk_{chunk_id:04d}.bin"
        write_chunk(path, batch)
        print(f"  Wrote {path.name} ({len(batch)} transitions)")

    total_bytes = sum(
        (chunks_dir / f"chunk_{i:04d}.bin").stat().st_size
        for i in range(num_new_chunks)
    )
    print()
    print(f"Done! {len(demo_blobs)} demo transitions in {num_new_chunks} chunk(s)")
    print(f"New size: {total_bytes / (1024**2):.1f} MB (was {sum(f.stat().st_size for f in chunk_files if f.exists()) / (1024**3):.2f} GB)")


if __name__ == "__main__":
    main()
