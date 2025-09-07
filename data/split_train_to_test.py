#!/usr/bin/env python3
"""
Split triplet data (hr/lr/phase_map) by MOVING a random subset from train -> test.

Example:
  python3 /home/data1/musong/workspace/python/spen-recons/data/split_train_to_test.py \
    --train_root /home/data1/musong/workspace/python/spen-recons/data/mixed_2000 \
    --test_root  /home/data1/musong/workspace/python/spen-recons/test_data_2025_9_7 \
    --test_size 10 \
    --seed 42

After running:
  - Selected *.mat files with identical basenames across hr, lr, phase_map
    are moved from train_root/{hr,lr,phase_map} -> test_root/{hr,lr,phase_map}.
  - A manifest of moved files is saved in test_root/split_manifest.txt
"""

import argparse
import os
from pathlib import Path
import random
import shutil
from typing import Set, List

SUBDIRS = ("hr", "lr", "phase_map")
EXT = ".mat"

def list_basenames(folder: Path) -> Set[str]:
    """Return set of basenames (e.g., 'T1_xxx_idx0001.mat') in a folder (files only)."""
    if not folder.exists():
        return set()
    return {p.name for p in folder.glob(f"*{EXT}") if p.is_file()}

def ensure_subdirs(root: Path) -> None:
    for sd in SUBDIRS:
        (root / sd).mkdir(parents=True, exist_ok=True)

def intersect_triplet_basenames(root: Path) -> Set[str]:
    """Basenames that exist in ALL three subfolders under root."""
    sets = []
    for sd in SUBDIRS:
        d = root / sd
        if not d.exists():
            raise FileNotFoundError(f"Missing subfolder: {d}")
        sets.append(list_basenames(d))
    if not sets:
        return set()
    common = sets[0]
    for s in sets[1:]:
        common = common.intersection(s)
    return common

def exclude_existing_in_test(candidates: Set[str], test_root: Path) -> Set[str]:
    """Exclude any basenames that already exist in test_root (in all or any subfolder)."""
    existing = set()
    for sd in SUBDIRS:
        existing |= list_basenames(test_root / sd)
    return candidates - existing

def move_triplet(basename: str, src_root: Path, dst_root: Path) -> None:
    """Move basename from each subdir hr/lr/phase_map; raise if any source missing."""
    for sd in SUBDIRS:
        src = src_root / sd / basename
        if not src.exists():
            raise FileNotFoundError(f"Source missing for '{basename}': {src}")
    # Ensure dest dirs exist
    ensure_subdirs(dst_root)
    # Move all three; if one move fails mid-way, you may end with partial move.
    # To be safer, copy-then-remove or move to temp then rename; here we move directly.
    for sd in SUBDIRS:
        src = src_root / sd / basename
        dst = dst_root / sd / basename
        shutil.move(str(src), str(dst))

def write_manifest(dst_root: Path, selected: List[str], total_after: int) -> None:
    manifest = dst_root / "split_manifest.txt"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(f"# Moved {len(selected)} files (triplets) into test set\n")
        for name in selected:
            f.write(name + "\n")
        f.write(f"# Total files now in test/hr: {total_after}\n\n")

def main():
    ap = argparse.ArgumentParser(description="Move a random test subset from train -> test.")
    ap.add_argument("--train_root", required=True, type=Path,
                    help="Path containing hr/, lr/, phase_map/ (source).")
    ap.add_argument("--test_root", required=True, type=Path,
                    help="Destination path for test set; will create hr/, lr/, phase_map/.")
    ap.add_argument("--test_size", required=True, type=int,
                    help="Number of triplets (hr+lr+phase_map) to move.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    ap.add_argument("--dry_run", action="store_true",
                    help="List what WOULD be moved, but do not move.")
    args = ap.parse_args()

    train_root: Path = args.train_root
    test_root: Path = args.test_root
    n_test: int = args.test_size
    seed: int = args.seed
    dry_run: bool = args.dry_run

    # Basic checks
    for sd in SUBDIRS:
        if not (train_root / sd).exists():
            raise FileNotFoundError(f"Missing train subfolder: {train_root/sd}")
    ensure_subdirs(test_root)

    # Compute candidates: present in ALL three train subfolders
    common_train = intersect_triplet_basenames(train_root)
    if not common_train:
        raise RuntimeError("No common triplet basenames found in train_root.")

    # Exclude anything already in test_root (avoid overlap if re-running)
    candidates = exclude_existing_in_test(common_train, test_root)
    if not candidates:
        raise RuntimeError("No eligible candidates after excluding files already in test_root.")

    if n_test > len(candidates):
        raise ValueError(f"Requested test_size={n_test} but only {len(candidates)} eligible triplets available.")

    # Deterministic sampling
    rng = random.Random(seed)
    selected = rng.sample(sorted(candidates), n_test)

    print(f"[INFO] Train root: {train_root}")
    print(f"[INFO] Test  root: {test_root}")
    print(f"[INFO] Eligible candidates: {len(candidates)}")
    print(f"[INFO] Will move: {len(selected)} triplets (seed={seed})")

    if dry_run:
        print("\n[DRY RUN] First 20 to be moved:")
        for name in selected[:20]:
            print("  ", name)
        print("\nNothing moved (dry run).")
        return

    # Move files
    moved_ok = 0
    for name in selected:
        move_triplet(name, train_root, test_root)
        moved_ok += 1

    total_in_test_hr = len(list_basenames(test_root / "hr"))
    write_manifest(test_root, selected, total_in_test_hr)

    print(f"[DONE] Moved {moved_ok} triplets.")
    print(f"[DONE] Now test/hr has {total_in_test_hr} files.")
    print(f"[DONE] Manifest: {test_root / 'split_manifest.txt'}")

if __name__ == "__main__":
    main()
