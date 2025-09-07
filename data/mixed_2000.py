#!/usr/bin/env python3
import os
import glob
import shutil
from pathlib import Path
from typing import Optional, Tuple, Sequence

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.io import savemat
from tqdm import tqdm
from spenpy.spen import spen


# ============================== IO & Utils ==============================

def ensure_out_dirs(root: str | Path) -> Path:
    root = Path(root)
    (root / "hr").mkdir(parents=True, exist_ok=True)
    (root / "lr").mkdir(parents=True, exist_ok=True)
    (root / "phase_map").mkdir(parents=True, exist_ok=True)
    return root

def collect_nifti_files(*roots: Optional[str]) -> list[str]:
    files: list[str] = []
    for r in roots:
        if not r:
            continue
        files += sorted(glob.glob(os.path.join(r, "*.nii")))
        files += sorted(glob.glob(os.path.join(r, "*.nii.gz")))
    return files

def clean_stem(path_str: str) -> str:
    name = Path(path_str).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(path_str).stem

def random_rot90_batch(bwh: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent random 90° CCW rotations for each slice in (b,w,h)."""
    ks = rng.integers(0, 4, size=bwh.shape[0])
    out = [np.rot90(sl, k=int(k)) for sl, k in zip(bwh, ks)]
    return np.stack(out, axis=0)

def resize_cwh(x: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Resize (b,w,h) → (b,out_w,out_h) using bilinear."""
    t = torch.from_numpy(x).float().unsqueeze(0)   # (1,b,w,h)
    t = F.interpolate(t, size=out_size, mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy().astype(np.float32)

def load_nii_random_slices(
    file_path: str,
    percent: float,
    out_size: Tuple[int, int],
    rng: np.random.Generator,
    rand_rotate: bool,
) -> np.ndarray:
    """
    Load NIfTI, crop z, transpose to (b,w,h), optional random 0/90/180/270 per-slice,
    then resize to out_size. Returns float32 (b, w', h').
    """
    img = nib.load(file_path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)  # (X, Y, Z)
    z0 = int(percent * data.shape[2])
    z1 = int((1 - percent) * data.shape[2])
    vol = data[:, :, z0:z1]            # (X, Y, z_crop)
    bwh = np.transpose(vol, (2, 0, 1)) # (b, w, h)
    if rand_rotate:
        bwh = random_rot90_batch(bwh, rng)
    bwh = resize_cwh(bwh, out_size)
    return bwh

def expand_phase_to_batch(phase_chunk, like_tensor: torch.Tensor):
    """Ensure phase map shape matches (b,w,h)."""
    if torch.is_tensor(phase_chunk):
        if phase_chunk.ndim == 2:
            return phase_chunk.unsqueeze(0).expand_as(like_tensor)
        return phase_chunk
    phase_chunk = np.asarray(phase_chunk)
    if phase_chunk.ndim == 2:
        return np.repeat(phase_chunk[None, ...], like_tensor.shape[0], axis=0)
    return phase_chunk

def maybe_prefix_modality(stem: str, modality: str, prefix_modality: bool) -> str:
    return f"{modality}_{stem}" if prefix_modality else stem


# ============================== RAT loaders =============================

def load_rat_mat_slices(
    file_path: str,
    key: str,
    rng: np.random.Generator,
    out_size: Tuple[int, int] | None = None,
    rand_rotate: bool = True,
) -> np.ndarray:
    """
    Load rat HR stack from .mat (complex), take magnitude, then optional rotate/resize.
    Expects array like (B, W, H) (or (B, H, W)) under `key`.
    Returns float32 (B, w, h) magnitudes.
    """
    data = loadmat(file_path)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {file_path}. Keys={list(data.keys())}")

    arr = data[key]                    # could be complex
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in '{key}' from {file_path}, got shape {arr.shape}")

    # 1) magnitude from complex -> real
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    else:
        # already real; still ensure float32 downstream
        arr = np.asarray(arr)

    # 2) cast to float32 and clean up any non-finite values
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, copy=False)

    # 3) optional random 90° rotations per slice (B, w, h)
    if rand_rotate:
        arr = random_rot90_batch(arr, rng)

    # 4) optional resize to out_size
    if out_size is not None:
        arr = resize_cwh(arr, out_size)

    return arr  # (B, w, h) float32 magnitudes


# ============================== LR Mix-in (unchanged) ===================

def mix_in_test_lr(test_lr_path: str, lr_out_dir: Path) -> None:
    srcs = sorted(glob.glob(os.path.join(test_lr_path, "*.mat")))
    if not srcs:
        print(f"[mix_test_lr] No .mat files found under: {test_lr_path}")
        return
    lr_out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for s in srcs:
        dst = lr_out_dir / Path(s).name
        try:
            shutil.copy2(s, dst)
            copied += 1
        except Exception as e:
            print(f"[mix_test_lr] Failed to copy {s} → {dst}: {e}")
    print(f"[mix_test_lr] Copied {copied} LR files into: {lr_out_dir}")


# ============================== Core saver ==============================

def _save_hr_lr_phase_triplets(
    out_root: Path,
    simulator: spen,
    batch_np: np.ndarray,      # (b,w,h) float32
    stem_prefix: str,
    idxs: np.ndarray,          # original indices used for suffixes
    total_saved: int,
    max_n: int,
    pbar: tqdm,
) -> int:
    """Simulate LR & phase, then save triplets for each slice in `batch_np`."""
    if batch_np.size == 0:
        return total_saved

    with torch.no_grad():
        chunk = torch.from_numpy(batch_np).float()            # (b,w,h)
        lr_chunk = simulator.sim(chunk)                       # complex (b,w,h)
        phase_chunk = simulator.get_phase_map(chunk)          # real (b,w,h) or (w,h)
        phase_chunk = expand_phase_to_batch(phase_chunk, chunk)

        hr_np = chunk.cpu().numpy().astype(np.float32)
        lr_np = lr_chunk.detach().cpu().numpy().astype(np.complex64)
        phase_np = (phase_chunk.detach().cpu().numpy().astype(np.float32)
                    if torch.is_tensor(phase_chunk)
                    else np.asarray(phase_chunk, dtype=np.float32))

        for i in range(hr_np.shape[0]):
            if total_saved >= max_n:
                break
            suffix = f"_idx{int(idxs[i]):04d}.mat"
            savemat(out_root / "hr" / f"{stem_prefix}{suffix}", {"hr": hr_np[i]})
            savemat(out_root / "lr" / f"{stem_prefix}{suffix}", {"lr": lr_np[i]})
            savemat(out_root / "phase_map" / f"{stem_prefix}{suffix}", {"phase_map": phase_np[i]})
            total_saved += 1
            pbar.update(1)

    return total_saved


# ============================== Main Runner =============================

def run_all_mixed_ixi_and_rat(
    # --- IXI sources (optional) ---
    IXI_T1_path: Optional[str] = "/home/data1/musong/data/IXI/T1/nii",
    IXI_T2_path: Optional[str] = "/home/data1/musong/data/IXI/T2",
    # --- RAT sources (required for rat portion) ---
    rat_files: Sequence[str] = (
        "/home/data1/musong/data/spen/rat/RAT_train_1000_CP120.mat",
        "/home/data1/musong/data/spen/rat/RAT_train_1000_CP300.mat",
    ),
    rat_key: str = "ImagAll",
    # --- Output & sim ---
    out_root: str = "/home/data1/musong/workspace/python/spen-recons/data/mixed_IXI_RAT",
    out_size: Tuple[int, int] = (96, 96),
    # --- Composition control ---
    rat_fraction_total: float = 0.40,  # e.g., 0.40 → 40% of final set from all RAT files combined
    # --- per-source sampling details ---
    ixi_percent_z_crop: float = 0.30,  # crop from both ends in z for IXI
    rand_rotate: bool = True,          # random 0/90/180/270 per slice (applies to both IXI and RAT)
    max_n: int = 2000,                 # global target across RAT + IXI
    max_slices_per_file_ixi: Optional[int] = None,  # cap per NIfTI
    max_batch: int = 50,
    seed: int = 42,
    prefix_modality: bool = True,      # prefix stems with modality/source
    # --- Optional LR-only mix-in (unchanged) ---
    mix_test_lr: bool = False,
    test_lr_path: Optional[str] = None,
) -> int:
    """
    Build a dataset composed of:
      • RAT .mat HR slices (key=`rat_key`) → simulate LR via SPEN.
      • IXI T1/T2 NIfTI slices → simulate LR via SPEN.
    The final dataset will contain ~`rat_fraction_total` of RAT slices (split evenly
    across provided rat_files by default), and the rest from IXI (if available).

    Notes
    -----
    - If IXI paths are None/empty or contain no files, the IXI portion is skipped.
    - If `rat_fraction_total=0`, only IXI is used (if present).
    - If `rat_fraction_total=1`, only RAT is used.
    """
    assert 0.0 <= rat_fraction_total <= 1.0, "rat_fraction_total must be in [0,1]"

    out_root = ensure_out_dirs(out_root)
    rng = np.random.default_rng(seed)
    simulator = spen(acq_point=out_size)

    # ------------------- Determine quotas -------------------
    target_rat = int(round(max_n * rat_fraction_total))
    target_ixi = max_n - target_rat

    # Split rat quota evenly across rat files (weights could be added if needed)
    rat_files = list(rat_files)
    n_rat_files = len(rat_files)
    if n_rat_files == 0 and target_rat > 0:
        raise ValueError("rat_fraction_total > 0 but no rat_files provided.")

    per_rat_quota = [0] * n_rat_files
    if n_rat_files > 0:
        base = target_rat // n_rat_files
        rem = target_rat % n_rat_files
        per_rat_quota = [base + (1 if i < rem else 0) for i in range(n_rat_files)]

    # ------------------- Collect IXI file list (optional) -------------------
    ixi_files: list[tuple[str, str]] = []
    if target_ixi > 0:
        t1_files = collect_nifti_files(IXI_T1_path) if IXI_T1_path else []
        t2_files = collect_nifti_files(IXI_T2_path) if IXI_T2_path else []
        # label by modality
        ixi_files = [("T1", fp) for fp in t1_files] + [("T2", fp) for fp in t2_files]
        rng.shuffle(ixi_files)
        if not ixi_files:
            print("[warn] No IXI files found; the IXI portion will be skipped.")
            target_ixi = 0  # prevent looping below

    total_saved = 0
    with tqdm(total=max_n, desc="Saving slices", unit="slice") as pbar:

        # =================== 1) RAT portion ===================
        for i, rf in enumerate(rat_files):
            if total_saved >= max_n or per_rat_quota[i] <= 0:
                continue

            try:
                rat_stack = load_rat_mat_slices(
                    file_path=rf,
                    key=rat_key,
                    rng=rng,
                    out_size=out_size,
                    rand_rotate=rand_rotate,
                )  # (B,w,h)
            except Exception as e:
                print(f"[rat] Failed to load {rf}: {e}")
                continue

            B = rat_stack.shape[0]
            perm = rng.permutation(B)
            k = min(per_rat_quota[i], B)
            if k <= 0:
                continue

            # Take k random slices
            idxs = perm[:k]
            batch_np = rat_stack[idxs]  # (k,w,h)

            stem_prefix = maybe_prefix_modality(
                f"RAT_{Path(rf).stem}", "RAT", prefix_modality
            )  # e.g., "RAT_RAT_train_1000_CP120"
            total_saved = _save_hr_lr_phase_triplets(
                out_root=Path(out_root),
                simulator=simulator,
                batch_np=batch_np,
                stem_prefix=stem_prefix,
                idxs=idxs,
                total_saved=total_saved,
                max_n=max_n,
                pbar=pbar,
            )

            if total_saved >= max_n:
                break

        # =================== 2) IXI portion ===================
        if target_ixi > 0 and total_saved < max_n:
            needed_ixi = max_n - total_saved
            for modality, fp in ixi_files:
                if needed_ixi <= 0 or total_saved >= max_n:
                    break

                stem = maybe_prefix_modality(clean_stem(fp), modality, prefix_modality)
                # Load volume (b,w,h)
                try:
                    vol_bwh = load_nii_random_slices(
                        fp, percent=ixi_percent_z_crop, out_size=out_size,
                        rng=rng, rand_rotate=rand_rotate
                    )
                except Exception as e:
                    print(f"[ixi] Failed to load {fp}: {e}")
                    continue

                b = vol_bwh.shape[0]
                perm = rng.permutation(b)

                # Optional per-NIfTI cap
                if max_slices_per_file_ixi is not None:
                    perm = perm[:max_slices_per_file_ixi]

                # Now slice into batches and save until we hit `needed_ixi`
                for start in range(0, len(perm), max_batch):
                    if total_saved >= max_n:
                        break
                    end = min(start + max_batch, len(perm))
                    idxs = perm[start:end]
                    if idxs.size == 0:
                        continue
                    batch_np = vol_bwh[idxs]  # (b_chunk,w,h)

                    total_saved = _save_hr_lr_phase_triplets(
                        out_root=Path(out_root),
                        simulator=simulator,
                        batch_np=batch_np,
                        stem_prefix=stem,
                        idxs=idxs,
                        total_saved=total_saved,
                        max_n=max_n,
                        pbar=pbar,
                    )

                    needed_ixi = max_n - total_saved
                    if needed_ixi <= 0:
                        break

    # Optional LR mix-in
    if mix_test_lr:
        if test_lr_path:
            mix_in_test_lr(test_lr_path, Path(out_root) / "lr")
        else:
            print("mix_test_lr=True but no test_lr_path provided; skipping mix-in.")

    print(f"\nDone. Saved {total_saved} slice .mat files (target {max_n}).")
    print(f"Outputs under: {out_root}")
    return total_saved


# ============================== Example ==============================
if __name__ == "__main__":
    run_all_mixed_ixi_and_rat(
        # IXI (kept enabled; set these to None to skip IXI entirely)
        IXI_T1_path="/home/data1/musong/data/IXI/T1/nii",
        IXI_T2_path="/home/data1/musong/data/IXI/T2",
        # RAT sources (two files)
        rat_files=(
            "/home/data1/musong/data/spen/rat/RAT_train_1000_CP120.mat",
            "/home/data1/musong/data/spen/rat/RAT_train_1000_CP300.mat",
        ),
        rat_key="ImagAll",
        out_root="/home/data1/musong/workspace/python/spen-recons/data/mixed_2000",
        out_size=(96, 96),

        # === Composition example ===
        # For max_n=2000 and rat_fraction_total=0.40 → 800 RAT slices total
        # (≈400 from each rat file), and 1200 from IXI.
        rat_fraction_total=0.40,

        # IXI specifics
        ixi_percent_z_crop=0.30,
        rand_rotate=True,
        max_n=2010,
        max_slices_per_file_ixi=4,   # per NIfTI cap (None to disable)
        max_batch=50,
        seed=42,
        prefix_modality=True,
        mix_test_lr=False,
        test_lr_path="/home/data1/musong/workspace/python/spen-recons/test_data/lr",
    )
