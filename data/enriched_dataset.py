import os
import glob
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.io import savemat
from tqdm import tqdm
from spenpy.spen import spen


# ------------------------------ IO & Utils ------------------------------

def ensure_out_dirs(root: str | Path) -> Path:
    root = Path(root)
    (root / "hr").mkdir(parents=True, exist_ok=True)
    (root / "lr").mkdir(parents=True, exist_ok=True)
    (root / "phase_map").mkdir(parents=True, exist_ok=True)
    return root

def collect_nifti_files(*roots: Optional[str]) -> list[str]:
    """Return sorted list of all .nii and .nii.gz files under provided roots."""
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
    """
    Apply an independent random 90° CCW rotation k∈{0,1,2,3} to each slice.
    bwh: (b, w, h) float32
    """
    ks = rng.integers(0, 4, size=bwh.shape[0])
    out = [np.rot90(sl, k=int(k)) for sl, k in zip(bwh, ks)]
    return np.stack(out, axis=0)

def resize_cwh(x: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize (b, w, h) → (b, out_w, out_h) using bilinear (channel-last batching).
    """
    t = torch.from_numpy(x).float().unsqueeze(0)  # (1, b, w, h)
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
    vol = data[:, :, z0:z1]                # (X, Y, z_crop)
    bwh = np.transpose(vol, (2, 0, 1))     # (b, w, h)
    if rand_rotate:
        bwh = random_rot90_batch(bwh, rng) # (b, w, h)
    bwh = resize_cwh(bwh, out_size)        # (b, w', h')
    return bwh

def expand_phase_to_batch(phase_chunk, like_tensor: torch.Tensor):
    """
    Ensure phase map shape matches (b, w, h). Accepts torch.Tensor or np.ndarray.
    """
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


# ------------------------------ LR Mix-in ------------------------------

def mix_in_test_lr(test_lr_path: str, lr_out_dir: Path) -> None:
    """
    Copy *.mat files from 'test_lr_path' into 'lr_out_dir' (CycleGAN use case).
    Keeps filenames; overwrites on conflict.
    """
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


# ------------------------------ Main Runner ------------------------------

def run_all_dual_modality(
    IXI_T1_path: str = "/home/data1/musong/data/IXI/T1/nii",
    IXI_T2_path: str = "/home/data1/musong/data/IXI/T2",
    out_root: str = "/home/data1/musong/workspace/2025/8/08-20/tr/data/IXI_sim",
    out_size: Tuple[int, int] = (96, 96),
    percent: float = 0.3,
    rand_rotate: bool = True,          # random 0/90/180/270 per slice
    max_n: int = 1000,                 # GLOBAL max slices across all files
    max_slices_per_file: Optional[int] = None,  # NEW: per-file slice cap
    seed: int = 42,
    max_batch: int = 50,
    prefix_modality: bool = True,      # prefix filenames with T1_/T2_
    mix_test_lr: bool = False,         # copy external LR mats into lr/
    test_lr_path: Optional[str] = None # used only if mix_test_lr=True
) -> int:
    """
    Build a mixed dataset from IXI T1 & T2:
      - Randomize file order and per-file slice order.
      - Per-slice random rotation among {0, 90, 180, 270} if rand_rotate=True.
      - Save:
          hr/<stem>_idx####.mat         {"hr": (w,h) float32}
          lr/<stem>_idx####.mat         {"lr": (w,h) complex64}
          phase_map/<stem>_idx####.mat  {"phase_map": (w,h) float32}
      - Stop when 'max_n' total slices are saved.
      - Respect an optional per-file cap via 'max_slices_per_file'.
      - Optionally mix in LR-only test mats into lr/ (no hr/phase_map).
    """
    out_root = ensure_out_dirs(out_root)
    rng = np.random.default_rng(seed)

    # Collect & label modality
    t1_files = collect_nifti_files(IXI_T1_path)
    t2_files = collect_nifti_files(IXI_T2_path)
    if not (t1_files or t2_files):
        raise FileNotFoundError(f"No NIfTI files found under: {IXI_T1_path} or {IXI_T2_path}")

    files = [("T1", fp) for fp in t1_files] + [("T2", fp) for fp in t2_files]
    rng.shuffle(files)

    total_saved = 0
    with tqdm(total=max_n, desc="Saving slices", unit="slice") as pbar:
        for modality, fp in files:
            if total_saved >= max_n:
                break

            stem = maybe_prefix_modality(clean_stem(fp), modality, prefix_modality)

            # Load full volume as (b, w', h') then randomized slice order
            vol_bwh = load_nii_random_slices(
                fp, percent=percent, out_size=out_size, rng=rng, rand_rotate=rand_rotate
            )
            b = vol_bwh.shape[0]
            perm = rng.permutation(b)

            # Apply per-file cap if requested
            if max_slices_per_file is not None:
                perm = perm[:max_slices_per_file]
                b = len(perm)
                if b == 0:
                    continue

            simulator = spen(acq_point=out_size)

            with torch.no_grad():
                # Process randomized slices in batches
                for start in range(0, b, max_batch):
                    if total_saved >= max_n:
                        break
                    end = min(start + max_batch, b)
                    idxs = perm[start:end]                        # original slice indices
                    chunk_np = vol_bwh[idxs]                      # (b_chunk, w, h)
                    chunk = torch.from_numpy(chunk_np).float()    # (b_chunk, w, h)

                    # SPEN simulation
                    lr_chunk = simulator.sim(chunk)               # complex (b,w,h)
                    phase_chunk = simulator.get_phase_map(chunk)  # real (b,w,h) or (w,h)
                    phase_chunk = expand_phase_to_batch(phase_chunk, chunk)

                    # to numpy for savemat
                    hr_np = chunk.cpu().numpy().astype(np.float32)
                    lr_np = lr_chunk.detach().cpu().numpy().astype(np.complex64)
                    if torch.is_tensor(phase_chunk):
                        phase_np = phase_chunk.detach().cpu().numpy().astype(np.float32)
                    else:
                        phase_np = np.asarray(phase_chunk, dtype=np.float32)

                    # Save per-slice
                    for i in range(hr_np.shape[0]):
                        if total_saved >= max_n:
                            break
                        original_idx = int(idxs[i])  # use original z index for suffix
                        suffix = f"_idx{original_idx:04d}.mat"

                        savemat(Path(out_root) / "hr" / f"{stem}{suffix}", {"hr": hr_np[i]})
                        savemat(Path(out_root) / "lr" / f"{stem}{suffix}", {"lr": lr_np[i]})
                        savemat(Path(out_root) / "phase_map" / f"{stem}{suffix}",
                                {"phase_map": phase_np[i]})

                        total_saved += 1
                        pbar.update(1)

                        if total_saved >= max_n:
                            break

    # Optional LR mix-in
    if mix_test_lr:
        if test_lr_path:
            mix_in_test_lr(test_lr_path, Path(out_root) / "lr")
        else:
            print("mix_test_lr=True but no test_lr_path provided; skipping mix-in.")

    print(f"\nDone. Saved {total_saved} slice .mat files (limit {max_n}).")
    print(f"Outputs under: {out_root}")
    return total_saved


# ------------------------------ Example ------------------------------
if __name__ == "__main__":
    run_all_dual_modality(
        IXI_T1_path="/home/data1/musong/data/IXI/T1/nii",
        IXI_T2_path="/home/data1/musong/data/IXI/T2",
        out_root="/home/data1/musong/workspace/python/spen-recons/data/IXI_enriched_2000",
        out_size=(96, 96),
        percent=0.3,
        rand_rotate=True,            # random 0/90/180/270 per slice
        max_n=2000,                  # global cap across all files
        max_slices_per_file=4,      # <-- per-file cap; set None to disable
        seed=42,
        max_batch=50,
        prefix_modality=True,        # filenames like T1_... / T2_...
        mix_test_lr=False,           # set True to mix LR-only test mats into lr/
        test_lr_path="/home/data1/musong/workspace/python/spen-recons/test_data/lr"
    )
