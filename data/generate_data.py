import os
from pathlib import Path
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.io import savemat
from tqdm import tqdm

from sys import path
path.append("/home/data1/musong/workspace/2025/8/08-20/spenpy")
from spenpy.spen import spen


def load_nii_slices(file_path, percent=0.3, rotate=True):
    img = nib.load(file_path)
    data = img.get_fdata()
    data = np.asarray(data, dtype=np.float32)

    # Crop along z
    z_start = int(percent * data.shape[2])
    z_end   = int((1 - percent) * data.shape[2])
    cropped = data[:, :, z_start:z_end]   # (X, Y, b)

    # Reorder to (b, w, h)
    cropped = np.transpose(cropped, (2, 0, 1))

    # Rotate each slice 90° CCW if requested
    if rotate:
        cropped = np.array([np.rot90(slice_, k=1) for slice_ in cropped])

    return cropped  # (b, w, h) float32


def resize_cwh(x, out_size):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.float()          # (b, w, h)
    x = x.unsqueeze(0)     # (1, b, w, h) -> treat b as channels
    x_resized = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
    return x_resized.squeeze(0)  # (b, w', h')


def ensure_out_dirs(root: str):
    root = Path(root)
    (root / "hr").mkdir(parents=True, exist_ok=True)
    (root / "lr").mkdir(parents=True, exist_ok=True)
    (root / "phase_map").mkdir(parents=True, exist_ok=True)
    return root


def clean_stem(path_str: str) -> str:
    name = Path(path_str).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(path_str).stem


def run_and_save_single_file(
    nii_path: str,
    out_root: Path,
    out_size=(96, 96),
    percent=0.3,
    rotate=True,
    max_batch: int = 50
):
    """
    Process one NIfTI and save per-slice .mat files into:
      data/<stem>_idx####.mat               (float32)
      final_rxyacq_ROFFT/<stem>_idx####.mat (complex64)
      phase_map/<stem>_idx####.mat          (float32)
    Batch size for SPEN is capped by `max_batch` to avoid OOM.
    (No tqdm here by request.)
    """
    stem = clean_stem(nii_path)

    # Load & resize the whole volume once (then chunk by slices)
    data_np = load_nii_slices(nii_path, percent=percent, rotate=rotate)       # (b,w,h)
    data_t  = resize_cwh(data_np, out_size).contiguous()                       # (b,w',h')
    B = data_t.shape[0]

    sim = spen(acq_point=out_size)

    with torch.no_grad():
        for start in range(0, B, max_batch):
            end = min(start + max_batch, B)
            chunk = data_t[start:end]               # (b_chunk, w', h')

            final_chunk = sim.sim(chunk)            # complex (b_chunk, w', h')
            phase_chunk = sim.get_phase_map(chunk)  # real (b_chunk, w', h') or (w', h')

            # Normalize shapes for saving
            if torch.is_tensor(phase_chunk):
                if phase_chunk.ndim == 2:
                    phase_chunk = phase_chunk.unsqueeze(0).expand_as(chunk)
            else:
                phase_chunk = np.asarray(phase_chunk)
                if phase_chunk.ndim == 2:
                    phase_chunk = np.repeat(phase_chunk[None, ...], chunk.shape[0], axis=0)

            # Move to numpy
            data_np_out  = chunk.cpu().numpy().astype(np.float32)
            final_np_out = final_chunk.detach().cpu().numpy().astype(np.complex64)
            if torch.is_tensor(phase_chunk):
                phase_np_out = phase_chunk.detach().cpu().numpy().astype(np.float32)
            else:
                phase_np_out = phase_chunk.astype(np.float32)

            # Save per-slice
            for i in range(end - start):
                idx = start + i  # global slice index within this file
                suffix = f"_idx{idx:04d}.mat"
                savemat(out_root / "hr" / f"{stem}{suffix}", {"data": data_np_out[i]})
                savemat(out_root / "lr" / f"{stem}{suffix}",
                        {"final_rxyacq_ROFFT": final_np_out[i]})
                savemat(out_root / "phase_map" / f"{stem}{suffix}",
                        {"phase_map": phase_np_out[i]})

    return B  # number of slices processed for this file


def run_all(
    IXI_path: str = "/home/data1/musong/data/IXI/T1/nii",
    out_root: str = "/home/data1/musong/workspace/2025/8/08-20/tr/data/IXI_sim",
    out_size=(96, 96),
    percent=0.3,
    rotate=True,
    max_n: int = 1000,   # total number of slice .mat files to save
    seed: int = 42,
    max_batch: int = 50
):
    """
    Process NIfTI files under IXI_path and save per-slice .mat files.
    Stops once `max_n` slice .mat files are saved in total.
    Progress bar is tied to `max_n`.
    """
    out_root = ensure_out_dirs(out_root)

    # Collect nii & nii.gz
    nii_files = sorted(glob.glob(os.path.join(IXI_path, "*.nii"))) + \
                sorted(glob.glob(os.path.join(IXI_path, "*.nii.gz")))
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI files found under: {IXI_path}")

    rng = np.random.default_rng(seed)
    rng.shuffle(nii_files)

    total_saved = 0
    with tqdm(total=max_n, desc="Saving slices", unit="slice") as pbar:
        for fp in nii_files:
            if total_saved >= max_n:
                break

            stem = clean_stem(fp)
            data_np = load_nii_slices(fp, percent=percent, rotate=rotate)
            data_t  = resize_cwh(data_np, out_size).contiguous()
            B = data_t.shape[0]

            sim = spen(acq_point=out_size)

            with torch.no_grad():
                for start in range(0, B, max_batch):
                    if total_saved >= max_n:
                        break
                    end = min(start + max_batch, B)
                    chunk = data_t[start:end]

                    final_chunk = sim.sim(chunk)
                    phase_chunk = sim.get_phase_map(chunk)
                    if torch.is_tensor(phase_chunk):
                        if phase_chunk.ndim == 2:
                            phase_chunk = phase_chunk.unsqueeze(0).expand_as(chunk)
                    else:
                        phase_chunk = np.asarray(phase_chunk)
                        if phase_chunk.ndim == 2:
                            phase_chunk = np.repeat(phase_chunk[None, ...], chunk.shape[0], axis=0)

                    data_np_out  = chunk.cpu().numpy().astype(np.float32)
                    final_np_out = final_chunk.detach().cpu().numpy().astype(np.complex64)
                    if torch.is_tensor(phase_chunk):
                        phase_np_out = phase_chunk.detach().cpu().numpy().astype(np.float32)
                    else:
                        phase_np_out = phase_chunk.astype(np.float32)

                    for i in range(end - start):
                        if total_saved >= max_n:
                            break
                        idx = start + i
                        suffix = f"_idx{idx:04d}.mat"

                        savemat(Path(out_root) / "hr" / f"{stem}{suffix}", {"hr": data_np_out[i]})
                        savemat(Path(out_root) / "lr" / f"{stem}{suffix}",
                                {"lr": final_np_out[i]})
                        savemat(Path(out_root) / "phase_map" / f"{stem}{suffix}",
                                {"phase_map": phase_np_out[i]})

                        total_saved += 1
                        pbar.update(1)   # update bar for each saved slice

    print(f"\nDone. Saved {total_saved} slice .mat files (limit {max_n}).")
    print(f"Outputs under: {out_root}")
    return total_saved

# ---------- Example ----------
run_all(
    IXI_path="/home/data1/musong/data/IXI/T1/nii",
    out_root="/home/data1/musong/workspace/2025/8/08-20/tr/data/IXI_sim",
    out_size=(96, 96),
    percent=0.3,
    rotate=True,
    max_n=1000,   
    seed=42,
    max_batch=50,  # cap SPEN batch size to avoid OOM
)
