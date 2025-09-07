#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from PIL import Image

# ---------- your function ----------
def mat_to_img01(path: str) -> np.ndarray:
    """Load .mat, pick first non-meta key, magnitude if complex, min-max to [0,1], return HxW float32."""
    md = loadmat(path)
    arr = None
    for k, v in md.items():
        if not k.startswith("__"):
            arr = v
            break
    if arr is None:
        raise KeyError(f"No data key found in {path} (only __meta keys).")

    x = np.asarray(arr).squeeze()
    if x.ndim > 2:
        x = x[..., 0]
    if np.iscomplexobj(x):
        x = np.abs(x)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax > xmin:
        x = (x - xmin) / (xmax - xmin)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x
# -----------------------------------

def save_img01_to_png(img01: np.ndarray, out_path: Path) -> None:
    """Save [0,1] float array as 8-bit grayscale PNG."""
    arr8 = (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr8, mode="L").save(out_path)

def convert_tree(src_root: Path, dst_root: Path, ext=".png") -> None:
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    total = ok = skipped = 0
    for mat_path in src_root.rglob("*.mat"):
        total += 1
        rel = mat_path.relative_to(src_root)                       # e.g. hr/IXI...mat
        out_path = dst_root / rel.parent / (rel.stem + ext)        # keep hr/lr/phase_map

        try:
            img01 = mat_to_img01(str(mat_path))
            save_img01_to_png(img01, out_path)
            ok += 1
            if ok % 50 == 0:
                print(f"[{ok}/{total}] {rel} -> {out_path.relative_to(dst_root)}")
        except Exception as e:
            skipped += 1
            print(f"[WARN] Skip {rel}: {e}")

    print(f"\nDone. total={total}, saved={ok}, skipped={skipped}")
    print(f"Output root: {dst_root}")

def main():
    parser = argparse.ArgumentParser(description="Export .mat images to PNG while preserving folder structure.")
    parser.add_argument("--src_root", type=Path,
        default="/home/data1/musong/workspace/python/spen-recons/test_data_2025_9_7/hr",
        help="Source root containing hr/lr/phase_map subfolders.")

    parser.add_argument("--out_root", type=Path,
        default="/home/data1/musong/workspace/python/spen-recons/temp/img/hr",
        help="Destination root (e.g., a log folder).")
    parser.add_argument("--ext", type=str, default=".png", help="Image extension (.png, .jpg). Default: .png")
    args = parser.parse_args()

    convert_tree(args.src_root, args.out_root, ext=args.ext)

if __name__ == "__main__":
    main()
