#!/usr/bin/env python3

""" 
# (optional) env
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main

# paths
TEST_ROOT=/home/data1/musong/workspace/2025/8/08-20/tr/test/test_data
CKPT=output_sr/checkpoints/netG_A2B_e0199.pth    # <- pick the epoch you want
OUT=/home/data1/musong/workspace/2025/8/08-20/tr/test/test_result

CUDA_VISIBLE_DEVICES=0 python cyclegan_sr_test.py \
  --dataroot "$TEST_ROOT" \
  --lr_dir final_rxyacq_ROFFT \
  --ckpt_G_A2B "$CKPT" \
  --outdir "$OUT" \
  --batchSize 1 \
  --cuda

"""

import argparse
import glob
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------- utils (match training) ----------
def first_data_key(matdict: dict) -> str:
    for k in matdict.keys():
        if not k.startswith("__"):
            return k
    raise KeyError("No data key found in .mat file")

def _to_complex64(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x.to(torch.complex64)
    return x.to(torch.float32).to(torch.complex64)

def cplx_to_2ch(x: torch.Tensor) -> torch.Tensor:
    x = _to_complex64(x)
    return torch.stack([x.real.float(), x.imag.float()], dim=0)

def ch2_to_cplx(x2: torch.Tensor) -> torch.Tensor:
    # x2: (2,W,H) float -> (W,H) complex64
    return (x2[0].to(torch.float32) + 1j * x2[1].to(torch.float32)).to(torch.complex64)

# ---------- dataset (paired stems; we only use LR at test) ----------
class SPENDatasetTest(Dataset):
    """
    Expects:
      <root>/data/*.mat                (HR exists but isn't required for saving)
      <root>/<lr_dir>/*.mat            (LR complex)
    Returns:
      dict('A'=LR_2ch, 'id'=stem, 'B'=HR_2ch [optional use])
    """
    def __init__(self, root: str, lr_dir: str = "final_rxyacq_ROFFT"):
        self.root = root
        self.lr_dir = lr_dir
        self.hr_files = sorted(glob.glob(os.path.join(root, "data", "*.mat")))
        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No .mat files in {os.path.join(root, 'data')}")

    @staticmethod
    def _load_mat(path: str) -> np.ndarray:
        d = sio.loadmat(path)
        arr = d[first_data_key(d)]
        return np.squeeze(np.asarray(arr))

    @staticmethod
    def _np_to_torch_complex(arr: np.ndarray) -> torch.Tensor:
        if np.iscomplexobj(arr):
            return torch.from_numpy(arr).to(torch.complex64)
        return torch.from_numpy(arr.astype(np.float32)).to(torch.complex64)

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        stem = os.path.splitext(os.path.basename(hr_path))[0]

        # load LR (required)
        lr_path = os.path.join(self.root, self.lr_dir, stem + ".mat")
        if not os.path.exists(lr_path):
            raise FileNotFoundError(f"Missing LR file: {lr_path}")
        lr_np = self._load_mat(lr_path)
        lr = self._np_to_torch_complex(lr_np)  # (W,H) complex
        A = cplx_to_2ch(lr)                    # (2,W,H) float

        # HR optional (not needed for saving predictions)
        try:
            hr_np = self._load_mat(hr_path)
            hr = self._np_to_torch_complex(hr_np)
            B = cplx_to_2ch(hr)
        except Exception:
            B = None

        out = {"A": A, "id": stem}
        if B is not None:
            out["B"] = B
        return out

# ---------- models (Generator only; must match training) ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        in_f = 64
        for _ in range(2):  # down
            model += [nn.Conv2d(in_f, in_f*2, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(in_f*2),
                      nn.ReLU(inplace=True)]
            in_f *= 2
        for _ in range(n_residual_blocks):  # res
            model += [ResidualBlock(in_f)]
        for _ in range(2):  # up
            model += [nn.ConvTranspose2d(in_f, in_f//2, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(in_f//2),
                      nn.ReLU(inplace=True)]
            in_f //= 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, x): return self.model(x)

# ---------- save helpers ----------
def save_complex_mat(x2: torch.Tensor, save_path: str):
    """
    x2: (2,W,H) float tensor (on CPU); save as complex64 .mat with same shape (W,H)
    """
    x_cplx = ch2_to_cplx(x2)  # (W,H) complex64
    x_np = x_cplx.detach().cpu().numpy()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sio.savemat(save_path, {"data": x_np})

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, required=True,
                    help="test root with data/ and <lr_dir>/")
    ap.add_argument("--lr_dir", type=str, default="final_rxyacq_ROFFT",
                    help="LR folder name under dataroot")
    ap.add_argument("--ckpt_G_A2B", type=str, required=True,
                    help="path to netG_A2B checkpoint (.pth)")
    ap.add_argument("--outdir", type=str, required=True,
                    help="directory to save predicted HR .mat files")
    ap.add_argument("--batchSize", type=int, default=1)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--n_cpu", type=int, default=4)
    ap.add_argument("--n_residual_blocks", type=int, default=9,
                    help="must match training")
    args = ap.parse_args()
    print(args)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    input_nc = output_nc = 2  # complex as 2ch

    # model
    netG_A2B = Generator(input_nc, output_nc, n_residual_blocks=args.n_residual_blocks).to(device)
    sd = torch.load(args.ckpt_G_A2B, map_location=device)
    netG_A2B.load_state_dict(sd)
    netG_A2B.eval()

    # data
    ds = SPENDatasetTest(args.dataroot, lr_dir=args.lr_dir)
    dl = DataLoader(ds, batch_size=args.batchSize, shuffle=False,
                    num_workers=args.n_cpu, pin_memory=True, drop_last=False)

    os.makedirs(args.outdir, exist_ok=True)

    with torch.no_grad():
        for batch in dl:
            A = batch["A"].to(device)  # (B,2,W,H) float
            stems = batch["id"]

            fake_B = netG_A2B(A)       # (B,2,W,H)

            # save each sample
            for b in range(fake_B.size(0)):
                pred_2ch = fake_B[b].detach().cpu()  # (2,W,H)
                save_path = os.path.join(args.outdir, f"{stems[b]}.mat")
                save_complex_mat(pred_2ch, save_path)

    print(f"[+] Done. Saved predictions to: {args.outdir}")

if __name__ == "__main__":
    main()
