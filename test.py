#!/usr/bin/env python3
"""
Quick sanity test for:
- SPENDataset (gt/lr complex, phase_map real float32 (W/2,H))
- physical_model forward/recons without resizing phase
- one forward pass through G_phi (2->1) and G_rec (2->2)
- compute L_phase / L_img / L_dc once
- SAVE predicted phase map and reconstructed image to .mat

Run:
  python test.py \
    --dataroot /home/data1/musong/workspace/2025/8/08-20/tr/test \
    --outdir test/test_result \
    --cuda
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as sio

from train import SPENDataset, physical_model, ResNetRefiner  # adjust path if needed


# ---------- small utils ----------
def cplx_to_2ch(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.real, x.imag], dim=1)

def ch2_to_cplx(x2: torch.Tensor) -> torch.Tensor:
    return x2[:, 0] + 1j * x2[:, 1]


@torch.no_grad()
def check_batch_shapes(batch):
    gt = batch["gt"]
    lr = batch["lr"]
    ph = batch["phase_map"]

    assert gt.ndim == 2 and lr.ndim == 2 and ph.ndim == 2, "Expect (W,H) for gt/lr and (W/2,H) for phase_map"
    W, H = gt.shape
    Wlr, Hlr = lr.shape
    assert (W, H) == (Wlr, Hlr), "gt and lr must have same shape"
    assert ph.shape == (W // 2, H), f"phase_map must be (W/2,H); got {ph.shape}, expected {(W//2, H)}"
    assert torch.is_complex(gt) and torch.is_complex(lr), "gt/lr must be complex"
    assert (not torch.is_complex(ph)) and ph.dtype == torch.float32, "phase_map must be real float32"
    return W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True, type=str)
    ap.add_argument("--lr_dir", default="final_rxyacq_ROFFT", type=str)
    ap.add_argument("--batch_size", default=2, type=int)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--n_blocks", default=2, type=int)
    ap.add_argument("--base_ch", default=64, type=int)
    ap.add_argument("--outdir", default="results_test", type=str)
    args = ap.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    os.makedirs(args.outdir, exist_ok=True)

    # ---- dataset / loader ----
    ds = SPENDataset(args.dataroot, lr_dir=args.lr_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ---- take one batch ----
    batch = next(iter(dl))
    names = batch["id"]  # keep sample IDs

    gt  = torch.stack([x.to(torch.complex64) for x in batch["gt"]], dim=0).to(device)
    lr  = torch.stack([x.to(torch.complex64) for x in batch["lr"]], dim=0).to(device)
    ph  = torch.stack([x.to(torch.float32)   for x in batch["phase_map"]], dim=0).to(device)

    W, H = gt.shape[-2], gt.shape[-1]

    print("== dataset sanity ==")
    print(f"B={gt.shape[0]}, W={W}, H={H}")
    print(f"gt dtype={gt.dtype}, lr dtype={lr.dtype}, phase dtype={ph.dtype}")
    print(f"phase_map shape = {ph.shape} (expected B x W/2 x H)")

    # ---- physical model ----
    PM = physical_model(img_size=(W, H))
    PM.AFinal = PM.AFinal.to(device)
    PM.InvA   = PM.InvA.to(device)

    # forward acquisition from gt
    y_syn = PM(gt, phase_map=ph)
    # recon from measured lr
    x0 = PM.recons(lr, phase_map=ph)

    # quick MSE
    mse = nn.MSELoss(reduction="mean")
    dc_meas = mse(cplx_to_2ch(y_syn).float(), cplx_to_2ch(lr).float()).item()
    print(f"PM MSE(y_syn vs lr) = {dc_meas:.6f}")

    # ---- generators ----
    G_phi = ResNetRefiner(in_ch=2, out_ch=1, n_blocks=args.n_blocks, base=args.base_ch).to(device)
    G_rec = ResNetRefiner(in_ch=2, out_ch=2, n_blocks=args.n_blocks, base=args.base_ch).to(device)

    # predict phase
    lr_2ch = cplx_to_2ch(lr).float()
    phase_full = G_phi(lr_2ch)                        # (B,1,W,H)
    phase_pred = phase_full[:, :, 1::2, :].squeeze(1) # (B,W/2,H)

    # recon with predicted phase
    x0_pred = PM.recons(lr, phase_map=phase_pred)

    # refinement
    x_rec_2ch = G_rec(cplx_to_2ch(x0_pred).float())
    x_rec = ch2_to_cplx(x_rec_2ch)

    # ---- losses ----
    l1 = nn.L1Loss()
    L_phase = l1(phase_pred, ph)
    y_pred  = PM(x_rec, phase_map=phase_pred)
    L_dc    = mse(cplx_to_2ch(y_pred).float(), cplx_to_2ch(lr).float())
    L_img   = l1(cplx_to_2ch(x_rec).float(), cplx_to_2ch(gt).float())

    print("== forward & losses ==")
    print(f"L_phase={L_phase.item():.6f}, L_dc={L_dc.item():.6f}, L_img={L_img.item():.6f}")

    for i, name in enumerate(names):
        out_phase = phase_pred[i].detach().cpu().numpy().astype(np.float32)  # (W/2,H)
        out_rec   = x_rec[i].detach().cpu().numpy()                          # (W,H) complex

        phase_path = os.path.join(args.outdir, f"{name}_phase_pred.mat")
        sio.savemat(phase_path, {"phase_pred": out_phase})

        rec_path = os.path.join(args.outdir, f"{name}_rec.mat")
        sio.savemat(rec_path, {"rec": out_rec})

        print(f"Saved {phase_path}")
        print(f"Saved {rec_path}")

    print("✓ test passed & results saved.")


if __name__ == "__main__":
    main()
