import argparse
import os
from glob import glob
import csv

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from spenpy.spen import spen


class physical_model:
    def __init__(self, img_size=(96, 96)):
        self.InvA, self.AFinal = spen(acq_point=img_size).get_InvA()

    def __call__(self, x: torch.Tensor, phase_map: torch.Tensor | None = None) -> torch.Tensor:
        # Forward acquisition: A * x, then optional phase on odd lines
        y = torch.matmul(self.AFinal.to(x.device) * 1j, x)
        if phase_map is not None:
            y = y.clone()
            y[:, 1::2, :] *= torch.exp(1j * phase_map)
        return y

    def recons(self, y: torch.Tensor, phase_map: torch.Tensor | None = None) -> torch.Tensor:
        # Inverse: phase correction then A^{-1}
        if phase_map is not None:
            y = y.clone()
            y[:, 1::2, :] *= torch.exp(-1j * phase_map)
        return torch.matmul(self.InvA.to(y.device), y)


# ----------------------------
# Utils: complex <-> 2ch real
# ----------------------------
def _to_complex64(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x.to(torch.complex64)
    return x.to(torch.float32).to(torch.complex64)

def _to_float32(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        x = x.real
    return x.to(torch.float32)

def cplx_to_2ch(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x.real, x.imag], dim=1)

def ch2_to_cplx(x2: torch.Tensor) -> torch.Tensor:
    return x2[:, 0] + 1j * x2[:, 1]

def first_data_key(matdict: dict) -> str:
    for k in matdict.keys():
        if not k.startswith("__"):
            return k
    raise KeyError("No data key found in .mat file")


# ----------------------------
# Dataset
# ----------------------------
class SPENDataset(Dataset):
    """
    For each <id>.mat it loads:
      gt    : <root>/data/<id>.mat
      y_file: <root>/final_rxyacq_ROFFT/<id>.mat
      phase : <root>/phase_map/<id>.mat
    """

    def __init__(self, root: str, lr_dir: str = "final_rxyacq_ROFFT", phase_dir: str = "phase_map"):
        self.root = root
        self.lr_dir = lr_dir
        self.phase_map_dir = phase_dir

        self.data_files = sorted(glob(os.path.join(root, "data", "*.mat")))
        if len(self.data_files) == 0:
            raise FileNotFoundError(f"No .mat files in {os.path.join(root, 'data')}")

    def __len__(self):
        return len(self.data_files)

    def _load_mat(self, path: str) -> np.ndarray:
        d = sio.loadmat(path)
        arr = d[first_data_key(d)]
        return np.squeeze(np.asarray(arr))

    def _np_to_torch_complex(self, arr: np.ndarray) -> torch.Tensor:
        if np.iscomplexobj(arr):
            return torch.from_numpy(arr).to(torch.complex64)
        return torch.from_numpy(arr.astype(np.float32)).to(torch.complex64)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        stem = os.path.splitext(os.path.basename(data_path))[0]
        
        gt_np = self._load_mat(data_path)
        gt = self._np_to_torch_complex(gt_np)

        ph_path = os.path.join(self.root, self.phase_map_dir, stem + ".mat")
        if not os.path.exists(ph_path):
            raise FileNotFoundError(f"Missing phase map file: {ph_path}")
        ph_np = self._load_mat(ph_path)
        phase_map = torch.from_numpy(ph_np)

        lr_path = os.path.join(self.root, self.lr_dir, stem + ".mat")
        if not os.path.exists(lr_path):
            raise FileNotFoundError(f"Missing LR file: {lr_path}")
        lr_np = self._load_mat(lr_path)
        lr = self._np_to_torch_complex(lr_np)

        return {"id": stem, "gt": gt, "lr": lr, "phase_map": phase_map}


# ----------------------------
# Networks
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )
    def forward(self, x):
        return x + self.block(x)

class ResNetRefiner(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, n_blocks=6, base=64):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, base, 7, bias=False),
            nn.InstanceNorm2d(base, affine=True),
            nn.ReLU(inplace=True),
        ]
        ch = base
        # Down
        for _ in range(2):
            layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            ch *= 2
        # Residuals
        for _ in range(n_blocks):
            layers += [ResidualBlock(ch)]
        # Up
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ch // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, out_ch, 7)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        def C(i, o, ks=4, s=2, p=1, norm=True):
            L = [nn.Conv2d(i, o, ks, s, p)]
            if norm:
                L += [nn.InstanceNorm2d(o)]
            L += [nn.LeakyReLU(0.2, inplace=True)]
            return L
        L = []
        L += C(in_ch, 64, norm=False)
        L += C(64, 128)
        L += C(128, 256)
        L += [nn.Conv2d(256, 512, 4, stride=1, padding=1),
              nn.InstanceNorm2d(512),
              nn.LeakyReLU(0.2, inplace=True)]
        L += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        return self.net(x)


# ----------------------------
# Training
# ----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    ds = SPENDataset(args.dataroot, lr_dir=args.lr_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.n_cpu, drop_last=True)

    # Infer size for PM
    W, H = ds[0]["gt"].shape[-2:]
    if (W, H) != (args.size, args.size):
        print(f"[!] Detected ({W},{H}) != --size ({args.size},{args.size}). Using detected size.")
        args.size = W
    PM = physical_model(img_size=(args.size, args.size))

    # Generators
    G_phi = ResNetRefiner(in_ch=2, out_ch=1, n_blocks=args.n_blocks, base=args.base_ch).to(device)  # phase net
    G_rec = ResNetRefiner(in_ch=2, out_ch=2, n_blocks=args.n_blocks, base=args.base_ch).to(device)  # recon net

    # Discriminators
    D_phi = PatchDiscriminator(in_ch=1).to(device) if args.use_gan else None
    D_img = PatchDiscriminator(in_ch=2).to(device) if args.use_gan else None

    # Losses / Optims
    l1, mse = nn.L1Loss(), nn.MSELoss()
    optG = torch.optim.Adam(list(G_phi.parameters()) + list(G_rec.parameters()),
                            lr=args.lr, betas=(0.5, 0.999))
    if args.use_gan:
        optD_phi = torch.optim.Adam(D_phi.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optD_img = torch.optim.Adam(D_img.parameters(), lr=args.lr, betas=(0.5, 0.999))

    schG = torch.optim.lr_scheduler.LambdaLR(
        optG, lr_lambda=lambda e: 1.0 - max(0, e + 1 - args.decay_epoch) / max(1, (args.n_epochs - args.decay_epoch))
    )
    if args.use_gan:
        schD_phi = torch.optim.lr_scheduler.LambdaLR(
            optD_phi, lr_lambda=lambda e: 1.0 - max(0, e + 1 - args.decay_epoch) / max(1, (args.n_epochs - args.decay_epoch))
        )
        schD_img = torch.optim.lr_scheduler.LambdaLR(
            optD_img, lr_lambda=lambda e: 1.0 - max(0, e + 1 - args.decay_epoch) / max(1, (args.n_epochs - args.decay_epoch))
        )

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "iter", "L_phase", "L_img", "L_dc", "G_tot"]
        if args.use_gan: header += ["G_gan_phi", "G_gan_img", "D_phi", "D_img"]
        writer.writerow(header)

    for epoch in range(1, args.n_epochs + 1):
        for it, batch in enumerate(loader, 1):
            # --- load batch ---
            hr  = torch.stack([_to_complex64(x) for x in batch["gt"]], dim=0).to(device)          # (B,W,H)
            lr_file = torch.stack([_to_complex64(x) for x in batch["lr"]], dim=0).to(device)       # (B,W,H)
            phi_gt = torch.stack([_to_float32(x) for x in batch["phase_map"]], dim=0).to(device)   # (B,W/2,H)

            # --- make physics-model LR (first branch) ---
            # This is the synthetic LR from the *ground-truth* HR and *ground-truth* phase.
            # If you instead want to use predicted phase for this branch, swap phi_gt -> phi_pred later.
            with torch.no_grad():
                lr_pm = PM(hr, phase_map=phi_gt)   # (B,W,H)

            # ---- Phase prediction (no resize; take odd rows only) ----
            lr_2ch = cplx_to_2ch(lr_file).float()             # (B,2,W,H)
            phase_full = G_phi(lr_2ch)                         # (B,1,W,H)
            phi_pred = phase_full[:, :, 1::2, :].squeeze(1)    # (B,W/2,H)  <-- only odd lines, no resize

            # ---- Physics recon with predicted phase (use measured LR for realism) ----
            with torch.no_grad():
                x0 = PM.recons(lr_file, phase_map=phi_pred)    # (B,W,H) complex

            # ---- Recon refinement ----
            x_rec_2ch = G_rec(cplx_to_2ch(x0).float())         # (B,2,W,H)
            x_rec = ch2_to_cplx(x_rec_2ch)                     # (B,W,H) complex

            # ---- Losses (Generators) ----
            # 1) Phase supervision (pred vs. GT phase)
            L_phase = l1(phi_pred, phi_gt)

            # 2) Data consistency to BOTH LR sources
            #    Forward project the reconstructed image with predicted phase
            y_pred = PM(x_rec, phase_map=phi_pred)             # (B,W,H)
            dc_file = mse(cplx_to_2ch(y_pred).float(), cplx_to_2ch(lr_file).float())
            dc_pm   = mse(cplx_to_2ch(y_pred).float(), cplx_to_2ch(lr_pm).float())
            L_dc    = args.dc_file_w * dc_file + args.dc_pm_w * dc_pm

            # 3) Image loss to HR
            L_img   = l1(cplx_to_2ch(x_rec).float(), cplx_to_2ch(hr).float())

            # (optional) GAN terms
            G_gan_phi = torch.tensor(0.0, device=device)
            G_gan_img = torch.tensor(0.0, device=device)
            if args.use_gan:
                pred_phi_fake = D_phi(phi_pred.unsqueeze(1))
                G_gan_phi = mse(pred_phi_fake, torch.ones_like(pred_phi_fake, device=device))

                pred_img_fake = D_img(x_rec_2ch.detach() * 0 + x_rec_2ch)
                G_gan_img = mse(pred_img_fake, torch.ones_like(pred_img_fake, device=device))

            G_total = (args.l_phase * L_phase +
                       args.l_img * L_img +
                       args.l_dc * L_dc +
                       (args.l_gan_phi * G_gan_phi if args.use_gan else 0.0) +
                       (args.l_gan_img * G_gan_img if args.use_gan else 0.0))

            optG.zero_grad(set_to_none=True)
            G_total.backward()
            optG.step()

            # ---- Discriminators ----
            if args.use_gan:
                # D_phi
                optD_phi.zero_grad(set_to_none=True)
                dphi_real = mse(D_phi(phi_gt.unsqueeze(1)), torch.ones_like(pred_phi_fake, device=device))
                dphi_fake = mse(D_phi(phi_pred.detach().unsqueeze(1)), torch.zeros_like(pred_phi_fake, device=device))
                Dphi_total = 0.5 * (dphi_real + dphi_fake)
                Dphi_total.backward()
                optD_phi.step()

                # D_img
                optD_img.zero_grad(set_to_none=True)
                dimg_real = mse(D_img(cplx_to_2ch(hr).float()), torch.ones_like(pred_img_fake, device=device))
                dimg_fake = mse(D_img(x_rec_2ch.detach()), torch.zeros_like(pred_img_fake, device=device))
                Dimg_total = 0.5 * (dimg_real + dimg_fake)
                Dimg_total.backward()
                optD_img.step()

            # ---- Log ----
            if it % args.log_every == 0:
                if args.use_gan:
                    print(f"[{epoch:03d}/{args.n_epochs}][{it:04d}/{len(loader)}] "
                          f"L_phase={L_phase.item():.4f}  L_img={L_img.item():.4f}  "
                          f"L_dc={L_dc.item():.4f} (dc_file={dc_file.item():.4f}, dc_pm={dc_pm.item():.4f})  "
                          f"G_tot={G_total.item():.4f}  G_gan_phi={G_gan_phi.item():.4f}  G_gan_img={G_gan_img.item():.4f}  "
                          f"D_phi={Dphi_total.item():.4f}  D_img={Dimg_total.item():.4f}")
                else:
                    print(f"[{epoch:03d}/{args.n_epochs}][{it:04d}/{len(loader)}] "
                          f"L_phase={L_phase.item():.4f}  L_img={L_img.item():.4f}  "
                          f"L_dc={L_dc.item():.4f} (dc_file={dc_file.item():.4f}, dc_pm={dc_pm.item():.4f})  "
                          f"G_tot={G_total.item():.4f}")

                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    row = [epoch, it, L_phase.item(), L_img.item(), L_dc.item(), G_total.item()]
                    if args.use_gan:
                        row += [G_gan_phi.item(), G_gan_img.item(), Dphi_total.item(), Dimg_total.item()]
                    writer.writerow(row)

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", type=str, required=True, help="Root with data/, phase_map/, <lr_dir>/")
    p.add_argument("--lr_dir", type=str, default="final_rxyacq_ROFFT")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--decay_epoch", type=int, default=100)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--n_cpu", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--outdir", type=str, default="log_dir")
    p.add_argument("--save_freq", type=int, default=50)
    p.add_argument("--use_gan", action="store_true")
    p.add_argument("--l_phase", type=float, default=1.0)
    p.add_argument("--l_img", type=float, default=1.0)
    p.add_argument("--l_dc", type=float, default=10.0)
    p.add_argument("--l_gan_phi", type=float, default=0.2)
    p.add_argument("--l_gan_img", type=float, default=0.5)
    p.add_argument("--dc_file_w", type=float, default=1.0, help="Weight for DC to file LR")
    p.add_argument("--dc_pm_w",   type=float, default=0.5, help="Weight for DC to PM LR")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()