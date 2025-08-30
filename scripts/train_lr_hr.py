#!/usr/bin/env python3
"""
# env (optional)
cd /home/data1/musong/workspace/2025/8/08-20/tr
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main

CUDA_VISIBLE_DEVICES=0 python /home/data1/musong/workspace/2025/8/08-20/tr/scripts/train_lr_hr.py \
  --dataroot /home/data1/musong/workspace/2025/8/08-20/tr/data/IXI_sim \
  --lr_dir final_rxyacq_ROFFT \
  --batchSize 4 \
  --n_epochs 200 --decay_epoch 100 \
  --lr 2e-4 \
  --cuda \
  --outdir /home/data1/musong/workspace/2025/8/08-20/tr/train \
  --log_freq 200 \
  --ckpt_freq 50
"""

import argparse, os, sys, time, glob, random, datetime, itertools
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # only used by logger for PNG saving

# ----------------------------
# Complex helpers
# ----------------------------
def first_data_key(matdict: dict) -> str:
    for k in matdict.keys():
        if not k.startswith("__"): return k
    raise KeyError("No data key found")

def _to_complex64(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x): return x.to(torch.complex64)
    return x.to(torch.float32).to(torch.complex64)

def cplx_to_2ch(x: torch.Tensor) -> torch.Tensor:
    x = _to_complex64(x)
    return torch.stack([x.real.float(), x.imag.float()], dim=0)  # (2,H,W)

def tensor_to_vis01(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W)
    - if C==2 -> magnitude, robust 0..1
    - if C==1 -> clamp to 0..1
    returns (B,1,H,W) in 0..1
    """
    eps = 1e-8
    if x.size(1) == 2:
        mag = torch.sqrt(x[:,0]**2 + x[:,1]**2)
        q = torch.quantile(mag.view(mag.size(0), -1), 0.999, dim=1, keepdim=True).clamp_min(eps)
        mag = (mag / q.view(-1,1,1)).clamp(0,1)
        return mag.unsqueeze(1)
    else:
        return x.clamp(0,1)

# ----------------------------
# Unpaired dataset  (CHANGED)
# ----------------------------
class SPENUnpairedDataset(Dataset):
    """
    Expects:
      <root>/data/*.mat              (HR domain B)  -> 1 channel (float), normalized to [0,1]
      <root>/<lr_dir>/*.mat          (LR domain A)  -> 2 channels (complex split), |A| max to 1
    Returns:
      'A' = random LR (2ch), 'B' = random HR (1ch), unpaired.
    """
    def __init__(self, root: str, lr_dir: str = "final_rxyacq_ROFFT", augment: bool = False):  # CHANGED: augment default False
        self.root = root
        self.lr_dir = lr_dir
        self.augment = augment  # we will not actually flip (see below)

        self.hr_files = sorted(glob.glob(os.path.join(root, "data", "*.mat")))
        self.lr_files = sorted(glob.glob(os.path.join(root, lr_dir, "*.mat")))
        if len(self.hr_files) == 0: raise FileNotFoundError(f"No HR .mat in {os.path.join(root,'data')}")
        if len(self.lr_files) == 0: raise FileNotFoundError(f"No LR .mat in {os.path.join(root,lr_dir)}")
        self.rng = random.Random(1234)

    @staticmethod
    def _load_mat(path: str) -> np.ndarray:
        d = sio.loadmat(path)
        return np.squeeze(np.asarray(d[first_data_key(d)]))

    @staticmethod
    def _np_to_torch_complex(arr: np.ndarray) -> torch.Tensor:
        if np.iscomplexobj(arr): return torch.from_numpy(arr).to(torch.complex64)
        return torch.from_numpy(arr.astype(np.float32)).to(torch.complex64)

    @staticmethod
    def _np_to_torch_real(arr: np.ndarray) -> torch.Tensor:
        # Accept real or complex; convert to float32 real-valued 1ch
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        else:
            arr = arr.astype(np.float32)
        return torch.from_numpy(arr).float()  # (H,W)

    def __len__(self):
        return max(len(self.lr_files), len(self.hr_files))

    def __getitem__(self, idx):
        # sample independently (unpaired)
        lr_path = self.lr_files[self.rng.randrange(len(self.lr_files))]
        hr_path = self.hr_files[self.rng.randrange(len(self.hr_files))]

        # LR -> complex (2ch), then scale by max |.| to 1  (CHANGED)
        lr = self._np_to_torch_complex(self._load_mat(lr_path))  # (H,W) complex
        A = cplx_to_2ch(lr)  # (2,H,W)
        eps = 1e-8
        mag = torch.sqrt(A[0]**2 + A[1]**2)
        s = torch.max(mag).clamp_min(eps)
        A = A / s  # now max magnitude == 1

        # HR -> 1ch float, min-max to [0,1]  (CHANGED)
        hr = self._np_to_torch_real(self._load_mat(hr_path))  # (H,W) float
        hmin, hmax = torch.min(hr), torch.max(hr)
        hr = (hr - hmin) / (hmax - hmin + 1e-8)
        B = hr.unsqueeze(0)  # (1,H,W)

        # NO FLIPS (CHANGED): user requested no augmentation
        idA = os.path.splitext(os.path.basename(lr_path))[0]
        idB = os.path.splitext(os.path.basename(hr_path))[0]
        return {"A": A, "B": B, "idA": idA, "idB": idB}

# ----------------------------
# Models (same as before, but I/O channels CHANGED)
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )
    def forward(self, x): return x + self.conv(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()
        m = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc,64,7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        ch = 64
        for _ in range(2):
            m += [nn.Conv2d(ch, ch*2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch*2), nn.ReLU(inplace=True)]
            ch *= 2
        for _ in range(n_residual_blocks): m += [ResidualBlock(ch)]
        for _ in range(2):
            m += [nn.ConvTranspose2d(ch, ch//2, 3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(ch//2), nn.ReLU(inplace=True)]
            ch //= 2
        m += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.net = nn.Sequential(*m)
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        m = [
            nn.Conv2d(input_nc,64,4,stride=2,padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,128,4,stride=2,padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,256,4,stride=2,padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256,512,4,padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512,1,4,padding=1),
        ]
        self.net = nn.Sequential(*m)
    def forward(self, x):
        x = self.net(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)

# ----------------------------
# Training utils
# ----------------------------
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0
        self.max_size = max_size
        self.data = []
    def push_and_pop(self, data):
        out = []
        for el in data.detach():
            el = el.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(el); out.append(el)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    out.append(self.data[i].clone()); self.data[i] = el
                else:
                    out.append(el)
        return torch.cat(out, dim=0)

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0
        self.n_epochs, self.offset, self.decay_start_epoch = n_epochs, offset, decay_start_epoch
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    n = m.__class__.__name__
    if 'Conv' in n:
        if hasattr(m, 'weight') and m.weight is not None: nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
    if 'InstanceNorm2d' in n or 'BatchNorm2d' in n:
        if hasattr(m, 'weight') and m.weight is not None: nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)

class Logger:
    def __init__(self, n_epochs, batches_epoch, log_dir='logs', log_freq=100):
        self.n_epochs, self.batches_epoch = n_epochs, batches_epoch
        self.epoch, self.batch = 1, 1
        self.prev_time, self.mean_period = time.time(), 0.0
        self.acc = {}
        self.log_freq = log_freq
        os.makedirs(log_dir, exist_ok=True)
        self.csv = os.path.join(log_dir, 'losses.csv')
        self.img_dir = os.path.join(log_dir, 'images'); os.makedirs(self.img_dir, exist_ok=True)
        with open(self.csv, 'w') as f:
            f.write('epoch,batch,loss_G,loss_G_identity,loss_G_GAN,loss_G_cycle,loss_D\n')
    def _save_png(self, tensor_1c, path):
        t = tensor_1c[0].detach().cpu().float().clamp(0,1)  # now expect 0..1
        arr = (t.squeeze(0).numpy() * 255.0).astype(np.uint8)  # (H,W)
        Image.fromarray(arr).save(path)
    def log(self, losses, images=None):
        self.mean_period += (time.time() - self.prev_time); self.prev_time = time.time()
        for k,v in losses.items(): self.acc[k] = self.acc.get(k,0.0) + float(v.detach().cpu())
        bd = self.batches_epoch*(self.epoch-1) + self.batch
        bl = self.batches_epoch*(self.n_epochs-self.epoch) + (self.batches_epoch-self.batch)
        eta = datetime.timedelta(seconds=int(bl * (self.mean_period/max(1,bd))))
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- %s -- ETA: %s' % (
            self.epoch, self.n_epochs, self.batch, self.batches_epoch,
            ' | '.join(f'{k}: {self.acc[k]/self.batch:.4f}' for k in
                       ['loss_G','loss_G_identity','loss_G_GAN','loss_G_cycle','loss_D'] if k in self.acc),
            str(eta)))
        with open(self.csv, 'a') as f:
            f.write(f"{self.epoch},{self.batch}," + ",".join(f"{float(losses[k].detach().cpu()):.6f}"
                  for k in ['loss_G','loss_G_identity','loss_G_GAN','loss_G_cycle','loss_D']) + "\n")
        if images and (self.batch % self.log_freq == 0):
            for name,t in images.items():
                self._save_png(t, os.path.join(self.img_dir, f'{self.epoch}_{self.batch}_{name}.png'))
        if self.batch % self.batches_epoch == 0:
            self.acc = {}; self.epoch += 1; self.batch = 1; sys.stdout.write('\n')
        else:
            self.batch += 1

# ----------------------------
# Train (unpaired)
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--n_epochs', type=int, default=200)
    ap.add_argument('--batchSize', type=int, default=4)
    ap.add_argument('--dataroot', type=str, required=True)
    ap.add_argument('--lr_dir', type=str, default='final_rxyacq_ROFFT')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--decay_epoch', type=int, default=100)
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--n_cpu', type=int, default=8)
    ap.add_argument('--outdir', type=str, default='output_sr_unpaired')
    ap.add_argument('--log_freq', type=int, default=200)
    ap.add_argument('--n_residual_blocks', type=int, default=9)
    ap.add_argument('--ckpt_freq', type=int, default=50)  # CHANGED: checkpoint frequency
    args = ap.parse_args()
    print(args)

    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'logs'), exist_ok=True)

    # CHANGED: channels
    input_nc_A = 2  # LR
    input_nc_B = 1  # HR

    # models  (CHANGED I/O)
    G_A2B = Generator(input_nc_A, input_nc_B, args.n_residual_blocks).to(device)  # 2 -> 1
    G_B2A = Generator(input_nc_B, input_nc_A, args.n_residual_blocks).to(device)  # 1 -> 2
    D_A = Discriminator(input_nc_A).to(device)  # judge LR  (2ch)
    D_B = Discriminator(input_nc_B).to(device)  # judge HR  (1ch)
    G_A2B.apply(weights_init_normal); G_B2A.apply(weights_init_normal)
    D_A.apply(weights_init_normal);   D_B.apply(weights_init_normal)

    # losses & optims
    crit_GAN = nn.MSELoss(); crit_cyc = nn.L1Loss(); crit_id = nn.L1Loss()
    opt_G  = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=args.lr, betas=(0.5,0.999))
    opt_DA = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5,0.999))
    opt_DB = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5,0.999))
    sch_G  = torch.optim.lr_scheduler.LambdaLR(opt_G,  lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    sch_DA = torch.optim.lr_scheduler.LambdaLR(opt_DA, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    sch_DB = torch.optim.lr_scheduler.LambdaLR(opt_DB, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

    # data (CHANGED: augment=False in ctor)
    ds = SPENUnpairedDataset(args.dataroot, lr_dir=args.lr_dir, augment=False)
    dl = DataLoader(ds, batch_size=args.batchSize, shuffle=True, num_workers=args.n_cpu, pin_memory=True, drop_last=True)

    buf_A, buf_B = ReplayBuffer(), ReplayBuffer()
    logger = Logger(args.n_epochs, len(dl), log_dir=os.path.join(args.outdir, 'logs'), log_freq=args.log_freq)

    for ep in range(args.epoch, args.n_epochs):
        for batch in dl:
            real_A = batch['A'].to(device)  # (B,2,H,W), already normalized (max |.| = 1)
            real_B = batch['B'].to(device)  # (B,1,H,W), already in [0,1]

            tgt_real = torch.ones(real_A.size(0),1, device=device)
            tgt_fake = torch.zeros(real_A.size(0),1, device=device)

            # ----- G -----
            opt_G.zero_grad()

            LAMBDA_ID  = 0.0   # disable identity loss
            LAMBDA_CYC = 5.0

            # identity (disabled because channels don't match)
            loss_id_A = torch.tensor(0.0, device=device)
            loss_id_B = torch.tensor(0.0, device=device)

            # adversarial
            fake_B = G_A2B(real_A)                         # 2->1
            loss_gan_A2B = crit_GAN(D_B(fake_B), tgt_real)

            fake_A = G_B2A(real_B)                         # 1->2
            loss_gan_B2A = crit_GAN(D_A(fake_A), tgt_real)

            # cycle
            rec_A = G_B2A(fake_B)                          # 1->2
            loss_cyc_ABA = crit_cyc(rec_A, real_A) * LAMBDA_CYC
            rec_B = G_A2B(fake_A)                          # 2->1
            loss_cyc_BAB = crit_cyc(rec_B, real_B) * LAMBDA_CYC

            loss_G = (loss_gan_A2B + loss_gan_B2A + loss_cyc_ABA + loss_cyc_BAB) \
                    + LAMBDA_ID * (loss_id_A + loss_id_B)  # stays numerically 0
            loss_G.backward(); opt_G.step()

            # ----- D_A -----
            opt_DA.zero_grad()
            loss_DA_real = crit_GAN(D_A(real_A), tgt_real)
            fake_A_buf = buf_A.push_and_pop(fake_A)
            loss_DA_fake = crit_GAN(D_A(fake_A_buf.detach()), tgt_fake)
            loss_D_A = 0.5*(loss_DA_real + loss_DA_fake)
            loss_D_A.backward(); opt_DA.step()

            # ----- D_B -----
            opt_DB.zero_grad()
            loss_DB_real = crit_GAN(D_B(real_B), tgt_real)
            fake_B_buf = buf_B.push_and_pop(fake_B)
            loss_DB_fake = crit_GAN(D_B(fake_B_buf.detach()), tgt_fake)
            loss_D_B = 0.5*(loss_DB_real + loss_DB_fake)
            loss_D_B.backward(); opt_DB.step()

            with torch.no_grad():
                vis_real_A = tensor_to_vis01(real_A)   # (B,1,H,W) in 0..1
                vis_real_B = tensor_to_vis01(real_B)
                vis_fake_A = tensor_to_vis01(fake_A)
                vis_fake_B = tensor_to_vis01(fake_B)

            logger.log(
                losses={
                    'loss_G': loss_G,
                    'loss_G_identity': (loss_id_A + loss_id_B),
                    'loss_G_GAN': (loss_gan_A2B + loss_gan_B2A),
                    'loss_G_cycle': (loss_cyc_ABA + loss_cyc_BAB),
                    'loss_D': (loss_D_A + loss_D_B),
                },
                images={'real_A': vis_real_A,'real_B': vis_real_B,'fake_A': vis_fake_A,'fake_B': vis_fake_B}
            )

        sch_G.step(); sch_DA.step(); sch_DB.step()

        # CHANGED: checkpoint frequency
        ck = os.path.join(args.outdir, 'checkpoints')
        last_epoch = (ep == args.n_epochs - 1)
        if ((ep + 1) % args.ckpt_freq == 0) or last_epoch:
            torch.save(G_A2B.state_dict(), os.path.join(ck, f'netG_A2B_e{ep:04d}.pth'))
            torch.save(G_B2A.state_dict(), os.path.join(ck, f'netG_B2A_e{ep:04d}.pth'))
            torch.save(D_A.state_dict(),  os.path.join(ck, f'netD_A_e{ep:04d}.pth'))
            torch.save(D_B.state_dict(),  os.path.join(ck, f'netD_B_e{ep:04d}.pth'))

if __name__ == "__main__":
    main()
