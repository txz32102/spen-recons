#!/usr/bin/python3

""" 
cd /home/data1/musong/workspace/python/spen-recons
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main
CUDA_VISIBLE_DEVICES=1 python3 /home/data1/musong/workspace/python/spen-recons/scripts/pm_lr_train.py \
--dataroot /home/data1/musong/workspace/python/spen-recons/data/IXI_sim \
--log_dir /home/data1/musong/workspace/python/spen-recons/log/pm_lr
"""


import argparse
import os
import sys
import glob
import random
import time
import datetime
from typing import Callable, Optional

import numpy as np
from PIL import Image
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from spenpy.spen import spen

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


def tensor2image(tensor):
    """
    Convert a torch tensor in [-1,1] to a grayscale numpy image.
    Handles:
      (1,H,W) -> (H,W)
      (2,H,W) -> sqrt(real^2 + imag^2) -> (H,W)
    """
    t = tensor.detach()
    arr = t[0].cpu().float().numpy()  # (C,H,W)

    if arr.shape[0] == 1:
        # Single channel -> grayscale
        img = arr[0]
        img = 127.5 * (img + 1.0)
    elif arr.shape[0] == 2:
        # Two channels -> magnitude
        real, imag = arr[0], arr[1]
        img = np.sqrt(((real+1)/2)**2 + ((imag+1)/2)**2)
        img = img * 255.0
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape[0]}")

    
    return img.astype(np.uint8)

class Logger:
    def __init__(self, n_epochs, batches_epoch, log_dir='logs', running_avg=False):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0.0

        self.running_avg = running_avg

        self.losses = {}              # per-batch values or running sums (see running_avg)
        self.counts = {}              # only used if running_avg=True
        self.header_written = False   # write header lazily on first log()

        # dirs
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses_log_path = os.path.join(log_dir, 'losses.csv')
        self.images_log_path = os.path.join(log_dir, 'images')
        os.makedirs(self.images_log_path, exist_ok=True)

        # create empty file
        with open(self.losses_log_path, 'w') as f:
            pass

    def _ensure_header(self, losses):
        if not self.header_written:
            keys = list(losses.keys())  # use names exactly as passed in
            with open(self.losses_log_path, 'a') as f:
                f.write('epoch,batch,' + ','.join(keys) + '\n')
            self.header_written = True

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        # Write header lazily when we first see losses
        if losses and not self.header_written:
            self._ensure_header(losses)

        # Progress line
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' %
                         (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        # Update tracked losses
        row_vals = []
        if losses:
            for i, (name, val) in enumerate(losses.items()):
                v = float(val.item() if hasattr(val, "item") else val)

                if self.running_avg:
                    # accumulate sums + counts
                    self.losses[name] = self.losses.get(name, 0.0) + v
                    self.counts[name] = self.counts.get(name, 0) + 1
                    disp = self.losses[name] / self.counts[name]
                else:
                    # per-batch value
                    self.losses[name] = v
                    disp = v

                row_vals.append(disp)

                sep = ' -- ' if (i + 1) == len(losses) else ' | '
                sys.stdout.write(f'{name}: {disp:.4f}{sep}')

        # ETA
        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + (self.batches_epoch - self.batch)
        eta = datetime.timedelta(seconds=(batches_left * self.mean_period / max(1, batches_done)))
        sys.stdout.write(f'ETA: {eta}')

        # Append CSV row
        if losses:
            with open(self.losses_log_path, 'a') as f:
                f.write(f'{self.epoch},{self.batch},' +
                        ','.join(f'{x:.6f}' for x in row_vals) + '\n')

        if images:
            for name, t in images.items():
                img = tensor2image(t)
                Image.fromarray(img).save(os.path.join(self.images_log_path, f'{self.epoch}_{self.batch}_{name}.png'))

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            if self.running_avg:
                # reset running stats at epoch end
                self.losses.clear()
                self.counts.clear()
            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def _first_data_array(mat_dict: dict) -> np.ndarray:
    """Return the first non-meta array from a scipy.io.loadmat dict."""
    for k, v in mat_dict.items():
        if not k.startswith("__"):
            return v
    raise KeyError("No data key found in .mat file (only __meta keys present).")

def _load_hr(path: str) -> torch.Tensor:
    """Load HR real image (HxW) -> 1xHxW tensor in [0,1]."""
    mat = loadmat(path)
    arr = _first_data_array(mat)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / arr.max()
    return torch.tensor(arr.astype(np.float32))

def _load_lr_mag(path: str) -> torch.Tensor:
    """
    Load LR complex image (H x W) from .mat file.
    Normalize by max(abs) to [0,1], then split into
    2xHxW tensor: [real, imag].
    """
    # Load array
    mat = loadmat(path)
    arr = _first_data_array(mat)
    arr = np.expand_dims(arr, axis=0)

    # Ensure complex dtype
    arr = arr.astype(np.complex64)

    # Normalize by max magnitude
    mag = np.abs(arr)
    max_val = mag.max()
    if max_val > 0:
        arr = arr / max_val

    # Split real & imag into 2 channels
    real = np.real(arr)
    imag = np.imag(arr)
    out = np.sqrt(real ** 2 + imag ** 2)

    return torch.from_numpy(out.astype(np.float32))

def _complex_to_1ch(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex tensor (B,1,W,H) -> real tensor (B,1,W,H).
    Uses magnitude, normalizes to [0,1], then rescales to [-1,1].
    """
    if not torch.is_complex(x):
        raise ValueError("Input must be a complex tensor")

    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected input of shape (B,1,W,H), got {tuple(x.shape)}")

    # Magnitude
    mag = x.abs()  # (B,1,W,H), real

    # Normalize to [0,1]
    max_val = mag.amax(dim=(2,3), keepdim=True)
    normed = mag  / max_val

    # Rescale to [-1,1]
    out = normed * 2.0 - 1.0
    return out

class SpenDataset(Dataset):
    """
    Loads paired .mat files from:
        <root>/hr/*.mat  (real, HxW)
        <root>/lr/*.mat  (complex, HxW)  -> use magnitude

    Args:
        root: dataset root folder
        transforms_: optional torchvision transforms to apply (PIL or tensor).
                     If the transforms expect PIL images, they will be given a PIL image
                     converted from the 3ch tensor. Otherwise they’ll receive a tensor.
        unaligned: if True, sample B from a random index instead of aligned index
    """
    def __init__(
        self,
        root: str,
        transforms_: Optional[Callable] = None,
        unaligned: bool = False,
    ):
        self.root = root
        self.unaligned = unaligned
        self.transform = transforms_

        # Collect .mat files
        self.file_lr = sorted(glob.glob(os.path.join(root, "lr", "*.mat")))
        self.file_hr = sorted(glob.glob(os.path.join(root, "hr", "*.mat")))

        if len(self.file_lr) == 0 or len(self.file_hr) == 0:
            raise FileNotFoundError(
                f"No .mat files found under '{root}/hr' or '{root}/lr'. "
                f"Got {len(self.file_lr)} HR files and {len(self.file_hr)} LR files."
            )

    def __len__(self):
        return max(len(self.file_lr), len(self.file_hr))

    def __getitem__(self, index: int):
        path_hr = self.file_hr[index % len(self.file_hr)]
        if self.unaligned:
            j = random.randint(0, len(self.file_hr) - 1)
        else:
            j = index % len(self.file_hr)
        path_lr = self.file_lr[j]

        hr = _load_hr(path_hr) * 2 - 1
        lr = _load_lr_mag(path_lr) * 2 - 1
    
        return {"hr": hr, "lr": lr}
    
class physical_model:
    def __init__(self, img_size=(96, 96), device='cuda'):
        self.InvA, self.AFinal = spen(acq_point=img_size).get_InvA()
        self.InvA = torch.as_tensor(self.InvA).detach().to(device=device, dtype=torch.complex64)
        self.AFinal = torch.as_tensor(self.AFinal).detach().to(device=device, dtype=torch.complex64)
    
    def __call__(self, x, phase_map=None):
        x = x.detach().to(torch.complex64)
        x = torch.matmul(self.AFinal * 1j, x)
        if phase_map is not None:
            x[:, 1::2, :] *= torch.exp(1j * phase_map)
        return x
    
    def recons(self, x, phase_map=None):
        if phase_map is not None:
            x[:, 1::2, :] *= torch.exp(-1j * phase_map)
        return torch.matmul(self.InvA, x)    

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/data1/musong/workspace/python/spen-recons/data/IXI_sim', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=96, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='disable GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--log_dir', type=str, default='/home/data1/musong/workspace/python/spen-recons/log/pm_lr', help='directory to save logs and model checkpoints')
parser.add_argument('--ckpt_save_freq', type=int, default=50, help='save checkpoint frequency (in epochs)')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# --- nets ---
netG = Generator(opt.input_nc, opt.output_nc)
netD_hr = Discriminator(opt.output_nc)   # disc over HR (1ch)
netD_lr = Discriminator(opt.output_nc)   # disc over LR (1ch mag)

PM = physical_model()  # HR->[complex LR]; we will convert to 1ch

if opt.cuda:
    netG.cuda(); netD_hr.cuda(); netD_lr.cuda()

netG.apply(weights_init_normal)
netD_hr.apply(weights_init_normal)
netD_lr.apply(weights_init_normal)

# --- losses & opts ---
criterion_GAN   = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

optimizer_G    = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_hr = torch.optim.Adam(netD_hr.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_lr = torch.optim.Adam(netD_lr.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G    = torch.optim.lr_scheduler.LambdaLR(optimizer_G,    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_hr = torch.optim.lr_scheduler.LambdaLR(optimizer_D_hr, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_lr = torch.optim.lr_scheduler.LambdaLR(optimizer_D_lr, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# --- helpers ---
device = torch.device("cuda" if opt.cuda else "cpu")


# targets shaped like D outputs (B,1)
target_real = torch.ones((opt.batchSize, 1), dtype=torch.float32, device=device)
target_fake = torch.zeros((opt.batchSize, 1), dtype=torch.float32, device=device)

fake_hr_buf = ReplayBuffer()
fake_lr_buf = ReplayBuffer()

dataloader = DataLoader(SpenDataset(opt.dataroot, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True,
                        num_workers=opt.n_cpu, drop_last=True)

os.makedirs(opt.log_dir, exist_ok=True)
os.makedirs(f'{opt.log_dir}/train', exist_ok=True)
logger = Logger(opt.n_epochs, len(dataloader), f'{opt.log_dir}/train')

# --- training ---
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_hr_11 = batch['hr'].to(device)         # HR in [-1,1], 1ch
        real_lr_11 = batch['lr'].to(device)         # LR may be 1ch or 2ch

        # --------- G step ---------
        optimizer_G.zero_grad()

        # path A: HR -> PM -> LR1 -> G -> HR_hat ; adversarial on HR
        pm_lr_1 = PM((real_hr_11+1)/2)             # 1ch [-1,1]
        pm_lr_1 = _complex_to_1ch(pm_lr_1)
        fake_hr = netG(pm_lr_1)                     # 1ch [-1,1]
        pred_fake_hr = netD_hr(fake_hr)
        loss_GAN_hr = criterion_GAN(pred_fake_hr, target_real)

        # path B: LR (data) -> G -> HR_tilde -> PM -> LR_tilde ; adversarial on LR
        recovered_hr = netG(real_lr_11)             # 1ch [-1,1]
        pm_lr_2 = PM((recovered_hr+1)/2)        # 1ch [-1,1]
        pm_lr_2 = _complex_to_1ch(pm_lr_2)
        pred_fake_lr = netD_lr(pm_lr_2)
        loss_GAN_lr = criterion_GAN(pred_fake_lr, target_real)

        # cycle losses (both in matching spaces)
        # LR->HR->PM should match LR (both 1ch [-1,1])
        loss_cycle_lrhrlr  = criterion_cycle(pm_lr_2, real_lr_11) * 5.0
        # HR->PM->G should reconstruct HR
        loss_cycle_hrlrhr  = criterion_cycle(fake_hr, real_hr_11) * 5.0

        loss_G = loss_GAN_hr + loss_GAN_lr + loss_cycle_lrhrlr + loss_cycle_hrlrhr
        loss_G.backward()
        optimizer_G.step()

        # --------- D_hr step ---------
        optimizer_D_hr.zero_grad()
        pred_real_hr = netD_hr(real_hr_11)
        loss_D_hr_real = criterion_GAN(pred_real_hr, target_real)

        fake_hr_buf_out = fake_hr_buf.push_and_pop(fake_hr.detach())
        pred_fake_hr = netD_hr(fake_hr_buf_out)
        loss_D_hr_fake = criterion_GAN(pred_fake_hr, target_fake)

        loss_D_hr = 0.5 * (loss_D_hr_real + loss_D_hr_fake)
        loss_D_hr.backward()
        optimizer_D_hr.step()

        # --------- D_lr step ---------
        optimizer_D_lr.zero_grad()
        pred_real_lr = netD_lr(real_lr_11)
        loss_D_lr_real = criterion_GAN(pred_real_lr, target_real)

        fake_lr_buf_out = fake_lr_buf.push_and_pop(pm_lr_2.detach())
        pred_fake_lr = netD_lr(fake_lr_buf_out)
        loss_D_lr_fake = criterion_GAN(pred_fake_lr, target_fake)

        loss_D_lr = 0.5 * (loss_D_lr_real + loss_D_lr_fake)
        loss_D_lr.backward()
        optimizer_D_lr.step()

        # --------- logging ---------
        if i == len(dataloader) - 1:
            logger.log(
                {
                    'loss_G': loss_G,
                    'loss_GAN_hr': loss_GAN_hr,
                    'loss_GAN_lr': loss_GAN_lr,
                    'loss_cycle_lrhrlr': loss_cycle_lrhrlr,
                    'loss_cycle_hrlrhr': loss_cycle_hrlrhr,
                    'loss_D_hr': loss_D_hr,
                    'loss_D_lr': loss_D_lr
                },
                images={
                    'real_hr': real_hr_11,
                    'real_lr': real_lr_11,
                    'fake_hr': fake_hr,
                    'pm_lr_from_recHR': pm_lr_2,
                    'pm_lr_from_realHR': pm_lr_1,
                    'recovered_hr': recovered_hr
                }
            )
        else:
            logger.log(
                {
                    'loss_G': loss_G,
                    'loss_GAN_hr': loss_GAN_hr,
                    'loss_GAN_lr': loss_GAN_lr,
                    'loss_cycle_lrhrlr': loss_cycle_lrhrlr,
                    'loss_cycle_hrlrhr': loss_cycle_hrlrhr,
                    'loss_D_hr': loss_D_hr,
                    'loss_D_lr': loss_D_lr
                }
            )

    # schedulers
    lr_scheduler_G.step()
    lr_scheduler_D_hr.step()
    lr_scheduler_D_lr.step()

    # checkpoints
    if (epoch % opt.ckpt_save_freq) == 0:
        wdir = f'{opt.log_dir}/weights'; os.makedirs(wdir, exist_ok=True)
        torch.save(netG.state_dict(),     f'{wdir}/netG_lr2hr.pth')
        torch.save(netD_hr.state_dict(),  f'{wdir}/netD_hr.pth')
        torch.save(netD_lr.state_dict(),  f'{wdir}/netD_lr.pth')
        print(f"[Checkpoint] Saved models at epoch {epoch}")
