#!/usr/bin/python3


""" 
cd /home/data1/musong/workspace/2025/8/08-20/tr
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main
CUDA_VISIBLE_DEVICES=7 python3 /home/data1/musong/workspace/2025/8/08-20/tr/scripts/abs_lr_hr_test.py \
--dataroot /home/data1/musong/workspace/2025/8/08-20/tr/test_data \
--cuda \
--log_dir /home/data1/musong/workspace/2025/8/08-20/tr/log/abs_IXI_sim/test \
--generator_lr2hr /home/data1/musong/workspace/2025/8/08-20/tr/log/abs_IXI_sim/weights/netG_lr2hr.pth
"""
import argparse
import sys
import os
import random
from pathlib import Path
from typing import Callable, Optional
import csv
import glob

from scipy.io import loadmat
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

def _first_data_array(mat_dict: dict) -> np.ndarray:
    """Return the first non-meta array from a scipy.io.loadmat dict."""
    for k, v in mat_dict.items():
        if not k.startswith("__"):
            return v
    raise KeyError("No data key found in .mat file (only __meta keys present).")

def _to_3ch_tensor(img2d: np.ndarray) -> torch.Tensor:
    """
    Min-max normalize a 2D float array to [0, 1], then make it 3xHxW tensor
    by repeating the single channel.
    """
    x = img2d.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = np.zeros_like(x, dtype=np.float32)

    t = torch.from_numpy(x)            # H x W
    t = t.unsqueeze(0).repeat(3, 1, 1) # 3 x H x W
    return t

def _load_hr(path: str) -> torch.Tensor:
    """Load HR real image (HxW) -> 3xHxW tensor in [0,1]."""
    mat = loadmat(path)
    arr = _first_data_array(mat)
    arr = np.asarray(arr).squeeze()
    if np.iscomplexobj(arr):
        # HR is stated as real; if complex sneaks in, use real part
        arr = np.real(arr)
    return _to_3ch_tensor(arr)

def _load_lr_mag(path: str) -> torch.Tensor:
    """Load LR complex image (HxW), take magnitude -> 3xHxW tensor in [0,1]."""
    mat = loadmat(path)
    arr = _first_data_array(mat)
    arr = np.asarray(arr).squeeze()
    # Ensure magnitude if complex
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    else:
        # If already real, treat as-is
        arr = np.abs(arr)
    return _to_3ch_tensor(arr)


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

def png_to_img01(path: str) -> np.ndarray:
    """Load PNG (H x W x C or H x W), convert to float32 [0,1], return grayscale HxW by taking channel 0 if needed."""
    img = Image.open(path)
    # Ensure float in [0,1]
    arr = np.array(img).astype(np.float32)
    # If PNG saved from torchvision.save_image of float tensors, it’s already 0..255 uint8
    if arr.max() > 1.0:
        arr /= 255.0
    # If 3-channel, take the first channel (your saved fake_hr/fake_lr are 3ch copies)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

def compute_set_metrics(
    data_root: str,
    log_dir: str,
    out_csv: str,
) -> None:
    png_dir = log_dir
    mat_dir = data_root

    png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))

    # Map mat files by stem
    mat_map = {Path(p).stem: p for p in glob.glob(os.path.join(mat_dir, "*.mat"))}
    if not mat_map:
        raise FileNotFoundError(f"No .mat files found under {mat_dir}")

    rows = []
    psnrs, ssims = [], []

    for p_png in png_files:
        # Your saved names are like "<id>_<iter>.png"
        stem_iter = Path(p_png).stem
        # split by last underscore to recover original id (robust if id contains underscores)
        parts = stem_iter.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            stem = parts[0]
            iter_idx = parts[1]
        else:
            stem = stem_iter
            iter_idx = ""

        if stem not in mat_map:
            # If unaligned or naming differs, skip politely
            # (You can adapt here to a different mapping scheme if needed.)
            continue

        p_mat = mat_map[stem]
        gt = mat_to_img01(p_mat)
        pr = png_to_img01(p_png)

        # skimage expects data_range set correctly for float images
        psnr = peak_signal_noise_ratio(gt, pr, data_range=1.0)
        ssim = structural_similarity(gt, pr, data_range=1.0)
        psnrs.append(psnr)
        ssims.append(ssim)

        rows.append({
            "id": stem,
            "iter": iter_idx,
            "png": p_png,
            "mat": p_mat,
            "H": gt.shape[0],
            "W": gt.shape[1],
            "PSNR": psnr,
            "SSIM": ssim
        })

    if not rows:
        raise RuntimeError("Found no matched (PNG, MAT) pairs. Check filenames and directories.")

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Wrote {len(rows)} rows -> {out_csv}")
    print(f"[=] Mean PSNR: {np.mean(psnrs):.4f} dB | Mean SSIM: {np.mean(ssims):.4f}")

class ImageDataset(Dataset):
    """
    Loads paired .mat files from:
        <root>/hr/*.mat  (real, HxW)
        <root>/lr/*.mat  (complex, HxW)  -> use magnitude

    Returns dict: {'A': HR_tensor_3ch, 'B': LRmag_tensor_3ch}

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

        self.files_lr = sorted(glob.glob(os.path.join(root, "lr", "*.mat")))

        if len(self.files_lr) == 0:
            raise FileNotFoundError(
                f"No .mat files found under '{root}/lr'. "
                f"Got {len(self.files_lr)} LR files."
            )

    def __len__(self):
        return len(self.files_lr)

    def __getitem__(self, index: int):
        if self.unaligned:
            j = random.randint(0, len(self.files_lr) - 1)
        else:
            j = index % len(self.files_lr)
        path_lr = self.files_lr[j]

        lr = _load_lr_mag(path_lr) * 2 - 1   # LR (complex) -> |.| -> 3ch tensor
        
        lr_id = os.path.splitext(os.path.basename(path_lr))[0]

        return {"lr": lr, "lr_id": lr_id, "lr_path": path_lr}

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


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=96, help='size of the data (squared assumed)')
parser.add_argument("--which", choices=["hr","lr"], default="hr", help="Evaluate recon set.")
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_lr2hr', type=str, default='output/netG_lr2hr.pth', help='B2A generator checkpoint file')
parser.add_argument('--log_dir', type=str, default='logs/', help='directory to save logs and model checkpoints')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_lr2hr = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_lr2hr.cuda()

# Load state dicts
netG_lr2hr.load_state_dict(torch.load(opt.generator_lr2hr))

# Set model's test mode
netG_lr2hr.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_lr = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)


for i, batch in enumerate(dataloader):
    real_lr = Variable(input_lr.copy_(batch['lr']))

    fake_hr = 0.5 * (netG_lr2hr(real_lr).data + 1.0)

    os.makedirs(f'{opt.log_dir}/hr', exist_ok=True)

    # Save each sample in the batch
    for b in range(len(batch['lr_id'])):
        save_image(fake_hr[b], f"{opt.log_dir}/hr/{batch['lr_id'][b]}_{i+1}.png")
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
    
    # calculate the metric
compute_set_metrics(
    data_root=f"{opt.dataroot}/hr",
    log_dir=f"{opt.log_dir}/hr",
    out_csv=f"{opt.log_dir}/metrix.csv",
)

sys.stdout.write('\n')
###################################