#!/usr/bin/python3
""" 
cd /home/data1/musong/workspace/2025/8/08-20/tr
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main
CUDA_VISIBLE_DEVICES=7 python3 /home/data1/musong/workspace/2025/8/08-20/tr/scripts/pm_lr_test.py \
--dataroot /home/data1/musong/workspace/2025/8/08-20/tr/test_data \
--cuda \
--log_dir /home/data1/musong/workspace/2025/8/08-20/tr/log/pm_lr/test \
--generator_lr2hr /home/data1/musong/workspace/2025/8/08-20/tr/log/pm_lr/weights/netG_lr2hr.pth
"""
import argparse
import sys
import os
from pathlib import Path
import csv
import glob

from scipy.io import loadmat
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
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

class SpenDataset(Dataset):
    def __init__(
        self,
        root: str,
    ):
        self.root = root

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
        path_lr = self.file_lr[index]
        lr_id = os.path.splitext(os.path.basename(path_lr))[0]

        lr = _load_lr_mag(path_lr) * 2 - 1
    
        return {"lr": lr, "lr_id": lr_id, "lr_path": path_lr}
    
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
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
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


netG = Generator(opt.input_nc, opt.output_nc)

device = torch.device("cuda" if opt.cuda else "cpu")

if opt.cuda:
    netG.cuda()

# Load state dicts
netG.load_state_dict(torch.load(opt.generator_lr2hr))

# Set model's test mode
netG.eval()

dataloader = DataLoader(SpenDataset(opt.dataroot),
                        batch_size=opt.batchSize,
                        num_workers=opt.n_cpu)

for i, batch in enumerate(dataloader):
    real_lr_11 = batch['lr'].to(device)   

    recovered_hr = netG(real_lr_11)

    os.makedirs(f'{opt.log_dir}/hr', exist_ok=True)

    # Save each sample in the batch
    for b in range(len(batch['lr_id'])):
        save_image((recovered_hr[b]+1)/2, f"{opt.log_dir}/hr/{batch['lr_id'][b]}_{i+1}.png")
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
    
    # calculate the metric
compute_set_metrics(
    data_root=f"{opt.dataroot}/hr",
    log_dir=f"{opt.log_dir}/hr",
    out_csv=f"{opt.log_dir}/metrix.csv",
)

sys.stdout.write('\n')
###################################