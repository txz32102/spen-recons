import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sys import path
path.append("/home/data1/musong/workspace/2025/8/08-20/spenpy")
from spenpy.spen import spen

def load_nii_slices(file_path, percent=0.3, rotate=True):
    """
    Load a NIfTI file, crop slices along the z-axis,
    and optionally rotate 90° anticlockwise.
    
    Args:
        file_path (str): Path to .nii or .nii.gz file.
        percent (float): Fraction to cut from both ends (0–0.5).
        rotate (bool): Whether to rotate slices 90° CCW.
        
    Returns:
        np.ndarray: Array of shape (b, w, h)
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    data = np.asarray(data, dtype=np.float32)

    # Crop along z
    z_start = int(percent * data.shape[2])
    z_end   = int((1 - percent) * data.shape[2])
    cropped = data[:, :, z_start:z_end]   # (X, Y, b)

    # Reorder to (b, w, h)
    cropped = np.transpose(cropped, (2, 0, 1))

    # Rotate each slice 90° CCW
    if rotate:
        cropped = np.array([np.rot90(slice_, k=1) for slice_ in cropped])

    return cropped

def resize_cwh(x, out_size):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.float()
    x = x.unsqueeze(0)
    x_resized = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
    return x_resized.squeeze(0)

IXI_path = "/home/data1/musong/data/IXI/T1/nii"
idx = "IXI579-Guys-1126-T1.nii.gz"

data = load_nii_slices(f"{IXI_path}/{idx}")
data = resize_cwh(data, (96, 96))

final_rxyacq_ROFFT = spen(acq_point=[96, 96]).sim(data)
phase_map = spen().get_phase_map(data)

_, AFinal = spen().get_InvA()