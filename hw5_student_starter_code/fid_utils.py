"""
Shared utilities for FID computation in the IDL HW5 Kaggle competition.

Uses torchmetrics' Inception-v3 model internally so FID values are
consistent with what students compute via FrechetInceptionDistance.

Supports two input modes:
  1. Tensor input  — pass image tensors directly (no disk I/O)
  2. Directory input — load images from a folder on disk

Provides:
- Inception-v3 feature extraction (2048-d pool3 features)
- FID computation from statistics
- CSV serialization for Kaggle submission
- NPZ save/load for reference stats
"""

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from scipy import linalg
from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance


FEATURE_DIM = 2048


# ============================================================
# Inception-v3 Feature Extraction
# ============================================================

def get_inception_model(device='cuda'):
    """
    Get the same Inception-v3 model that torchmetrics uses for FID.

    This ensures feature extraction is identical whether students use
    torchmetrics' FrechetInceptionDistance directly or these utilities.

    The model expects float tensors in [0, 1] range, any spatial size
    (it resizes to 299x299 internally).
    """
    fid_metric = FrechetInceptionDistance(feature=FEATURE_DIM)
    model = fid_metric.inception
    model.eval()
    model.to(device)
    return model


class FlatImageDataset(Dataset):
    """
    Loads all images from a directory tree (any depth).
    Works with both ImageFolder structure and flat directories.
    Returns float tensors in [0, 1] range (matching torchmetrics' expectation).
    """

    EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.PNG', '*.JPG')

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform or transforms.ToTensor()
        self.image_paths = []
        for ext in self.EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
        self.image_paths = sorted(self.image_paths)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


@torch.no_grad()
def extract_features_from_dir(image_dir, device='cuda', batch_size=64, num_workers=4):
    """
    Extract Inception features from all images in a directory.

    Images are loaded and converted to float [0, 1] via ToTensor().
    The Inception model handles resizing to 299x299 internally.

    Args:
        image_dir: path to images (ImageFolder structure or flat)
        device: torch device string
        batch_size: inference batch size
        num_workers: dataloader workers

    Returns:
        np.ndarray of shape (N, 2048)
    """
    model = get_inception_model(device)
    dataset = FlatImageDataset(image_dir, transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.PILToTensor(),  # uint8 [0, 255] as expected by torchmetrics Inception
    ]))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    print(f"Extracting features from {len(dataset)} images in {image_dir}")
    all_features = []
    for batch in tqdm(dataloader, desc="Inception features"):
        batch = batch.to(device)
        features = model(batch)  # (B, 2048)
        all_features.append(features.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    print(f"Extracted features shape: {features.shape}")
    return features


@torch.no_grad()
def extract_features_from_tensors(images, device='cuda', batch_size=64):
    """
    Extract Inception features from a tensor of images.

    Accepts two input ranges:
      - float [0, 1]   — e.g. from torchvision transforms
      - float [-1, 1]  — e.g. raw output from a diffusion model

    If the min value is negative, the images are automatically rescaled
    from [-1, 1] to [0, 1].

    Args:
        images: torch.Tensor of shape (N, 3, H, W)
        device: torch device string
        batch_size: inference batch size

    Returns:
        np.ndarray of shape (N, 2048)
    """
    model = get_inception_model(device)

    # Auto-detect [-1, 1] range and rescale to [0, 1]
    if images.min() < -0.01:
        print("Detected [-1, 1] range, rescaling to [0, 1]")
        images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)

    print(f"Extracting features from {len(images)} image tensors")
    all_features = []
    for i in tqdm(range(0, len(images), batch_size), desc="Inception features"):
        batch = images[i:i + batch_size].to(device)
        features = model(batch)  # (B, 2048)
        all_features.append(features.cpu())

    features = torch.cat(all_features, dim=0).numpy()
    print(f"Extracted features shape: {features.shape}")
    return features


# Legacy alias for backward compatibility
extract_features = extract_features_from_dir


# ============================================================
# Statistics
# ============================================================

def compute_statistics(features):
    """
    Compute mean and covariance of Inception features.

    Args:
        features: np.ndarray of shape (N, 2048)

    Returns:
        mu:    np.ndarray of shape (2048,)
        sigma: np.ndarray of shape (2048, 2048)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Frechet Inception Distance between two Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrtm(sigma1 @ sigma2))

    Returns:
        float: FID score (lower = better)
    """
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # Discard negligible imaginary parts from numerical error
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(
                "Significant imaginary component in matrix square root. "
                "This usually means one of the covariance matrices is not PSD."
            )
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


# ============================================================
# NPZ save / load
# ============================================================

def save_stats_npz(mu, sigma, filepath):
    """Save (mu, sigma) to a compressed .npz file."""
    np.savez_compressed(filepath, mu=mu, sigma=sigma)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved stats to {filepath} ({size_mb:.1f} MB)")


def load_stats_npz(filepath):
    """Load (mu, sigma) from a .npz file."""
    data = np.load(filepath)
    return data['mu'], data['sigma']


# ============================================================
# CSV serialization (for Kaggle submission)
# ============================================================

def stats_to_csv(mu, sigma, filepath, is_solution=False):
    """
    Save (mu, sigma) as a wide CSV for Kaggle.

    Solution format:  id, mu, s_0, ..., s_2047, Usage  (2051 cols)
    Submission format: mu, s_0, ..., s_2047             (2049 cols)

    Kaggle strips id + Usage from solution and expects submission
    to have exactly the remaining columns.
    """
    import pandas as pd

    # id column is always included (Kaggle needs it for row matching)
    data_dict = {'id': np.arange(FEATURE_DIM), 'mu': mu}
    data_dict.update({f's_{j}': sigma[:, j] for j in range(FEATURE_DIM)})

    if is_solution:
        data_dict['Usage'] = 'Public'

    df = pd.DataFrame(data_dict)

    print(f"Writing {df.shape[0]} rows x {df.shape[1]} cols to {filepath} ...")
    df.to_csv(filepath, index=False, float_format='%.8f')
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved CSV to {filepath} ({size_mb:.1f} MB)")


def csv_to_stats(filepath):
    """
    Reconstruct (mu, sigma) from a wide Kaggle CSV.

    Handles both submission format (id, mu, s_0..s_2047)
    and solution format (id, mu, s_0..s_2047, Usage).

    Returns:
        mu:    np.ndarray (2048,)
        sigma: np.ndarray (2048, 2048)
    """
    import pandas as pd

    df = pd.read_csv(filepath)

    if 'mu' not in df.columns:
        raise ValueError("CSV must have a 'mu' column")

    # Drop non-data columns if present
    df = df.drop(columns=['id', 'Usage'], errors='ignore')

    mu = df['mu'].values.astype(np.float64)
    sigma_cols = [f's_{j}' for j in range(FEATURE_DIM)]
    sigma = df[sigma_cols].values.astype(np.float64)

    if mu.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected {FEATURE_DIM} rows, got {mu.shape[0]}")

    return mu, sigma


# ============================================================
# Validation helpers
# ============================================================

def validate_covariance(sigma, atol=1e-5):
    """
    Check that sigma is a valid covariance matrix (symmetric, PSD).
    Returns (is_valid, message).
    """
    if sigma.shape != (FEATURE_DIM, FEATURE_DIM):
        return False, f"Wrong shape: {sigma.shape}"

    if not np.allclose(sigma, sigma.T, atol=atol):
        return False, "Not symmetric"

    eigenvalues = np.linalg.eigvalsh(sigma)
    if np.min(eigenvalues) < -atol:
        return False, f"Not PSD: min eigenvalue = {np.min(eigenvalues):.6e}"

    return True, "OK"
