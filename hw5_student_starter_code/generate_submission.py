#!/usr/bin/env python3
"""
Student-facing script: generate a Kaggle submission CSV from generated images.

Two modes:
  1. CLI mode:  point to a directory of saved images (5000 expected)
  2. Python API: call generate_submission_from_tensors() directly from
                 your inference script — no need to save images to disk.

CLI Usage:
    # From saved images
    python generate_submission.py \
        --image_dir /path/to/generated_images \
        --output submission.csv

    # With local FID check against validation reference
    python generate_submission.py \
        --image_dir /path/to/generated_images \
        --output submission.csv \
        --reference val_stats.npz

Python API Usage (inside your inference.py):
    from generate_submission import generate_submission_from_tensors

    # all_images: tensor (5000, 3, H, W) in [-1, 1] or [0, 1]
    all_images = torch.cat(all_images, dim=0)
    generate_submission_from_tensors(
        all_images,
        output_csv="submission.csv",
        reference_npz="val_stats.npz",  # optional, for local FID check
    )
"""

import argparse
import os

import torch

from fid_utils import (
    extract_features_from_dir,
    extract_features_from_tensors,
    compute_statistics,
    compute_fid,
    stats_to_csv,
    save_stats_npz,
    load_stats_npz,
    validate_covariance,
)


def _process_features(features, output_csv, reference_npz=None, save_npz=None):
    """Shared logic: features -> stats -> CSV + optional FID check."""

    if features.shape[0] != 5000:
        print(f"\n*** WARNING: Expected 5000 images, found {features.shape[0]}. ***")
        print("*** The competition expects exactly 5000 generated images (50 per class). ***\n")

    mu, sigma = compute_statistics(features)

    is_valid, msg = validate_covariance(sigma)
    if not is_valid:
        print(f"\n*** WARNING: Covariance matrix validation failed: {msg} ***\n")

    print(f"\nSaving submission to {output_csv} ...")
    stats_to_csv(mu, sigma, output_csv)

    if save_npz:
        save_stats_npz(mu, sigma, save_npz)

    if reference_npz:
        print(f"\nComputing FID against {reference_npz} ...")
        mu_ref, sigma_ref = load_stats_npz(reference_npz)
        fid = compute_fid(mu, sigma, mu_ref, sigma_ref)
        print(f"\n{'=' * 40}")
        print(f"  FID (local reference) = {fid:.4f}")
        print(f"{'=' * 40}")
        print("(Note: Kaggle leaderboard uses test set reference, which may differ.)")
        return fid

    print("\nDone! Submit", output_csv, "to Kaggle.")
    return None


def generate_submission_from_tensors(
    images,
    output_csv="submission.csv",
    reference_npz=None,
    save_npz=None,
    device="cuda",
    batch_size=64,
):
    """
    Generate a Kaggle submission directly from image tensors.
    Call this from your inference script — no need to save images to disk.

    Args:
        images:        torch.Tensor (N, 3, H, W), float [-1,1] or [0,1]
        output_csv:    path for the submission CSV
        reference_npz: optional .npz for local FID check
        save_npz:      optional path to also save stats as .npz
        device:        torch device
        batch_size:    Inception inference batch size

    Returns:
        FID value if reference_npz is provided, else None
    """
    print("=" * 60)
    print("Extracting Inception-v3 features from image tensors ...")
    print("=" * 60)
    features = extract_features_from_tensors(images, device=device, batch_size=batch_size)
    return _process_features(features, output_csv, reference_npz, save_npz)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission CSV from generated images")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing generated images (5000 expected)")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output CSV file path")
    parser.add_argument("--reference", type=str, default=None,
                        help="Optional: reference .npz for local FID check")
    parser.add_argument("--save_npz", type=str, default=None,
                        help="Optional: also save stats as .npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("Extracting Inception-v3 features from image directory ...")
    print("=" * 60)
    features = extract_features_from_dir(
        args.image_dir, device=args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    _process_features(features, args.output, args.reference, args.save_npz)


if __name__ == "__main__":
    main()
