"""
VAE Evaluation Script

Evaluate VAE compression quality and reconstruction quality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from model.vae import VAE
from dataloader.data import SimpleMotionDataset
from torch.utils.data import DataLoader


def evaluate_vae():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))
    vae_checkpoint = 'checkpoints/vae_best.pt'

    print("=" * 60)
    print("VAE Evaluation")
    print("=" * 60)
    print(f"  Device: {device}")

    # Load VAE
    print("\n[1/4] Loading VAE...")
    vae = VAE(
        nfeats=70,
        embed_dim=384,
        n_head=6,
        d_ff=1536,
        num_layers=4,
        latent_dim=128,
    ).to(device)

    checkpoint_path = Path(vae_checkpoint)
    if not checkpoint_path.exists():
        print(f"  ERROR: VAE checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    print(f"  Loaded VAE from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")

    # Load validation data
    print("\n[2/4] Loading validation data...")
    val_dataset = SimpleMotionDataset(
        datadir='dataset',
        split='val',
        history_len=10,
        future_len=20,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    print(f"  Val dataset: {len(val_dataset)} samples")

    # Evaluate
    print("\n[3/4] Evaluating VAE...")

    all_latents = []
    all_recon_errors = []
    all_kl_divs = []

    with torch.no_grad():
        for history, future, text_emb in val_loader:
            history = history.to(device)
            future = future.to(device)

            batch_size = history.shape[0]

            # Encode
            latent, dist = vae.encode(history, future)
            all_latents.append(latent.cpu())

            # Decode (correct order: latent, history_motion, future_len)
            future_recon = vae.decode(latent, history, future_len=20)

            # Reconstruction error (MSE per feature)
            mse_per_feature = ((future_recon - future) ** 2).mean(dim=(1, 2))
            all_recon_errors.append(mse_per_feature.cpu())

            # KL divergence (KL(N(mu, sigma) || N(0, 1)))
            # Need to compute manually: KL = log(1/sigma) + (sigma^2 + mu^2)/2 - 0.5
            mu = dist.loc
            std = dist.scale
            kl = -torch.log(std) + (std**2 + mu**2) / 2 - 0.5
            kl = kl.sum(dim=1)  # Sum over latent dimensions
            all_kl_divs.append(kl.cpu())

    # Concatenate
    all_latents = torch.cat(all_latents, dim=0)  # (N, 128)
    all_recon_errors = torch.cat(all_recon_errors, dim=0)  # (N,)
    all_kl_divs = torch.cat(all_kl_divs, dim=0)  # (N,)

    # Statistics
    print("\n[4/4] Results:")
    print("=" * 60)

    # Latent statistics
    latent_mean = all_latents.mean(dim=0)
    latent_std = all_latents.std(dim=0)
    latent_global_std = all_latents.std().item()

    print(f"\n  Latent Statistics:")
    print(f"    Shape: {all_latents.shape}")
    print(f"    Global mean: {all_latents.mean().item():.6f}")
    print(f"    Global std:  {latent_global_std:.6f}")
    print(f"    Per-dim std range: [{latent_std.min().item():.4f}, {latent_std.max().item():.4f}]")
    print(f"    Near-zero dims (std < 0.01): {(latent_std < 0.01).sum().item()}/128")

    # Reconstruction error
    print(f"\n  Reconstruction Error (MSE):")
    print(f"    Mean:  {all_recon_errors.mean().item():.6f}")
    print(f"    Std:   {all_recon_errors.std().item():.6f}")
    print(f"    Min:   {all_recon_errors.min().item():.6f}")
    print(f"    Max:   {all_recon_errors.max().item():.6f}")

    # KL divergence
    print(f"\n  KL Divergence:")
    print(f"    Mean:  {all_kl_divs.mean().item():.6f}")
    print(f"    Std:   {all_kl_divs.std().item():.6f}")

    # Diagnosis
    print("\n" + "=" * 60)
    print("Diagnosis:")
    print("=" * 60)

    # Latent quality check
    if latent_global_std < 0.1:
        print("  ⚠️  WARNING: Latent std is too low! VAE may have collapsed.")
        print("     Consider: higher KL weight, retraining VAE")
    elif latent_global_std < 0.5:
        print("  ⚠️  WARNING: Latent std is low. VAE may be underutilized.")
    else:
        print("  ✅ Latent std is reasonable.")

    # Near-zero dimensions check
    near_zero = (latent_std < 0.01).sum().item()
    if near_zero > 10:
        print(f"  ⚠️  WARNING: {near_zero}/128 dims are near-zero!")
    else:
        print(f"  ✅ Only {near_zero}/128 dims are near-zero.")

    # Reconstruction check
    recon_mse = all_recon_errors.mean().item()
    if recon_mse > 0.5:
        print("  ⚠️  WARNING: Reconstruction error is high!")
    elif recon_mse > 0.2:
        print("  ⚠️  Reconstruction error is moderate.")
    else:
        print("  ✅ Reconstruction error is acceptable.")

    # KL check
    kl_mean = all_kl_divs.mean().item()
    if kl_mean < 0.001:
        print("  ⚠️  WARNING: KL is near zero! Posterior collapse.")
    elif kl_mean > 10:
        print("  ⚠️  WARNING: KL is too high! Latent may not match prior.")
    else:
        print("  ✅ KL divergence is reasonable.")

    print("=" * 60)

    # Visualize latent distribution
    print("\n  Saving latent distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Latent mean histogram
    axes[0].hist(latent_mean.numpy(), bins=50, alpha=0.7)
    axes[0].set_xlabel('Latent Mean')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Latent Mean Distribution (global={latent_global_std:.4f})')
    axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

    # Latent std histogram
    axes[1].hist(latent_std.numpy(), bins=50, alpha=0.7, color='orange')
    axes[1].set_xlabel('Per-Dim Std')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Per-Dimension Std Distribution')

    plt.tight_layout()
    plt.savefig('checkpoints/vae_latent_dist.png', dpi=150)
    print(f"  Saved: checkpoints/vae_latent_dist.png")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    evaluate_vae()
