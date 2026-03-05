#!/usr/bin/env python3
"""
VAE Reconstruction Visualization - Simple Feature Comparison

Compare 70-dim motion features: original vs reconstructed.

Usage:
    python visualize_vae_reconstruction.py
    python visualize_vae_reconstruction.py --idx 5
    python visualize_vae_reconstruction.py --save checkpoints/recon.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model.vae import VAE
from dataloader.data import SimpleMotionDataset


# Feature groups (70-dim)
FEATURE_GROUPS = [
    ('Root Height', 0, 1),
    ('Root Rotation', 1, 7),
    ('Root Velocity', 7, 10),
    ('Joint Pos', 10, 33),
    ('Joint Vel', 33, 56),
    ('Left Foot', 56, 59),
    ('Right Foot', 59, 62),
    ('Left Hand', 62, 65),
    ('Right Hand', 65, 68),
    ('Contact', 68, 70),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/vae_best.pt')
    parser.add_argument('--split', default='val', choices=['train', 'val'])
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--save', default=None, help='Save path (default: checkpoints/vae_recon_{idx}.png)')
    parser.add_argument('--top-k', type=int, default=5, help='Show top-k worst features')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load VAE (config must match checkpoint: embed_dim=384, n_head=6, d_ff=1536, num_layers=4)
    print(f"Loading VAE from {args.checkpoint}...")
    vae = VAE(nfeats=70, latent_dim=128, embed_dim=384, n_head=6,
              d_ff=1536, num_layers=4).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt['model_state_dict'])
    vae.eval()
    print(f"  Epoch {ckpt['epoch']}, val_loss: {ckpt['val_loss']:.4f}")

    # Load data
    dataset = SimpleMotionDataset('dataset', args.split, 10, 20)
    history, future, _ = dataset[args.idx]

    # Reconstruct
    with torch.no_grad():
        latent, _ = vae.encode(history.unsqueeze(0).to(device),
                               future.unsqueeze(0).to(device))
        future_recon = vae.decode(latent, history.unsqueeze(0).to(device), 20)
    future_recon = future_recon.squeeze(0).cpu()

    # Metrics
    mse_per_feat = ((future - future_recon) ** 2).mean(dim=0)
    total_mse = mse_per_feat.mean().item()

    print(f"\nSample {args.idx}: Total MSE = {total_mse:.6f}")
    print(f"Top-{args.top_k} worst features:")
    worst_idxs = mse_per_feat.topk(args.top_k).indices.tolist()
    for i, idx in enumerate(worst_idxs, 1):
        group = next((n for n, s, e in FEATURE_GROUPS if s <= idx < e), 'Unknown')
        print(f"  {i}. Feature {idx} ({group}): MSE = {mse_per_feat[idx]:.4f}")

    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    for i, (name, start, end) in enumerate(FEATURE_GROUPS):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        for j in range(start, end):
            offset = (j - start) * 0.3
            ax.plot(future[:, j] + offset, 'b-', alpha=0.3, lw=0.8, label='Ground Truth' if j == start else '')
            ax.plot(future_recon[:, j] + offset, 'r-', alpha=0.7, lw=1.5, label='VAE Reconstructed' if j == start else '')
        group_mse = mse_per_feat[start:end].mean().item()
        ax.set_title(f"{name} (MSE: {group_mse:.4f})", fontsize=9)
        ax.tick_params(labelsize=7)

    # Legend (only once, in the last subplot)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', fontsize=10)

    fig.suptitle(f"VAE Reconstruction - Sample {args.idx} | Total MSE: {total_mse:.6f}",
                 fontsize=12, fontweight='bold')

    # Auto-save to checkpoints if not specified
    save_path = args.save if args.save else f"checkpoints/vae_recon_{args.idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    main()
