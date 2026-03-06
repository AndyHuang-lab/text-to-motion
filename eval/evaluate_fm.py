"""
Flow Matching Evaluation - VAE + FM Generation Quality

Metrics:
- Conditional MSE: Generated vs GT motion (lower is better)
- Diversity: Mean pairwise L2 between samples (higher = more diverse)

Usage:
    python eval/evaluate_fm.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from model.vae import VAE
from model.flow_matching import FlowMatchingTransformer
from dataloader.data import SimpleMotionDataset


@torch.no_grad()
def sample(flow_model, vae, history_motion, text_embedding, device,
           num_steps=10, guidance_scale=3.0):
    """Flow Matching sampling (Euler). Model predicts z; v = (pred_z - x)/(1-t). Returns: [B, future_len, 70]"""
    batch_size = history_motion.shape[0]
    latent_dim = flow_model.latent_dim
    future_len = 20
    dt = 1.0 / num_steps
    eps = 1e-5  # avoid div by zero when t -> 1

    x = torch.randn(batch_size, latent_dim, device=device)

    for i in range(num_steps):
        t = torch.full((batch_size,), i * dt, device=device)

        pred_z_cond = flow_model(x_t=x.unsqueeze(1), timesteps=t, history_motion=history_motion,
                                 text=text_embedding, all_mask=False).squeeze(1)
        pred_z_uncond = flow_model(x_t=x.unsqueeze(1), timesteps=t, history_motion=history_motion,
                                  text=text_embedding, all_mask=True).squeeze(1)
        pred_z = pred_z_uncond + guidance_scale * (pred_z_cond - pred_z_uncond)

        one_minus_t = (1.0 - t).clamp(min=eps).unsqueeze(-1)  # (B, 1) for broadcasting
        v = (pred_z - x) / one_minus_t
        x = x + dt * v

    vae.eval()
    return vae.decode(x, history_motion, future_len)


def mean_pairwise_l2(x):
    """x: (N, T, D). Mean pairwise L2 distance."""
    N = x.shape[0]
    if N < 2:
        return 0.0
    x_flat = x.reshape(N, -1)
    d = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            d += np.linalg.norm(x_flat[i] - x_flat[j])
            count += 1
    return d / count if count else 0.0


def main():
    parser = argparse.ArgumentParser(description='Flow Matching Evaluation (VAE + FM)')
    parser.add_argument('--vae', type=str, default='checkpoints/vae_best.pt')
    parser.add_argument('--fm', type=str, default='checkpoints/fm_best.pt')
    parser.add_argument('--datadir', type=str, default='dataset')
    parser.add_argument('--batches', type=int, default=50, help='Max val batches')
    parser.add_argument('--diversity_conditions', type=int, default=30)
    parser.add_argument('--diversity_samples', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--guidance', type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                         ('mps' if torch.backends.mps.is_available() else 'cpu'))

    print("=" * 60)
    print("Flow Matching Evaluation (VAE + FM)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Solver steps: {args.steps}")
    print(f"  Guidance: {args.guidance}")
    print(f"  Max batches: {args.batches}")

    # Load models
    print("\n[1/2] Loading models...")
    vae = VAE(nfeats=70, embed_dim=384, n_head=6, d_ff=1536,
              num_layers=4, latent_dim=128).to(device)
    ckpt = torch.load(args.vae, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt['model_state_dict'])
    vae.eval()
    print(f"  VAE: epoch {ckpt['epoch']}")

    flow_model = FlowMatchingTransformer(
        embed_dim=512, d_ff=2048, n_head=8, num_layers=6, dropout=0.1,
        activation='gelu', clip_dim=512, history_dim=70, latent_dim=128, mask_prob=0.1,
    ).to(device)
    ckpt = torch.load(args.fm, map_location=device, weights_only=False)
    flow_model.load_state_dict(ckpt['flow_model_state_dict'])
    flow_model.eval()
    print(f"  FM: epoch {ckpt['epoch']} (val_loss: {ckpt['val_loss']:.4f})")

    # Evaluation
    print("\n[2/2] Evaluating...")
    val_dataset = SimpleMotionDataset(datadir=args.datadir, split='val', history_len=10, future_len=20)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    mse_list, div_list = [], []

    with torch.no_grad():
        for batch_idx, (history, future_gt, text_emb) in enumerate(val_loader):
            if batch_idx >= args.batches:
                break

            history = history.to(device)
            future_gt = future_gt.to(device)
            text_emb = text_emb.to(device)
            B = history.shape[0]

            # Generate motion
            future_gen = sample(flow_model, vae, history, text_emb, device,
                               num_steps=args.steps, guidance_scale=args.guidance)

            # MSE
            mse = ((future_gen - future_gt) ** 2).mean(dim=(1, 2))
            mse_list.extend(mse.cpu().numpy().tolist())

            # Diversity
            if batch_idx < args.diversity_conditions:
                for i in range(min(B, 2)):
                    if len(div_list) >= args.diversity_conditions:
                        break
                    h = history[i:i+1].repeat(args.diversity_samples, 1, 1)
                    t = text_emb[i:i+1].repeat(args.diversity_samples, 1)
                    multi = sample(flow_model, vae, h, t, device,
                                 num_steps=args.steps, guidance_scale=args.guidance)
                    div_list.append(mean_pairwise_l2(multi.cpu().numpy()))

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{min(args.batches, len(val_loader))}")

    # Results
    mse_arr = np.array(mse_list)
    div_arr = np.array(div_list) if div_list else np.array([0.0])

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Conditional MSE:   {mse_arr.mean():.6f} ± {mse_arr.std():.6f}  (lower = better)")
    print(f"  Diversity (L2):     {div_arr.mean():.6f} ± {div_arr.std():.6f}  (higher = more diverse)")
    print("=" * 60)


if __name__ == '__main__':
    main()
