"""
LDM Evaluation - VAE + DDPM Generation Quality

Metrics:
- Conditional MSE: Generated vs GT motion (lower is better)
- Diversity: Mean pairwise L2 between samples (higher = more diverse)

Usage:
    python eval/evaluate_ldm.py
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
from model.ldm import DenoiserTransformer
from dataloader.data import SimpleMotionDataset


def get_schedule(num_timesteps=10):
    """Linear diffusion schedule (must match training)"""
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    return betas, alphas_cumprod, alphas_cumprod_prev


@torch.no_grad()
def sample(denoiser, vae, history_motion, text_embedding, device,
           num_timesteps=10, guidance_scale=3.0):
    """DDPM sampling. Returns: [B, future_len, 70]"""
    batch_size = history_motion.shape[0]
    latent_dim = denoiser.noise_dim
    future_len = 20

    betas, alphas_cumprod, alphas_cumprod_prev = get_schedule(num_timesteps)
    alphas = 1 - betas

    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    alphas_cumprod_prev = alphas_cumprod_prev.to(device)

    x = torch.randn(batch_size, 1, latent_dim, device=device)

    for t in reversed(range(num_timesteps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        noise_cond = denoiser(x_t=x, timesteps=t_batch, history_motion=history_motion,
                             text=text_embedding, all_mask=False)
        noise_uncond = denoiser(x_t=x, timesteps=t_batch, history_motion=history_motion,
                               text=text_embedding, all_mask=True)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        alpha_cumprod_prev_t = alphas_cumprod_prev[t]
        beta_t = betas[t]

        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -3, 3)
        mean = (alpha_cumprod_prev_t.sqrt() * beta_t * pred_x0 +
                (1 - alpha_cumprod_prev_t) * alpha_t.sqrt() * x) / (1 - alpha_cumprod_t)

        if t > 0:
            posterior_variance = beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)
            x = mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
        else:
            x = mean

    latent = x.squeeze(1)
    vae.eval()
    return vae.decode(latent, history_motion, future_len)


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
    parser = argparse.ArgumentParser(description='LDM Evaluation (VAE + DDPM)')
    parser.add_argument('--vae', type=str, default='checkpoints/vae_best.pt')
    parser.add_argument('--ldm', type=str, default='checkpoints/ldm_best.pt')
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
    print("LDM Evaluation (VAE + DDPM)")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Timesteps: {args.steps}")
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

    denoiser = DenoiserTransformer(
        embed_dim=512, d_ff=2048, n_head=8, num_layers=6, dropout=0.1,
        activation='gelu', clip_dim=512, history_dim=70, noise_dim=128, mask_prob=0.1,
    ).to(device)
    ckpt = torch.load(args.ldm, map_location=device, weights_only=False)
    denoiser.load_state_dict(ckpt['denoiser_state_dict'])
    denoiser.eval()
    print(f"  LDM: epoch {ckpt['epoch']} (val_loss: {ckpt['val_loss']:.4f})")

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
            future_gen = sample(denoiser, vae, history, text_emb, device,
                               num_timesteps=args.steps, guidance_scale=args.guidance)

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
                    multi = sample(denoiser, vae, h, t, device,
                                 num_timesteps=args.steps, guidance_scale=args.guidance)
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
