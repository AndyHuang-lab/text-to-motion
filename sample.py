"""
Motion Sampling Script with Flow Matching

Usage:
    python sample.py --text "A person walking forward" --steps 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import numpy as np
from pathlib import Path

from model.vae import VAE
from model.flow_matching import FlowMatchingTransformer


def load_text_embedding(_text, device):
    """Generate CLIP text embedding for the given text prompt."""
    # For now, return a random embedding
    # In production, you'd use the cached embeddings or CLIP model
    return torch.randn(512, device=device)


def sample_motion(
    flow_model,
    vae,
    history_motion,
    text_embedding,
    device,
    num_steps=10,
    guidance_scale=3.0
):
    """
    Sample motion using the trained flow matching model.

    Args:
        flow_model: Trained FlowMatchingTransformer
        vae: Trained VAE model
        history_motion: History motion [1, history_len, 70]
        text_embedding: CLIP text embedding [1, 512]
        device: torch device
        num_steps: Number of ODE solver steps (10-50 recommended)
        guidance_scale: Classifier-free guidance scale

    Returns:
        Generated future motion [1, future_len, 70]
    """
    batch_size = history_motion.shape[0]
    latent_dim = flow_model.latent_dim
    dt = 1.0 / num_steps

    # Start from noise
    x = torch.randn(batch_size, latent_dim, device=device)

    # Euler integration
    for i in range(num_steps):
        t = torch.full((batch_size,), i * dt, device=device)

        # Predict with and without conditioning
        v_cond = flow_model(
            x_t=x.unsqueeze(1),
            timesteps=t,
            history_motion=history_motion,
            text=text_embedding,
            all_mask=False
        ).squeeze(1)

        v_uncond = flow_model(
            x_t=x.unsqueeze(1),
            timesteps=t,
            history_motion=history_motion,
            text=text_embedding,
            all_mask=True
        ).squeeze(1)

        # Classifier-free guidance
        v = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Euler step
        x = x + dt * v

    # Decode to motion
    vae.eval()
    with torch.no_grad():
        future_motion = vae.decode(history_motion, x)

    return future_motion


def main():
    parser = argparse.ArgumentParser(description='Sample motion from trained model')
    parser.add_argument('--text', type=str, default='A person walking forward',
                        help='Text prompt for motion generation')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of sampling steps (10-50 recommended)')
    parser.add_argument('--guidance', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--vae_checkpoint', type=str, default='checkpoints/vae_best.pt',
                        help='Path to VAE checkpoint')
    parser.add_argument('--fm_checkpoint', type=str, default='checkpoints/fm_best.pt',
                        help='Path to Flow Matching checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu, mps)')
    parser.add_argument('--output', type=str, default='sampled_motion.npy',
                        help='Output path for sampled motion')

    args = parser.parse_args()

    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else
                             ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("Motion Sampling with Flow Matching")
    print("=" * 60)
    print(f"  Text prompt: {args.text}")
    print(f"  Sampling steps: {args.steps}")
    print(f"  Guidance scale: {args.guidance}")
    print(f"  Device: {device}")

    # Load models
    print("\n[1/4] Loading VAE...")
    vae = VAE(
        nfeats=70,
        embed_dim=384,
        n_head=6,
        d_ff=1536,
        num_layers=4,
        latent_dim=128,
    ).to(device)

    vae_checkpoint = Path(args.vae_checkpoint)
    if vae_checkpoint.exists():
        checkpoint = torch.load(vae_checkpoint, map_location=device, weights_only=False)
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded VAE from epoch {checkpoint['epoch']}")
    else:
        print(f"  WARNING: VAE checkpoint not found at {vae_checkpoint}")

    vae.eval()

    print("\n[2/4] Loading Flow Matching model...")
    flow_model = FlowMatchingTransformer(
        embed_dim=512,
        d_ff=2048,
        n_head=8,
        num_layers=6,
        clip_dim=512,
        history_dim=70,
        latent_dim=128,
        mask_prob=0.1,
    ).to(device)

    fm_checkpoint = Path(args.fm_checkpoint)
    if fm_checkpoint.exists():
        checkpoint = torch.load(fm_checkpoint, map_location=device, weights_only=False)
        flow_model.load_state_dict(checkpoint['flow_model_state_dict'])
        print(f"  Loaded Flow Matching model from epoch {checkpoint['epoch']}")
    else:
        print(f"  WARNING: Flow Matching checkpoint not found at {args.fm_checkpoint}")

    flow_model.eval()

    print("\n[3/4] Preparing inputs...")

    # For demo, use a random history motion
    # In practice, you'd load actual motion data
    history_motion = torch.randn(1, 10, 70, device=device)

    # Get text embedding
    text_embedding = load_text_embedding(args.text, device).unsqueeze(0)

    print(f"  History shape: {history_motion.shape}")
    print(f"  Text embedding shape: {text_embedding.shape}")

    # Sample motion
    print("\n[4/4] Sampling motion...")
    with torch.no_grad():
        future_motion = sample_motion(
            flow_model=flow_model,
            vae=vae,
            history_motion=history_motion,
            text_embedding=text_embedding,
            device=device,
            num_steps=args.steps,
            guidance_scale=args.guidance
        )

    print(f"  Generated motion shape: {future_motion.shape}")

    # Save output
    output_path = Path(args.output)
    np.save(output_path, future_motion.cpu().numpy())
    print(f"  Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Sampling complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
