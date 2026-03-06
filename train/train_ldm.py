"""
LDM Training Script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vae import VAE
from model.ldm import DenoiserTransformer
from dataloader.data import SimpleMotionDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt


def get_schedule(num_timesteps=10):
    """Simple linear diffusion schedule"""
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def train_ldm():
    # Config
    config = {
        'latent_dim': 128,  # VAE latent dim
        'embed_dim': 512,
        'n_head': 8,
        'd_ff': 2048,
        'num_layers': 6,
        'clip_dim': 512,
        'history_dim': 70,  # nfeats (per-timestep feature dimension with end_effector_pos)
        'num_timesteps': 10,
        'history_len': 10,
        'future_len': 20,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'vae_checkpoint': 'checkpoints/vae_best.pt',
        'save_dir': Path('checkpoints'),
        'log_interval': 10,  # Print every N batches
        'mask_prob': 0.1,  # For classifier-free guidance
    }

    print("=" * 60)
    print("LDM Training Configuration")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Create save directory
    config['save_dir'].mkdir(exist_ok=True)

    # Create datasets
    print("\n[1/5] Creating datasets...")
    train_dataset = SimpleMotionDataset(
        datadir='dataset',
        split='train',
        history_len=config['history_len'],
        future_len=config['future_len'],
    )
    val_dataset = SimpleMotionDataset(
        datadir='dataset',
        split='val',
        history_len=config['history_len'],
        future_len=config['future_len'],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False,
    )

    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Val dataset: {len(val_dataset)} samples")
    print(f"  Device: {config['device']}")

    # Load pretrained VAE
    print(f"\n[2/5] Loading pretrained VAE from {config['vae_checkpoint']}...")
    vae = VAE(
        nfeats=70,  # Updated to include end_effector_pos
        embed_dim=384,
        n_head=6,
        d_ff=1536,
        num_layers=4,
        latent_dim=config['latent_dim'],
    ).to(config['device'])

    checkpoint_path = Path(config['vae_checkpoint'])
    if not checkpoint_path.exists():
        print(f"  WARNING: VAE checkpoint not found at {checkpoint_path}")
        print(f"  Please train VAE first using train_vae_simple.py")
        print(f"  Using untrained VAE for now...")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=config['device'], weights_only=False)
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded VAE from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")

    vae.eval()

    # Create denoiser
    print("\n[3/5] Creating denoiser model...")
    denoiser = DenoiserTransformer(
        embed_dim=config['embed_dim'],
        d_ff=config['d_ff'],
        n_head=config['n_head'],
        num_layers=config['num_layers'],
        dropout=0.1,
        activation='gelu',
        clip_dim=config['clip_dim'],
        history_dim=config['history_dim'],
        latent_dim=config['latent_dim'],
        mask_prob=config['mask_prob'],
    ).to(config['device'])

    # Count parameters
    total_params = sum(p.numel() for p in denoiser.parameters())
    trainable_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create optimizer
    print("\n[4/5] Creating optimizer...")
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=config['lr'])
    print(f"  Optimizer: AdamW (lr={config['lr']})")

    # Diffusion schedule
    print("\n[5/5] Creating diffusion schedule...")
    betas, alphas_cumprod = get_schedule(config['num_timesteps'])
    print(f"  Timesteps: {config['num_timesteps']}")
    print(f"  Beta range: [{betas[0]:.6f}, {betas[-1]:.6f}]")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(config['epochs']):
        # Training
        denoiser.train()
        train_total_loss = 0
        num_batches = 0

        for batch_idx, (history, future, text_emb) in enumerate(train_loader):
            history = history.to(config['device'])  # (B, history_len, nfeats)
            future = future.to(config['device'])  # (B, future_len, nfeats)
            text_emb = text_emb.to(config['device'])  # (B, 512)
            batch_size = history.shape[0]

            with torch.no_grad():
                # Encode future to latent using VAE
                latent_gt, _ = vae.encode(history, future)  # (B, latent_dim)
                x_start = latent_gt.unsqueeze(1)  # (B, 1, latent_dim)

            # Sample random timesteps
            t = torch.randint(0, config['num_timesteps'], (batch_size,), device=config['device'])

            # Add noise
            noise = torch.randn_like(x_start)
            alpha_t = alphas_cumprod[t].view(batch_size, 1, 1)
            x_t = torch.sqrt(alpha_t) * x_start + torch.sqrt(1 - alpha_t) * noise

            # Predict latent (x0)
            pred_latent = denoiser(
                x_t=x_t,
                timesteps=t,
                history_motion=history,
                text=text_emb,
                all_mask=False  # Don't mask during training (except when mask_prob applies internally)
            )  # (B, 1, latent_dim)

            loss = F.mse_loss(pred_latent, x_start)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)

            optimizer.step()

            # Track loss
            train_total_loss += loss.item()
            num_batches += 1

            # Print progress
            if (batch_idx + 1) % config['log_interval'] == 0:
                avg_loss = train_total_loss / num_batches
                print(f"  Epoch {epoch+1}/{config['epochs']} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f}")

        # Epoch summary
        avg_train_loss = train_total_loss / num_batches
        print(f"\n  Train Epoch {epoch+1}/{config['epochs']} | Loss: {avg_train_loss:.4f}")

        # Validation
        denoiser.eval()
        val_total_loss = 0
        val_num_batches = 0

        with torch.no_grad():
            for history, future, text_emb in val_loader:
                history = history.to(config['device'])
                future = future.to(config['device'])
                text_emb = text_emb.to(config['device'])
                batch_size = history.shape[0]

                # Encode future to latent
                latent_gt, _ = vae.encode(history, future)
                x_start = latent_gt.unsqueeze(1)

                # Sample timesteps
                t = torch.randint(0, config['num_timesteps'], (batch_size,), device=config['device'])

                # Add noise
                noise = torch.randn_like(x_start)
                alpha_t = alphas_cumprod[t].view(batch_size, 1, 1)
                x_t = torch.sqrt(alpha_t) * x_start + torch.sqrt(1 - alpha_t) * noise

                pred_latent = denoiser(
                    x_t=x_t,
                    timesteps=t,
                    history_motion=history,
                    text=text_emb,
                    all_mask=False
                )
                loss = F.mse_loss(pred_latent, x_start)

                val_total_loss += loss.item()
                val_num_batches += 1

        avg_val_loss = val_total_loss / val_num_batches
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"  Val Epoch {epoch+1}/{config['epochs']} | Loss: {avg_val_loss:.4f}")
        print("=" * 60)

        # Save best model only
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = config['save_dir'] / 'ldm_best.pt'
            torch.save({
                'epoch': epoch,
                'denoiser_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
            }, best_path)
            print(f"  Saved best model: {best_path} (val_loss: {avg_val_loss:.4f})")

    # Save loss curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, 'b-', label='Train', linewidth=2)
    ax.plot(epochs_range, val_losses, 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('LDM Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_curve_path = config['save_dir'] / 'ldm_loss_curves.png'
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved loss curves: {loss_curve_path}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    train_ldm()
