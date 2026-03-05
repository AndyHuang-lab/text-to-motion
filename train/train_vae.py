"""
VAE Training Script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vae import VAE
from dataloader.data import SimpleMotionDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import json
import numpy as np


def _plot_loss_curves(history, save_dir):
    """
    Plot and save training loss curves.

    Args:
        history: Dictionary with loss history
        save_dir: Directory to save plots
    """
    epochs = range(1, len(history['train_total_loss']) + 1)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Total Loss
    axes[0].plot(epochs, history['train_total_loss'], 'b-', label='Train Total Loss', linewidth=2)
    axes[0].plot(epochs, history['val_total_loss'], 'r-', label='Val Total Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 2. Reconstruction Loss
    axes[1].plot(epochs, history['train_recon_loss'], 'b-', label='Train Recon Loss', linewidth=2)
    axes[1].plot(epochs, history['val_recon_loss'], 'r-', label='Val Recon Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # 3. KL Loss (with KL weight on secondary y-axis)
    ax3 = axes[2]
    ax3.plot(epochs, history['train_kl_loss'], 'b-', label='Train KL Loss', linewidth=2)
    ax3.plot(epochs, history['val_kl_loss'], 'r-', label='Val KL Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('KL Loss', fontsize=12, color='black')
    ax3.set_title('KL Loss & Weight', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Add KL weight on secondary y-axis
    ax3_right = ax3.twinx()
    ax3_right.plot(epochs, history['kl_weight'], 'g--', label='KL Weight', linewidth=2, alpha=0.7)
    ax3_right.set_ylabel('KL Weight', fontsize=12, color='green')
    ax3_right.tick_params(axis='y', labelcolor='green')
    ax3_right.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    # Save figure
    loss_curve_path = save_dir / 'vae_loss_curves.png'
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    print(f"  Saved loss curves to: {loss_curve_path}")

    # Also create a summary plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(epochs, history['train_total_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_total_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('VAE Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add best epoch marker
    best_epoch = np.argmin(history['val_total_loss']) + 1
    best_val = min(history['val_total_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(best_epoch, ax.get_ylim()[0], f' Best: epoch {best_epoch}\n val_loss={best_val:.4f}',
            fontsize=10, color='green', verticalalignment='bottom')

    summary_path = save_dir / 'vae_training_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"  Saved summary plot to: {summary_path}")

    plt.close('all')


def train_vae():
    # Config
    config = {
        'nfeats': 70,
        'embed_dim': 384,
        'n_head': 6,
        'd_ff': 1536,
        'num_layers': 4,
        'latent_dim': 128,
        'history_len': 10,
        'future_len': 20,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 300,
        'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        'kl_weight': 0.001,  # Target Beta for beta-VAE
        'kl_annealing_epochs': 50,  # Gradually increase KL weight over N epochs
        'save_dir': Path('checkpoints'),
        'log_interval': 10,  # Print every N batches
        'save_interval': 50,  # Save every N epochs
    }

    print("=" * 60)
    print("VAE Training Configuration")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Create save directory
    config['save_dir'].mkdir(exist_ok=True)

    # Create datasets
    print("\n[1/4] Creating datasets...")
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

    # Create model
    print("\n[2/4] Creating VAE model...")
    vae = VAE(
        nfeats=config['nfeats'],
        embed_dim=config['embed_dim'],
        n_head=config['n_head'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        latent_dim=config['latent_dim'],
    ).to(config['device'])

    # Count parameters
    total_params = sum(p.numel() for p in vae.parameters())
    trainable_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create optimizer
    print("\n[3/4] Creating optimizer...")
    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=config['lr'],
    )
    print(f"  Optimizer: Adam (lr={config['lr']})")

    # Training loop
    print("\n[4/4] Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')

    # Loss history for plotting
    loss_history = {
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_total_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'kl_weight': [],
    }

    for epoch in range(config['epochs']):
        # KL annealing: gradually increase KL weight
        if epoch < config['kl_annealing_epochs']:
            # Linear warmup: 0 → kl_weight
            current_kl_weight = config['kl_weight'] * (epoch + 1) / config['kl_annealing_epochs']
        else:
            # Use target kl_weight
            current_kl_weight = config['kl_weight']

        # Training
        vae.train()
        train_total_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0
        num_batches = 0

        for batch_idx, (history, future, _) in enumerate(train_loader):
            history = history.to(config['device'])  # (B, history_len, nfeats)
            future = future.to(config['device'])  # (B, future_len, nfeats)

            # Forward
            latent, dist = vae.encode(history, future)
            recon = vae.decode(latent, history, future_len=config['future_len'])

            # Loss
            recon_loss = F.mse_loss(recon, future)

            # KL divergence
            # dist.mean: (B, latent_dim), dist.stddev: (B, latent_dim)
            logvar = 2 * torch.log(dist.stddev)
            kl = -0.5 * torch.sum(1 + logvar - dist.mean.pow(2) - logvar.exp(), dim=1)
            kl_loss = kl.mean()

            loss = recon_loss + current_kl_weight * kl_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

            optimizer.step()

            # Track losses
            train_total_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            num_batches += 1

            # Print progress
            if (batch_idx + 1) % config['log_interval'] == 0:
                avg_loss = train_total_loss / num_batches
                avg_recon = train_recon_loss / num_batches
                avg_kl = train_kl_loss / num_batches
                print(f"  Epoch {epoch+1}/{config['epochs']} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.6f}) | "
                      f"KL_weight: {current_kl_weight:.6f}")

        # Epoch summary
        avg_train_loss = train_total_loss / num_batches
        avg_train_recon = train_recon_loss / num_batches
        avg_train_kl = train_kl_loss / num_batches

        print(f"\n  Train Epoch {epoch+1}/{config['epochs']} | "
              f"Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.6f})")

        # Validation
        vae.eval()
        val_total_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_num_batches = 0

        with torch.no_grad():
            for history, future, _ in val_loader:
                history = history.to(config['device'])
                future = future.to(config['device'])

                # Forward
                latent, dist = vae.encode(history, future)
                recon = vae.decode(latent, history, future_len=config['future_len'])

                # Loss
                recon_loss = F.mse_loss(recon, future)
                logvar = 2 * torch.log(dist.stddev)
                kl = -0.5 * torch.sum(1 + logvar - dist.mean.pow(2) - logvar.exp(), dim=1)
                kl_loss = kl.mean()
                loss = recon_loss + config['kl_weight'] * kl_loss

                val_total_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                val_num_batches += 1

        avg_val_loss = val_total_loss / val_num_batches
        avg_val_recon = val_recon_loss / val_num_batches
        avg_val_kl = val_kl_loss / val_num_batches

        print(f"  Val Epoch {epoch+1}/{config['epochs']} | "
              f"Loss: {avg_val_loss:.4f} (Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.6f})")
        print("=" * 60)

        # Record loss history
        loss_history['train_total_loss'].append(avg_train_loss)
        loss_history['train_recon_loss'].append(avg_train_recon)
        loss_history['train_kl_loss'].append(avg_train_kl)
        loss_history['val_total_loss'].append(avg_val_loss)
        loss_history['val_recon_loss'].append(avg_val_recon)
        loss_history['val_kl_loss'].append(avg_val_kl)
        loss_history['kl_weight'].append(current_kl_weight)

        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = config['save_dir'] / f'vae_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': loss_history,
                'config': config,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = config['save_dir'] / 'vae_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': loss_history,
                'config': config,
            }, best_path)
            print(f"  Saved best model: {best_path} (val_loss: {avg_val_loss:.4f})")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Save loss history
    history_path = config['save_dir'] / 'vae_training_history.json'
    with open(history_path, 'w') as f:
        # Convert to serializable format
        history_serializable = {k: [float(v) for v in vals] for k, vals in loss_history.items()}
        json.dump(history_serializable, f, indent=2)
    print(f"  Saved training history to: {history_path}")

    # Plot loss curves
    _plot_loss_curves(loss_history, config['save_dir'])


if __name__ == '__main__':
    train_vae()
