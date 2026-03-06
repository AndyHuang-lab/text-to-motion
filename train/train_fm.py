"""
Flow Matching Training Script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vae import VAE
from model.flow_matching import FlowMatchingTransformer
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
    Plot and save training loss curves for Flow Matching.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    ax = plt.subplots(1, 1, figsize=(10, 6))[1]

    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Flow Matching Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add best epoch marker
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(best_epoch, ax.get_ylim()[0], f' Best: epoch {best_epoch}\n val_loss={best_val:.4f}',
            fontsize=10, color='green', verticalalignment='bottom')

    plt.tight_layout()

    # Save figure
    loss_curve_path = save_dir / 'fm_loss_curves.png'
    plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
    print(f"  Saved loss curves to: {loss_curve_path}")

    plt.close()


def train_flow_matching():
    # Config
    config = {
        'latent_dim': 128,
        'embed_dim': 512,
        'n_head': 8,
        'd_ff': 2048,
        'num_layers': 6,
        'clip_dim': 512,
        'history_dim': 70,
        'history_len': 10,
        'future_len': 20,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 200,
        'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        'vae_checkpoint': 'checkpoints/vae_best.pt',
        'save_dir': Path('checkpoints'),
        'log_interval': 10,
        'save_interval': 50,
        'mask_prob': 0.1,  # For classifier-free guidance
        'num_steps': 10,  # ODE solver steps for sampling/evaluation
    }

    print("=" * 60)
    print("Flow Matching Training Configuration")
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

    # Load pretrained VAE
    print(f"\n[2/4] Loading pretrained VAE from {config['vae_checkpoint']}...")
    vae = VAE(
        nfeats=70,
        embed_dim=384,
        n_head=6,
        d_ff=1536,
        num_layers=4,
        latent_dim=128,
    ).to(config['device'])

    checkpoint_path = Path(config['vae_checkpoint'])
    if not checkpoint_path.exists():
        print(f"  WARNING: VAE checkpoint not found at {checkpoint_path}")
        print(f"  Please train VAE first using train_vae.py")
        print(f"  Using untrained VAE for now...")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=config['device'], weights_only=False)
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded VAE from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")

    vae.eval()

    # Create flow matching model
    print("\n[3/4] Creating Flow Matching model...")
    flow_model = FlowMatchingTransformer(
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
    total_params = sum(p.numel() for p in flow_model.parameters())
    trainable_params = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create optimizer
    print("\n[4/4] Creating optimizer...")
    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=config['lr'])
    print(f"  Optimizer: AdamW (lr={config['lr']})")

    # Training loop
    print("\nStarting training...")
    print("=" * 60)

    best_val_loss = float('inf')

    # Loss history for plotting
    loss_history = {
        'train_loss': [],
        'val_loss': [],
    }

    for epoch in range(config['epochs']):
        # Training
        flow_model.train()
        train_total_loss = 0
        num_batches = 0

        for history_motion, future, text_emb in train_loader:
            history_motion = history_motion.to(config['device'])
            future = future.to(config['device'])
            text_emb = text_emb.to(config['device'])
            batch_size = history_motion.shape[0]

            with torch.no_grad():
                # Encode future to latent using VAE
                latent_gt, _ = vae.encode(history_motion, future)
                x_1 = latent_gt

            # Sample random timesteps in [0, 1]
            t = torch.rand(batch_size, device=config['device'])

            # Sample noise (starting point, t=0)
            x_0 = torch.randn_like(x_1)

            # Linear interpolation: x_t = (1 - t) * x_0 + t * x_1
            t_expanded = t.view(batch_size, 1)
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            x_t = x_t.unsqueeze(1)  # (B, 1, latent_dim)

            # Predict target latent z (x_1)
            pred_z = flow_model(
                x_t=x_t,
                timesteps=t,
                history_motion=history_motion,
                text=text_emb,
                all_mask=False
            )
            loss = F.mse_loss(pred_z, x_1.unsqueeze(1))

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track loss
            train_total_loss += loss.item()
            num_batches += 1

            # Print progress
            if num_batches % config['log_interval'] == 0:
                avg_loss = train_total_loss / num_batches
                print(f"  Epoch {epoch+1}/{config['epochs']} | Batches {num_batches}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f}")

        # Epoch summary
        avg_train_loss = train_total_loss / num_batches

        # Validation
        flow_model.eval()
        val_total_loss = 0
        val_num_batches = 0

        with torch.no_grad():
            for history_motion, future, text_emb in val_loader:
                history_motion = history_motion.to(config['device'])
                future = future.to(config['device'])
                text_emb = text_emb.to(config['device'])
                batch_size = history_motion.shape[0]

                # Encode future to latent
                latent_gt, _ = vae.encode(history_motion, future)
                x_1 = latent_gt

                # Sample timesteps
                t = torch.rand(batch_size, device=config['device'])

                # Sample noise
                x_0 = torch.randn_like(x_1)

                # Interpolate
                t_expanded = t.view(batch_size, 1)
                x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
                x_t = x_t.unsqueeze(1)

                pred_z = flow_model(
                    x_t=x_t,
                    timesteps=t,
                    history_motion=history_motion,
                    text=text_emb,
                    all_mask=False
                )
                loss = F.mse_loss(pred_z, x_1.unsqueeze(1))

                val_total_loss += loss.item()
                val_num_batches += 1

        avg_val_loss = val_total_loss / val_num_batches
        print(f"\n  Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print("=" * 60)

        # Record loss history
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['val_loss'].append(avg_val_loss)

        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = config['save_dir'] / f'fm_epoch_{epoch+1}.pt'
            checkpoint = {
                'epoch': epoch,
                'flow_model_state_dict': flow_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': loss_history,
                'config': config,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = config['save_dir'] / 'fm_best.pt'
            checkpoint = {
                'epoch': epoch,
                'flow_model_state_dict': flow_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': loss_history,
                'config': config,
            }
            torch.save(checkpoint, best_path)
            print(f"  Saved best model: {best_path} (val_loss: {avg_val_loss:.4f})")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Save loss history
    history_path = config['save_dir'] / 'fm_training_history.json'
    with open(history_path, 'w') as f:
        history_serializable = {k: [float(v) for v in vals] for k, vals in loss_history.items()}
        json.dump(history_serializable, f, indent=2)
    print(f"  Saved training history to: {history_path}")

    # Plot loss curves
    _plot_loss_curves(loss_history, config['save_dir'])


if __name__ == '__main__':
    train_flow_matching()
