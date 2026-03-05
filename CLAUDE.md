# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Text-conditioned motion generation for Unitree G1 humanoid robot using **VAE + Generative Model**.

- **23-DOF** humanoid, **70-dim** features (includes end_effector_pos)
- CLIP text conditioning (512-dim)
- Target: Real-time generation (<100ms)

## Architecture

Two-stage generation pipeline:

1. **VAE** (`model/vae.py`): Compresses 70-dim motion sequences to 128-dim latent vectors
2. **Generative Model**: Either **Flow Matching** (fast) or **LDM/DDPM** (baseline)

**Data Flow**: Raw Motion → 70-dim Features → VAE encode → Generative Model → VAE decode → Motion

| Model | Sampling | Steps | Speed |
|-------|----------|-------|-------|
| Flow Matching | ODE Euler | 10 | Fastest |
| LDM (DDPM) | Iterative denoising | 10 | Baseline |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Compute feature statistics (first time only)
python -m dataloader.compute_statistics

# 2. Train VAE (required first step)
python -m train.train_vae

# 3a. Train Flow Matching (recommended, fast)
python -m train.train_fm

# 3b. OR Train LDM/DDPM (baseline)
python -m train.train_ldm

# 4. Evaluate generation quality
python eval/evaluate_fm.py --steps 10   # Flow Matching
python eval/evaluate_ldm.py --steps 10  # LDM/DDPM
```

## Model Config

```python
VAE(nfeats=70, latent_dim=128, embed_dim=384, n_head=6, num_layers=4)
FlowMatchingTransformer(embed_dim=512, clip_dim=512, history_dim=70, latent_dim=128)
DenoiserTransformer(embed_dim=512, clip_dim=512, history_dim=70, noise_dim=128)  # LDM
```

## Feature Breakdown (70-dim)

```python
[0:1]    Root height (absolute Z position)
[1:7]    Root rotation 6D (continuous rotation from quaternion)
[7:10]   Root velocity (v_x, v_y, yaw_dot)
[10:33]  Joint positions (normalized DOF values)
[33:56]  Joint velocities (angular velocities)
[56:59]  Left foot position (relative to root, x,y,z)
[59:62]  Right foot position (relative to root, x,y,z)
[62:65]  Left hand position (relative to root, x,y,z)
[65:68]  Right hand position (relative to root, x,y,z)
[68:70]  Contact mask (LEFT, RIGHT)
```

## Shape Conventions

- Motion features: `(batch, seq_len, 70)` - seq_len: 10 (history) or 20 (future)
- VAE latent: `(batch, 128)`
- Text embedding: `(batch, 512)` - CLIP-encoded, pre-computed and cached
- Flow Matching input: `(batch, 128)` (no seq dim)
- LDM input: `(batch, 1, 128)` (with seq dim)

## File Structure

```
model/
├── vae.py              # VAE (70-dim → 128-dim latent)
├── flow_matching.py    # Flow Matching (recommended, fast)
└── ldm.py              # LDM/DDPM (baseline)

dataloader/
├── data.py             # 70-dim feature extraction, normalization
└── compute_statistics.py  # Compute dataset mean/std

train/
├── train_vae.py        # Train VAE (required first)
├── train_fm.py         # Train Flow Matching (num_steps=10)
└── train_ldm.py        # Train LDM/DDPM (num_timesteps=10)

eval/
├── evaluate_fm.py      # Evaluate Flow Matching (MSE + Diversity)
└── evaluate_ldm.py     # Evaluate LDM/DDPM (MSE + Diversity)

dataset/
├── train.pkl           # Motion data (4.9GB)
├── val.pkl             # Validation data (1.8GB)
├── train_text_embed.pkl # CLIP text embeddings (cached)
├── val_text_embed.pkl
└── meanstd_70d.pkl     # Normalization stats (REQUIRED)
```

## Training Configuration

All models use:
- Batch size: 32
- Learning rate: 1e-4
- Device auto-detection: CUDA > MPS > CPU
- Classifier-free guidance (mask_prob=0.1)

| Model | Epochs | Timesteps/Steps | Save Interval |
|-------|--------|-----------------|---------------|
| VAE | 100 | - | 10 |
| Flow Matching | 200 | 10 (sampling) | 50 |
| LDM/DDPM | 100 | 10 (training + sampling) | 10 (best only) |

## Evaluation Metrics

```bash
# Both evaluators output:
# - Conditional MSE: Generated vs GT motion (lower = better)
# - Diversity (L2): Mean pairwise distance (higher = more diverse)

python eval/evaluate_fm.py --batches 20 --steps 10
python eval/evaluate_ldm.py --batches 20 --steps 10
```

## Device Support

```python
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
```

## Critical Gotchas

1. **70-dim features** - Feature representation includes end_effector_pos
2. **Normalization**: Must use `meanstd_70d.pkl` - run `python -m dataloader.compute_statistics` first
3. **Train VAE first** - All models require VAE checkpoint at `checkpoints/vae_best.pt`
4. **Text embeddings**: Pre-computed and cached in `dataset/*_text_embed.pkl`
5. **Timesteps must match**: If training with `num_timesteps=10`, sampling must also use 10
6. **End effector joints**: From smpl_joints - left_foot=11, right_foot=27, left_hand=21, right_hand=22

## Dataset

- **Source**: BABEL-AMASS-ROBOT dataset
- **FPS**: 50
- **Format**: PKL files with motion dict (dof, root_trans_offset, root_rot, smpl_joints, contact_mask, frame_ann)
- **Text annotations**: Stored in `frame_ann` as (start_time, end_time, text_desc)

## Dependencies

```bash
torch>=2.0.0
numpy
joblib
scipy
matplotlib
```

For CLIP text encoding (if generating new embeddings):
```bash
pip install git+https://github.com/openai/CLIP.git
```
