"""
Compute mean and std for motion features (70-dim with end_effector_pos)

This script calculates normalization statistics for the new 70-dimensional features.
Run this after updating the feature extraction.

Features (70-dim):
- root_height: 1 (absolute height p_z)
- root_rot_6d: 6 (continuous rotation representation)
- root_vel: 3 (v_x, v_y + yaw_dot)
- joint_pos: 23 (normalized DOF values)
- joint_vel: 23 (joint angular velocities)
- end_effector_pos: 12 (hand/foot positions relative to root)
- contact_mask: 2 (foot contact states)

Usage:
    python compute_statistics.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import torch
from pathlib import Path


def quat_to_6d(quat):
    """Convert quaternion (w,x,y,z) to 6D continuous rotation representation."""
    from scipy.spatial.transform import Rotation as sRot
    quat_xyzw = quat[:, [1, 2, 3, 0]]
    rot = sRot.from_quat(quat_xyzw)
    mat = rot.as_matrix()
    return mat[..., :2].reshape(quat.shape[0], -1)


def compute_velocity(data):
    """Compute velocity using central difference."""
    vel = np.zeros_like(data)
    vel[1:-1] = (data[2:] - data[:-2]) / 2
    vel[0] = data[1] - data[0]
    vel[-1] = data[-1] - data[-2]
    return vel


def compute_yaw_velocity(quat):
    """
    Compute yaw (rotation around Z-axis) velocity from quaternion.
    """
    from scipy.spatial.transform import Rotation as sRot
    quat_xyzw = quat[:, [1, 2, 3, 0]]
    rot = sRot.from_quat(quat_xyzw)
    euler = rot.as_euler('xyz')  # (T, 3) -> [roll, pitch, yaw]
    yaw = euler[:, 2]  # (T,)

    # Compute yaw velocity
    yaw_vel = np.zeros_like(yaw)
    yaw_vel[1:-1] = (yaw[2:] - yaw[:-2]) / 2
    yaw_vel[0] = yaw[1] - yaw[0]
    yaw_vel[-1] = yaw[-1] - yaw[-2]

    return yaw_vel


def extract_features_70d(motion_dict, start_frame, num_frames):
    """
    Extract 70-dimensional motion features with end_effector_pos.

    Features:
    - root_height: 1
    - root_rot_6d: 6
    - root_vel: 3 (v_x, v_y + yaw_dot)
    - joint_pos: 23
    - joint_vel: 23
    - end_effector_pos: 12
    - contact_mask: 2
    """
    # Extract raw data
    dof = motion_dict['dof'][start_frame:start_frame+num_frames].astype(np.float32)
    root_trans = motion_dict['root_trans_offset'][start_frame:start_frame+num_frames].astype(np.float32)
    root_rot_quat = motion_dict['root_rot'][start_frame:start_frame+num_frames].astype(np.float32)
    contact_mask = motion_dict['contact_mask'][start_frame:start_frame+num_frames].astype(np.float32)
    smpl_joints = motion_dict['smpl_joints'][start_frame:start_frame+num_frames].astype(np.float32)

    # 1. Root height: 1
    root_height = root_trans[:, 2:3]  # (T, 1)

    # 2. Root rotation 6D: 6
    root_rot_6d = quat_to_6d(root_rot_quat).astype(np.float32)

    # 3. Root velocity: 3 (v_x, v_y + yaw_dot)
    vel_root_xy = compute_velocity(root_trans[:, :2]).astype(np.float32)
    yaw_dot = compute_yaw_velocity(root_rot_quat).astype(np.float32)[:, None]
    root_vel = np.concatenate([vel_root_xy, yaw_dot], axis=1)

    # 4. Joint position: 23
    joint_pos = dof

    # 5. Joint velocity: 23
    joint_vel = compute_velocity(dof).astype(np.float32)

    # 6. End effector positions: 12 (relative to root)
    # Indices: left_foot=11, right_foot=27, left_hand=21, right_hand=22
    left_foot = smpl_joints[:, 11, :] - root_trans
    right_foot = smpl_joints[:, 27, :] - root_trans
    left_hand = smpl_joints[:, 21, :] - root_trans
    right_hand = smpl_joints[:, 22, :] - root_trans

    end_effector_pos = np.concatenate([
        left_foot, right_foot, left_hand, right_hand
    ], axis=1)

    # 7. Contact mask: 2
    # Already in correct format

    # Concatenate all features: 1 + 6 + 3 + 23 + 23 + 12 + 2 = 70
    features = np.concatenate([
        root_height, root_rot_6d, root_vel, joint_pos, joint_vel,
        end_effector_pos, contact_mask
    ], axis=1)

    return features


def compute_statistics(split='train', datadir='dataset'):
    """
    Compute mean and std for all features in the dataset.

    Args:
        split: 'train' or 'val'
        datadir: directory containing PKL files
    """
    print("="*60)
    print(f"Computing {split} statistics (70-dim features)")
    print("="*60)

    datadir = Path(datadir)
    pkl_path = datadir / f'{split}.pkl'

    print(f"\n[1/3] Loading data from {pkl_path}...")
    data = joblib.load(pkl_path)
    print(f"  Loaded {len(data)} sequences")

    # Collect all features
    print(f"\n[2/3] Extracting features...")
    all_features = []

    for i, seq in enumerate(data):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(data)} sequences...")

        motion_dict = seq['motion']
        num_frames = seq['length']

        # Extract features for entire sequence
        try:
            features = extract_features_70d(motion_dict, 0, num_frames)
            all_features.append(features)
        except Exception as e:
            print(f"\n  Warning: Failed to process sequence: {e}")
            continue

    # Concatenate all features
    print(f"\n  Concatenating features...")
    all_features = np.concatenate(all_features, axis=0)  # (total_frames, 70)

    print(f"  Total frames: {all_features.shape[0]}")
    print(f"  Feature dim: {all_features.shape[1]}")

    # Compute statistics
    print(f"\n[3/3] Computing mean and std...")

    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)

    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)

    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")

    # Save
    output_path = datadir / f'meanstd_70d.pkl'
    print(f"\n[4/4] Saving to {output_path}...")
    torch.save((torch.from_numpy(mean), torch.from_numpy(std)), output_path)
    print(f"  Saved!")

    # Print per-feature statistics
    print(f"\n{'='*60}")
    print("Feature Statistics (70 dimensions)")
    print(f"{'='*60}")

    feature_names = [
        "Root Height",
        "Root Rot 6D [0-5]",
        "Root Vel X",
        "Root Vel Y",
        "Root Vel Yaw",
        "Joint Pos [0-22]",
        "Joint Vel [0-22]",
        "Left Foot Pos [0-2]",
        "Right Foot Pos [0-2]",
        "Left Hand Pos [0-2]",
        "Right Hand Pos [0-2]",
        "Contact Mask Left",
        "Contact Mask Right",
    ]

    idx = 0
    for name in feature_names:
        if '[' in name:
            # Range of features
            parts = name.split('[')
            base_name = parts[0]
            range_str = parts[1].rstrip(']')
            if '-' in range_str:
                start, end = map(int, range_str.split('-'))
                count = end - start + 1
                print(f"  {base_name:<20} dims [{idx:2d}-{idx+count-1:2d}]: "
                      f"mean={np.mean(mean[idx:idx+count]):7.4f}, "
                      f"std={np.mean(std[idx:idx+count]):7.4f}")
                idx += count
            else:
                # Single feature index
                print(f"  {name:<20} dim  [{idx:2d}]: "
                      f"mean={mean[idx]:7.4f}, "
                      f"std={std[idx]:7.4f}")
                idx += 1
        else:
            # Single feature
            print(f"  {name:<20} dim  [{idx:2d}]: "
                  f"mean={mean[idx]:7.4f}, "
                  f"std={std[idx]:7.4f}")
            idx += 1

    print(f"{'='*60}\n")

    return mean, std


def main():
    """Compute statistics for train and val splits."""
    datadir = 'dataset'

    # Compute for train
    print("\n" + "="*60)
    print("COMPUTE TRAIN STATISTICS")
    print("="*60)
    train_mean, train_std = compute_statistics('train', datadir)

    # Copy to val (use train statistics for normalization)
    print("\n" + "="*60)
    print("COPY TRAIN STATISTICS TO VAL")
    print("="*60)
    val_path = Path(datadir) / 'meanstd_70d.pkl'
    print(f"  Using same statistics for val split")
    print(f"  Already saved to {val_path}")

    print("\n" + "="*60)
    print("✅ Statistics computed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Update dataloader to use meanstd_70d.pkl (already done)")
    print("  2. Update all model configs from nfeats=59 to nfeats=70")
    print("  3. Retrain VAE and Flow Matching models")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
