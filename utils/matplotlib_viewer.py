"""
Matplotlib 3D Skeleton Viewer for G1 Humanoid Robot

Visualizes 59-dim motion features as a 3D skeleton using matplotlib.
Does NOT require MuJoCo - works on any system with matplotlib.

Usage:
    python utils/matplotlib_viewer.py --data-path dataset/val.pkl
    python utils/matplotlib_viewer.py --checkpoint checkpoints/fm_best.pt --text "walk forward"
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from pathlib import Path
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataloader.data import SimpleMotionDataset
from utils.mujoco_utils import features_to_qpos
import torch


# G1 Skeleton definition (simplified forward kinematics)
# Joint offsets and hierarchy based on typical humanoid proportions
# Values are approximate - for visualization only

G1_JOINT_OFFSETS = {
    # Body offsets in meters (relative to parent)
    'pelvis': np.array([0, 0, 0]),
    'left_hip': np.array([0, 0.065, -0.103]),
    'left_knee': np.array([0, 0, -0.35]),
    'left_ankle': np.array([0, 0, -0.35]),
    'left_foot': np.array([0, 0, -0.05]),
    'right_hip': np.array([0, -0.065, -0.103]),
    'right_knee': np.array([0, 0, -0.35]),
    'right_ankle': np.array([0, 0, -0.35]),
    'right_foot': np.array([0, 0, -0.05]),
    'torso': np.array([0, 0, 0.054]),
    'head': np.array([0, 0, 0.2]),
    'left_shoulder': np.array([0, 0.1, 0.24]),
    'left_elbow': np.array([0.08, 0, 0]),
    'left_hand': np.array([0.1, 0, 0]),
    'right_shoulder': np.array([0, -0.1, 0.24]),
    'right_elbow': np.array([0.08, 0, 0]),
    'right_hand': np.array([0.1, 0, 0]),
}

# Skeleton tree (parent -> children)
G1_SKELETON_TREE = {
    'pelvis': ['left_hip', 'right_hip', 'torso'],
    'left_hip': ['left_knee'],
    'left_knee': ['left_ankle'],
    'left_ankle': ['left_foot'],
    'right_hip': ['right_knee'],
    'right_knee': ['right_ankle'],
    'right_ankle': ['right_foot'],
    'torso': ['head', 'left_shoulder', 'right_shoulder'],
    'left_shoulder': ['left_elbow'],
    'left_elbow': ['left_hand'],
    'right_shoulder': ['right_elbow'],
    'right_elbow': ['right_hand'],
    'head': [],
    'left_foot': [],
    'right_foot': [],
    'left_hand': [],
    'right_hand': [],
}

# DOF index mapping (23 total)
DOF_TO_JOINT = {
    0: 'left_hip_pitch',
    1: 'left_hip_roll',
    2: 'left_hip_yaw',
    3: 'left_knee',
    4: 'left_ankle_pitch',
    5: 'left_ankle_roll',
    6: 'right_hip_pitch',
    7: 'right_hip_roll',
    8: 'right_hip_yaw',
    9: 'right_knee',
    10: 'right_ankle_pitch',
    11: 'right_ankle_roll',
    12: 'waist_yaw',
    13: 'left_shoulder_pitch',
    14: 'left_shoulder_roll',
    15: 'left_shoulder_yaw',
    16: 'left_elbow',
    17: 'left_wrist_roll',
    18: 'right_shoulder_pitch',
    19: 'right_shoulder_roll',
    20: 'right_shoulder_yaw',
    21: 'right_elbow',
    22: 'right_wrist_roll',
}


def rotation_matrix_from_euler(pitch, roll, yaw):
    """Create rotation matrix from Euler angles (pitch, roll, yaw)."""
    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    # Rotation around Y (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    # Rotation around Z (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def compute_joint_positions(features_59d):
    """
    Compute 3D joint positions from 59-dim features.

    Args:
        features_59d: (T, 59) array

    Returns:
        joint_positions: Dict mapping joint names to (T, 3) arrays
    """
    features_59d = np.asarray(features_59d)
    if features_59d.ndim == 1:
        features_59d = features_59d[np.newaxis, :]

    T = features_59d.shape[0]

    # Extract components
    dof = features_59d[:, 0:23]
    root_trans = features_59d[:, 23:26]
    root_rot_6d = features_59d[:, 26:32]

    # Convert 6D rotation to rotation matrix
    from scipy.spatial.transform import Rotation as sRot

    # Initialize joint positions
    positions = {joint: np.zeros((T, 3)) for joint in G1_SKELETON_TREE.keys()}

    # For each frame
    for t in range(T):
        # Root position
        root_pos = root_trans[t]

        # Root rotation (6D to matrix)
        b1 = root_rot_6d[t, :3]
        b1 = b1 / np.linalg.norm(b1)
        b2 = root_rot_6d[t, 3:]
        b2 = b2 - np.sum(b1 * b2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        root_rot = np.stack([b1, b2, b3], axis=1)

        # Compute positions using simplified FK
        # This is a simplified version - real FK would use proper joint chains

        # Pelvis (root)
        positions['pelvis'][t] = root_pos

        # Legs (simplified - using offsets with joint angles)
        angles = dof[t]

        # Left leg
        left_hip_pos = root_pos + root_rot @ np.array([0, 0.065, -0.103])
        positions['left_hip'][t] = left_hip_pos

        left_knee_pos = left_hip_pos + root_rot @ np.array([
            0.08 * np.sin(angles[3]),  # knee bend
            0.02 * np.sin(angles[1]),  # hip roll influence
            -0.35
        ])
        positions['left_knee'][t] = left_knee_pos

        left_ankle_pos = left_knee_pos + root_rot @ np.array([
            0.05 * np.sin(angles[3]),
            0.02 * np.sin(angles[5]),  # ankle roll
            -0.35
        ])
        positions['left_ankle'][t] = left_ankle_pos

        positions['left_foot'][t] = left_ankle_pos + root_rot @ np.array([0, 0, -0.05])

        # Right leg
        right_hip_pos = root_pos + root_rot @ np.array([0, -0.065, -0.103])
        positions['right_hip'][t] = right_hip_pos

        right_knee_pos = right_hip_pos + root_rot @ np.array([
            0.08 * np.sin(angles[9]),  # knee bend
            0.02 * np.sin(angles[7]),  # hip roll influence
            -0.35
        ])
        positions['right_knee'][t] = right_knee_pos

        right_ankle_pos = right_knee_pos + root_rot @ np.array([
            0.05 * np.sin(angles[9]),
            0.02 * np.sin(angles[11]),  # ankle roll
            -0.35
        ])
        positions['right_ankle'][t] = right_ankle_pos

        positions['right_foot'][t] = right_ankle_pos + root_rot @ np.array([0, 0, -0.05])

        # Torso
        torso_pos = root_pos + root_rot @ np.array([0, 0, 0.054])
        positions['torso'][t] = torso_pos

        # Head
        positions['head'][t] = torso_pos + root_rot @ np.array([0, 0, 0.2])

        # Left arm
        left_shoulder_pos = torso_pos + root_rot @ np.array([0, 0.1, 0.24])
        positions['left_shoulder'][t] = left_shoulder_pos

        # Elbow position (simplified)
        elbow_offset = np.array([
            0.08 * np.cos(angles[13]),  # shoulder pitch
            0.08 * np.sin(angles[14]),  # shoulder roll
            -0.05
        ])
        left_elbow_pos = left_shoulder_pos + root_rot @ elbow_offset
        positions['left_elbow'][t] = left_elbow_pos

        positions['left_hand'][t] = left_elbow_pos + root_rot @ np.array([0.1, 0, 0])

        # Right arm
        right_shoulder_pos = torso_pos + root_rot @ np.array([0, -0.1, 0.24])
        positions['right_shoulder'][t] = right_shoulder_pos

        elbow_offset = np.array([
            0.08 * np.cos(angles[18]),  # shoulder pitch
            0.08 * np.sin(angles[19]),  # shoulder roll
            -0.05
        ])
        right_elbow_pos = right_shoulder_pos + root_rot @ elbow_offset
        positions['right_elbow'][t] = right_elbow_pos

        positions['right_hand'][t] = right_elbow_pos + root_rot @ np.array([0.1, 0, 0])

    return positions


def plot_skeleton_3d(positions, frame_idx=0, ax=None, title="G1 Skeleton"):
    """
    Plot a single frame of the skeleton.

    Args:
        positions: Dict from compute_joint_positions
        frame_idx: Which frame to plot
        ax: Optional matplotlib axis
        title: Plot title
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Get joint positions for this frame
    frame_positions = {j: pos[frame_idx] for j, pos in positions.items()}

    # Plot bones (edges in skeleton tree)
    for parent, children in G1_SKELETON_TREE.items():
        parent_pos = frame_positions[parent]
        for child in children:
            child_pos = frame_positions[child]
            ax.plot3D(
                [parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                [parent_pos[2], child_pos[2]],
                'b-', linewidth=2
            )

    # Plot joints
    for joint, pos in frame_positions.items():
        color = 'r' if joint == 'pelvis' else 'orange'
        size = 50 if joint == 'pelvis' else 20
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=size)

    # Set equal aspect ratio
    all_pos = np.array(list(frame_positions.values()))
    max_range = np.array([
        all_pos[:, 0].max() - all_pos[:, 0].min(),
        all_pos[:, 1].max() - all_pos[:, 1].min(),
        all_pos[:, 2].max() - all_pos[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_pos[:, 0].max() + all_pos[:, 0].min()) * 0.5
    mid_y = (all_pos[:, 1].max() + all_pos[:, 1].min()) * 0.5
    mid_z = (all_pos[:, 2].max() + all_pos[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    return ax


def visualize_motion(features_59d, save_path=None, fps=50):
    """
    Visualize motion sequence as 3D skeleton animation.

    Args:
        features_59d: (T, 59) array
        save_path: Optional path to save animation
        fps: Frames per second
    """
    print("Computing joint positions...")
    positions = compute_joint_positions(features_59d)
    T = features_59d.shape[0]

    print(f"Creating animation with {T} frames...")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        plot_skeleton_3d(positions, frame_idx=frame, ax=ax,
                        title=f"G1 Skeleton - Frame {frame}/{T-1}")
        return ax,

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=1000/fps, blit=False
    )

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps)
        else:
            ani.save(save_path + '.gif', writer='pillow', fps=fps)

    plt.show()
    return ani


def visualize_sample_from_dataset(dataset_path='dataset/val.pkl', idx=0, save_gif=None):
    """Visualize a sample from the dataset."""
    print(f"Loading dataset from {dataset_path}...")

    datadir = str(Path(dataset_path).parent)
    split = Path(dataset_path).stem

    dataset = SimpleMotionDataset(
        datadir=datadir,
        split=split,
        history_len=10,
        future_len=20,
    )

    print(f"Dataset loaded: {len(dataset)} samples")

    # Get a sample
    history, future, text_emb = dataset[idx]

    # Denormalize
    mean, std = dataset.mean, dataset.std
    history = history * std + mean
    future = future * std + mean

    # Combine history and future
    motion = torch.cat([history, future], dim=0).numpy()

    print(f"Motion shape: {motion.shape}")

    # Visualize
    visualize_motion(motion, save_path=save_gif)


def main():
    parser = argparse.ArgumentParser(description='Visualize G1 motion with matplotlib')
    parser.add_argument('--data-path', type=str, default='dataset/val.pkl',
                        help='Path to dataset pickle file')
    parser.add_argument('--idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--save-gif', type=str, default=None,
                        help='Path to save GIF animation')
    parser.add_argument('--fps', type=int, default=50,
                        help='Frames per second for animation')

    args = parser.parse_args()

    visualize_sample_from_dataset(
        dataset_path=args.data_path,
        idx=args.idx,
        save_gif=args.save_gif
    )


if __name__ == '__main__':
    main()
