"""
Motion Dataset for VAE/LDM/FM Training
"""

import joblib
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def quat_to_6d(quat):
    """
    Convert quaternion (w,x,y,z) to 6D continuous rotation representation.

    Args:
        quat: (N, 4) array in wxyz format

    Returns:
        (N, 6) array with first 2 columns of rotation matrix
    """
    from scipy.spatial.transform import Rotation as sRot

    # Convert wxyz to xyzw format for scipy
    quat_xyzw = quat[:, [1, 2, 3, 0]]
    rot = sRot.from_quat(quat_xyzw)
    mat = rot.as_matrix()

    # Take first 2 columns of rotation matrix and flatten
    return mat[..., :2].reshape(quat.shape[0], -1)  # (N, 6)


def compute_velocity(data):
    """
    Compute velocity using central difference.

    Args:
        data: (T, D) array

    Returns:
        (T, D) velocity array
    """
    vel = np.zeros_like(data)
    vel[1:-1] = (data[2:] - data[:-2]) / 2
    vel[0] = data[1] - data[0]
    vel[-1] = data[-1] - data[-2]
    return vel


def compute_yaw_velocity(quat):
    """
    Compute yaw (rotation around Z-axis) velocity from quaternion.

    Args:
        quat: (T, 4) array in wxyz format

    Returns:
        (T,) yaw velocity array
    """
    from scipy.spatial.transform import Rotation as sRot

    quat_xyzw = quat[:, [1, 2, 3, 0]]
    rot = sRot.from_quat(quat_xyzw)

    # Extract Euler angles (xyz convention)
    euler = rot.as_euler('xyz')  # (T, 3) -> [roll, pitch, yaw]
    yaw = euler[:, 2]  # (T,)

    # Compute yaw velocity
    yaw_vel = np.zeros_like(yaw)
    yaw_vel[1:-1] = (yaw[2:] - yaw[:-2]) / 2
    yaw_vel[0] = yaw[1] - yaw[0]
    yaw_vel[-1] = yaw[-1] - yaw[-2]

    return yaw_vel


class SimpleMotionDataset(Dataset):
    """
    Minimal dataset for VAE/LDM/FM training.
    Loads motion data, extracts features, handles normalization.
    """

    def __init__(
        self,
        datadir,
        split,
        history_len=10,
        future_len=20,
    ):
        super().__init__()

        self.datadir = Path(datadir)
        self.split = split
        self.history_len = history_len
        self.future_len = future_len

        print(f"[SimpleMotionDataset] Loading {split} data...")

        # Load PKL files
        pkl_path = self.datadir / f'{split}.pkl'
        print(f"[SimpleMotionDataset] Loading {pkl_path}...")
        self.data = joblib.load(pkl_path)
        print(f"[SimpleMotionDataset] Loaded {len(self.data)} sequences")

        # Load mean/std for normalization (70-dim with end_effector_pos)
        meanstd_path = self.datadir / 'meanstd_70d.pkl'
        print(f"[SimpleMotionDataset] Loading {meanstd_path}...")
        self.mean, self.std = torch.load(meanstd_path, map_location='cpu')
        print(f"[SimpleMotionDataset] Mean/std shape: {self.mean.shape}, {self.std.shape}")

        # Load cached text embeddings
        text_embed_path = self.datadir / f'{split}_text_embed.pkl'
        print(f"[SimpleMotionDataset] Loading {text_embed_path}...")
        self.text_embeds = torch.load(text_embed_path, map_location='cpu')
        print(f"[SimpleMotionDataset] Loaded {len(self.text_embeds)} text embeddings")

        # Validate data (FIX: Check actual array length, not declared length)
        total_len = self.history_len + self.future_len
        self.valid_indices = []
        for i, item in enumerate(self.data):
            # Use actual array length instead of declared length
            actual_length = len(item['motion']['dof'])
            if actual_length >= total_len:
                self.valid_indices.append(i)

        print(f"[SimpleMotionDataset] Valid sequences: {len(self.valid_indices)}/{len(self.data)}")

    def extract_features(self, motion_dict, start_frame, num_frames):
        """
        Extract 70-dimensional motion features:
        - root_height: 1 (absolute height p_z)
        - root_rot_6d: 6 (continuous rotation representation)
        - root_vel: 3 (v_x, v_y horizontal velocity + yaw_dot angular velocity)
        - joint_pos: 23 (normalized joint angles/radians)
        - joint_vel: 23 (joint angular velocities)
        - end_effector_pos: 12 (left/right hand/foot positions relative to root)
        - contact_mask: 2 (foot contact states: 0/1)

        End effector indices (from smpl_joints):
        - left_foot: 11, right_foot: 27
        - left_hand: 21, right_hand: 22
        """
        # Extract raw data
        dof = motion_dict['dof'][start_frame:start_frame+num_frames].astype(np.float32)  # (T, 23)
        root_trans = motion_dict['root_trans_offset'][start_frame:start_frame+num_frames].astype(np.float32)  # (T, 3)
        root_rot_quat = motion_dict['root_rot'][start_frame:start_frame+num_frames].astype(np.float32)  # (T, 4)
        contact_mask = motion_dict['contact_mask'][start_frame:start_frame+num_frames].astype(np.float32)  # (T, 2)
        smpl_joints = motion_dict['smpl_joints'][start_frame:start_frame+num_frames].astype(np.float32)  # (T, 29, 3)

        # 1. Root height: 1
        root_height = root_trans[:, 2:3]  # (T, 1)

        # 2. Root rotation 6D: 6
        root_rot_6d = quat_to_6d(root_rot_quat).astype(np.float32)  # (T, 6)

        # 3. Root velocity: 3 (v_x, v_y + yaw_dot)
        vel_root_xy = compute_velocity(root_trans[:, :2]).astype(np.float32)  # (T, 2)
        yaw_dot = compute_yaw_velocity(root_rot_quat).astype(np.float32)[:, None]  # (T, 1)
        root_vel = np.concatenate([vel_root_xy, yaw_dot], axis=1)  # (T, 3)

        # 4. Joint position: 23 (normalized DOF values)
        joint_pos = dof  # (T, 23)

        # 5. Joint velocity: 23
        joint_vel = compute_velocity(dof).astype(np.float32)  # (T, 23)

        # 6. End effector positions: 12 (relative to root)
        # Extract hand/foot positions and make relative to root
        # Indices: left_foot=11, right_foot=27, left_hand=21, right_hand=22
        left_foot = smpl_joints[:, 11, :]  # (T, 3)
        right_foot = smpl_joints[:, 27, :]  # (T, 3)
        left_hand = smpl_joints[:, 21, :]  # (T, 3)
        right_hand = smpl_joints[:, 22, :]  # (T, 3)

        # Make relative to root
        left_foot_rel = left_foot - root_trans  # (T, 3)
        right_foot_rel = right_foot - root_trans  # (T, 3)
        left_hand_rel = left_hand - root_trans  # (T, 3)
        right_hand_rel = right_hand - root_trans  # (T, 3)

        end_effector_pos = np.concatenate([
            left_foot_rel, right_foot_rel, left_hand_rel, right_hand_rel
        ], axis=1)  # (T, 12)

        # 7. Contact mask: 2
        # Already in correct format

        # Concatenate all features: 1 + 6 + 3 + 23 + 23 + 12 + 2 = 70
        features = np.concatenate([
            root_height, root_rot_6d, root_vel, joint_pos, joint_vel,
            end_effector_pos, contact_mask
        ], axis=1)  # (T, 70)

        return torch.from_numpy(features)

    def get_text_embedding(self, seq, start_frame):
        """
        Get text embedding for a segment.
        Uses the first annotation that overlaps with the future segment.
        """
        future_start = start_frame + self.history_len
        future_end = start_frame + self.history_len + self.future_len
        fps = 50  # From dataset statistics

        # Find overlapping annotations
        for ann in seq['frame_ann']:
            ann_start = int(ann[0] * fps)
            ann_end = int(ann[1] * fps)
            # Check if annotation overlaps with future segment
            if not (ann_end <= future_start or ann_start >= future_end):
                text = ann[2]
                # Return cached embedding or zeros if not found
                return self.text_embeds.get(text, torch.zeros(512))

        # Default to zero embedding if no text found
        return torch.zeros(512)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get valid sequence
        seq_idx = self.valid_indices[idx]
        seq = self.data[seq_idx]

        # Use actual array length instead of declared length
        motion = seq['motion']
        actual_length = len(motion['dof'])

        # Random sample start frame
        total_len = self.history_len + self.future_len
        max_start = actual_length - total_len
        start = random.randint(0, max_start)

        # Extract features separately for history and future to ensure correct shapes
        history = self.extract_features(motion, start, self.history_len)  # (history_len, 70)
        future = self.extract_features(motion, start + self.history_len, self.future_len)  # (future_len, 70)

        # Ensure correct shapes (debug check)
        assert history.shape == (self.history_len, 70), f"History shape mismatch: {history.shape}"
        assert future.shape == (self.future_len, 70), f"Future shape mismatch: {future.shape}"

        # Get text embedding for this segment
        text_emb = self.get_text_embedding(seq, start)

        # Normalize
        history = (history - self.mean) / self.std
        future = (future - self.mean) / self.std

        return history, future, text_emb
