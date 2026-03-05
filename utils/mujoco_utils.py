"""
Utilities for converting between 59-dim motion features and MuJoCo format.

59-dim features:
    [0:23]   DOF joint angles
    [23:26]  Root translation (X, Y, Z)
    [26:32]  Root rotation 6D
    [32:55]  Joint velocities
    [55:57]  Root velocity XY
    [57:59]  Contact mask (LEFT, RIGHT)

MuJoCo QPOS (30-dim):
    [0:3]   Root position (X, Y, Z)
    [3:7]   Root quaternion (w, x, y, z)
    [7:13]  Left leg (6 joints)
    [13:19] Right leg (6 joints)
    [19]    Waist yaw
    [20:25] Left arm (5 joints)
    [25:30] Right arm (5 joints)
"""

import numpy as np
import torch


def quat_from_6d(rot_6d):
    """
    Convert 6D continuous rotation to quaternion.

    Args:
        rot_6d: (..., 6) array - first two columns of rotation matrix

    Returns:
        quat: (..., 4) array in wxyz format
    """
    from scipy.spatial.transform import Rotation as sRot

    shape = rot_6d.shape
    if len(shape) == 1:
        rot_6d = rot_6d.reshape(1, 6)

    # Reconstruct rotation matrix from 6D representation
    # 6D: [r00, r01, r02, r10, r11, r12]
    b1 = rot_6d[:, :3]  # First column
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)

    b2 = rot_6d[:, 3:]  # Second column
    b2 = b2 - np.sum(b1 * b2, axis=1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)

    b3 = np.cross(b1, b2)

    rot_mat = np.stack([b1, b2, b3], axis=2)  # (N, 3, 3)

    # Convert to quaternion
    rot = sRot.from_matrix(rot_mat)
    quat = rot.as_quat()  # (N, 4) in xyzw format

    # Convert to wxyz
    quat_wxyz = np.concatenate([quat[:, 3:4], quat[:, :3]], axis=1)

    if len(shape) == 1:
        quat_wxyz = quat_wxyz[0]

    return quat_wxyz


def features_to_qpos(features_59d):
    """
    Convert 59-dim motion features to MuJoCo qpos (30-dim).

    DOF order (23):
        Legs (12): left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
                   left_ankle_pitch, left_ankle_roll,
                   right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
                   right_ankle_pitch, right_ankle_roll
        Waist (1): waist_yaw
        Arms (10): left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
                   left_elbow, left_wrist_roll,
                   right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
                   right_elbow, right_wrist_roll

    Args:
        features_59d: (T, 59) array or (59,) array

    Returns:
        qpos: (T, 30) array or (30,) array
    """
    features_59d = np.asarray(features_59d)
    squeeze = False
    if features_59d.ndim == 1:
        features_59d = features_59d[np.newaxis, :]
        squeeze = True

    T = features_59d.shape[0]
    qpos = np.zeros((T, 30))

    # Extract components
    dof = features_59d[:, 0:23]       # Joint angles
    root_trans = features_59d[:, 23:26]  # Root translation
    root_rot_6d = features_59d[:, 26:32]  # Root rotation 6D

    # Build qpos
    # [0:3] Root position
    qpos[:, 0:3] = root_trans

    # [3:7] Root quaternion (convert 6D to quat)
    for t in range(T):
        quat_wxyz = quat_from_6d(root_rot_6d[t])
        qpos[t, 3:7] = quat_wxyz

    # [7:13] Left leg (6 joints) - indices 0-5 from dof
    qpos[:, 7:13] = dof[:, 0:6]

    # [13:19] Right leg (6 joints) - indices 6-11 from dof
    qpos[:, 13:19] = dof[:, 6:12]

    # [19] Waist yaw - index 18 from dof (after 12 leg joints, assuming arm joints are later)
    # Actually, need to check the exact order. Let me use a more flexible mapping:
    # Based on g1_23dof.xml joint order:
    # Legs first, then waist, then arms

    # Re-map DOF based on typical G1 order:
    # left_hip_pitch(0), left_hip_roll(1), left_hip_yaw(2), left_knee(3),
    # left_ankle_pitch(4), left_ankle_roll(5),
    # right_hip_pitch(6), right_hip_roll(7), right_hip_yaw(8), right_knee(9),
    # right_ankle_pitch(10), right_ankle_roll(11),
    # waist_yaw(12),
    # left_shoulder_pitch(13), left_shoulder_roll(14), left_shoulder_yaw(15),
    # left_elbow(16), left_wrist_roll(17),
    # right_shoulder_pitch(18), right_shoulder_roll(19), right_shoulder_yaw(20),
    # right_elbow(21), right_wrist_roll(22)

    qpos[:, 7:13] = dof[:, 0:6]    # Left leg
    qpos[:, 13:19] = dof[:, 6:12]  # Right leg
    qpos[:, 19] = dof[:, 12]       # Waist yaw
    qpos[:, 20:25] = dof[:, 13:18] # Left arm (5 joints)
    qpos[:, 25:30] = dof[:, 18:23] # Right arm (5 joints)

    if squeeze:
        qpos = qpos[0]

    return qpos


def qpos_to_features(qpos, velocity=None, root_vel=None, contact_mask=None):
    """
    Convert MuJoCo qpos to 59-dim motion features.

    Args:
        qpos: (T, 30) array
        velocity: Optional joint velocities (T, 23)
        root_vel: Optional root velocity (T, 2)
        contact_mask: Optional contact mask (T, 2)

    Returns:
        features_59d: (T, 59) array
    """
    qpos = np.asarray(qpos)
    squeeze = False
    if qpos.ndim == 1:
        qpos = qpos[np.newaxis, :]
        squeeze = True

    T = qpos.shape[0]
    features = np.zeros((T, 59))

    # Extract from qpos
    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]
    left_leg = qpos[:, 7:13]
    right_leg = qpos[:, 13:19]
    waist = qpos[:, 19:20]
    left_arm = qpos[:, 20:25]
    right_arm = qpos[:, 25:30]

    # Reconstruct DOF (23)
    dof = np.concatenate([left_leg, right_leg, waist, left_arm, right_arm], axis=1)

    # Convert quaternion to 6D
    from scipy.spatial.transform import Rotation as sRot
    root_rot_6d = np.zeros((T, 6))
    for t in range(T):
        # Convert wxyz to xyzw
        quat_xyzw = np.concatenate([root_quat[t, 1:4], root_quat[t, 0:1]])
        rot = sRot.from_quat(quat_xyzw)
        mat = rot.as_matrix()
        root_rot_6d[t] = mat[:2, :].flatten()

    # Fill features
    features[:, 0:23] = dof
    features[:, 23:26] = root_pos
    features[:, 26:32] = root_rot_6d

    # Fill optional components
    if velocity is not None:
        features[:, 32:55] = velocity
    if root_vel is not None:
        features[:, 55:57] = root_vel
    if contact_mask is not None:
        features[:, 57:59] = contact_mask

    if squeeze:
        features = features[0]

    return features


# Joint parent mapping for G1 (23 DOF)
# Based on g1_23dof.xml skeleton structure
G1_SKELETON = {
    # Joint name -> (parent_index, offset_in_qpos)
    'root': (-1, -1),
    'left_hip_pitch': ('root', 7),
    'left_hip_roll': ('left_hip_pitch', 8),
    'left_hip_yaw': ('left_hip_roll', 9),
    'left_knee': ('left_hip_yaw', 10),
    'left_ankle_pitch': ('left_knee', 11),
    'left_ankle_roll': ('left_ankle_pitch', 12),
    'right_hip_pitch': ('root', 13),
    'right_hip_roll': ('right_hip_pitch', 14),
    'right_hip_yaw': ('right_hip_roll', 15),
    'right_knee': ('right_hip_yaw', 16),
    'right_ankle_pitch': ('right_knee', 17),
    'right_ankle_roll': ('right_ankle_pitch', 18),
    'waist_yaw': ('root', 19),
    'left_shoulder_pitch': ('waist_yaw', 20),
    'left_shoulder_roll': ('left_shoulder_pitch', 21),
    'left_shoulder_yaw': ('left_shoulder_roll', 22),
    'left_elbow': ('left_shoulder_yaw', 23),
    'left_wrist_roll': ('left_elbow', 24),
    'right_shoulder_pitch': ('waist_yaw', 25),
    'right_shoulder_roll': ('right_shoulder_pitch', 26),
    'right_shoulder_yaw': ('right_shoulder_roll', 27),
    'right_elbow': ('right_shoulder_yaw', 28),
    'right_wrist_roll': ('right_elbow', 29),
}


if __name__ == '__main__':
    # Test conversion
    print("Testing 59-dim to 30-dim conversion...")

    # Create dummy 59-dim features
    dummy_59d = np.random.randn(59)
    dummy_59d[0:23] = 0  # Zero joint angles
    dummy_59d[23:26] = [0, 0, 0.793]  # Root at default height
    dummy_59d[26:32] = [1, 0, 0, 0, 1, 0]  # Identity rotation

    qpos = features_to_qpos(dummy_59d)
    print(f"Input 59-dim shape: {dummy_59d.shape}")
    print(f"Output qpos shape: {qpos.shape}")
    print(f"Qpos: {qpos}")
