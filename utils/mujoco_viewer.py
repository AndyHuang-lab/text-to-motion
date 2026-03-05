"""
MuJoCo Viewer for G1 Motion Visualization

This script provides real-time 3D visualization using MuJoCo.
Requires MuJoCo to be installed.

Installation:
    pip install mujoco

On macOS, MuJoCo requires additional setup. Consider using the
matplotlib viewer (matplotlib_viewer.py) instead.

Usage:
    # Visualize from dataset
    python utils/mujoco_viewer.py --data-path dataset/val.pkl

    # Visualize with MuJoCo model
    python utils/mujoco_viewer.py --xml g1_description/mjcf/scene_23dof.xml

    # Generate from model and visualize
    python utils/mujoco_viewer.py --fm-checkpoint checkpoints/fm_best.pt --text "walk forward"
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not installed. Install with: pip install mujoco")
    print("On macOS, use matplotlib_viewer.py instead.")

from utils.mujoco_utils import features_to_qpos


# G1 23-DOF joint names in MuJoCo order
G1_JOINT_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
]


def load_model(xml_path='g1_description/mjcf/scene_23dof.xml'):
    """Load MuJoCo model from XML file."""
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo is not installed")

    # Handle relative path
    xml_path = Path(xml_path)
    if not xml_path.is_absolute():
        # Assume relative to project root
        project_root = Path(__file__).parent.parent
        xml_path = project_root / xml_path

    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    return model, data


def set_qpos_from_features(data, features_59d):
    """
    Set MuJoCo data.qpos from 59-dim features.

    Args:
        data: MuJoCo MjData object
        features_59d: (59,) or (T, 59) array
    """
    qpos = features_to_qpos(features_59d)

    if qpos.ndim == 2:
        qpos = qpos[0]  # Take first frame

    # MuJoCo qpos should be (30,) for G1 23DOF
    data.qpos[:] = qpos


def visualize_with_mujoco(features_59d, xml_path='g1_description/mjcf/scene_23dof.xml',
                         title="G1 Motion Visualization"):
    """
    Visualize motion using MuJoCo viewer.

    Args:
        features_59d: (T, 59) array of motion features
        xml_path: Path to MuJoCo XML file
        title: Window title
    """
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available. Falling back to matplotlib viewer...")
        from utils.matplotlib_viewer import visualize_motion
        return visualize_motion(features_59d)

    model, data = load_model(xml_path)
    features_59d = np.asarray(features_59d)

    if features_59d.ndim == 1:
        features_59d = features_59d[np.newaxis, :]

    T = features_59d.shape[0]

    print(f"Starting MuJoCo viewer with {T} frames...")
    print("Controls:")
    print("  Space: Pause/Resume")
    print("  Esc: Exit")
    print("  Tab: Switch view")

    with mujoco.viewer.launch_passive(model, data, title=title) as viewer:
        frame_idx = 0
        start_time = 0
        paused = False

        while viewer.is_running():
            # Get current time from viewer
            current_time = viewer.read_time

            # Update position based on time
            if not paused:
                frame_idx = int(current_time * 50) % T  # Assume 50 FPS
                set_qpos_from_features(data, features_59d[frame_idx])

            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Handle pause
            if viewer.user_input:
                key = viewer.user_input
                if key == 32:  # Space
                    paused = not paused


def launch_interactive_viewer(xml_path='g1_description/mjcf/scene_23dof.xml'):
    """
    Launch interactive MuJoCo viewer.

    Allows keyboard control to navigate through frames manually.
    """
    if not MUJOCO_AVAILABLE:
        raise ImportError("MuJoCo is not installed")

    model, data = load_model(xml_path)

    with mujoco.viewer.launch(model, data) as viewer:
        print("Interactive MuJoCo Viewer")
        print("Controls:")
        print("  Arrow keys: Move camera")
        print("  Ctrl + arrows: Rotate camera")
        print("  Scroll: Zoom")
        print("  Esc: Exit")

        while viewer.is_running():
            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize motion with MuJoCo')
    parser.add_argument('--data-path', type=str, help='Path to dataset')
    parser.add_argument('--xml', type=str, default='g1_description/mjcf/scene_23dof.xml',
                        help='Path to MuJoCo XML')
    parser.add_argument('--idx', type=int, default=0, help='Sample index')

    args = parser.parse_args()

    if args.data_path:
        from dataloader.data import SimpleMotionDataset
        import torch

        datadir = str(Path(args.data_path).parent)
        split = Path(args.data_path).stem

        dataset = SimpleMotionDataset(
            datadir=datadir,
            split=split,
            history_len=10,
            future_len=20,
        )

        history, future, _ = dataset[args.idx]

        # Denormalize
        mean, std = dataset.mean, dataset.std
        history = history * std + mean
        future = future * std + mean

        motion = torch.cat([history, future], dim=0).numpy()

        visualize_with_mujoco(motion, xml_path=args.xml)
    else:
        launch_interactive_viewer(xml_path=args.xml)
