#!/usr/bin/env python3
"""
Simple Motion Visualization Script

Visualizes motion from dataset using matplotlib 3D skeleton viewer.

Usage:
    # Visualize a sample from validation set
    python visualize_motion.py

    # Visualize specific sample
    python visualize_motion.py --idx 5

    # Save as GIF
    python visualize_motion.py --save-gif output.gif

    # Use different split
    python visualize_motion.py --split train --idx 10
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.matplotlib_viewer import visualize_sample_from_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Visualize G1 motion with matplotlib 3D skeleton',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_motion.py              # Visualize first validation sample
  python visualize_motion.py --idx 5      # Visualize 6th sample
  python visualize_motion.py --save-gif walk.gif   # Save as GIF
  python visualize_motion.py --split train --idx 10   # Use training set
        """
    )
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to use (default: val)')
    parser.add_argument('--idx', type=int, default=0,
                        help='Sample index to visualize (default: 0)')
    parser.add_argument('--save-gif', type=str, default=None,
                        help='Path to save GIF animation (optional)')
    parser.add_argument('--fps', type=int, default=50,
                        help='Frames per second for animation (default: 50)')

    args = parser.parse_args()

    dataset_path = f'dataset/{args.split}.pkl'

    print("=" * 60)
    print("G1 Motion Visualization")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Sample index: {args.idx}")
    if args.save_gif:
        print(f"Save GIF: {args.save_gif}")
    print("=" * 60)

    visualize_sample_from_dataset(
        dataset_path=dataset_path,
        idx=args.idx,
        save_gif=args.save_gif
    )


if __name__ == '__main__':
    main()
