#!/usr/bin/env python3
"""
Plot action trajectories for episodes within a task.

Features:
  - Read action data from all HDF5 files in a task directory
  - Plot each action dimension in a separate subplot with all episodes overlaid
  - Mark ±π reference lines and automatically highlight out-of-range dimensions
  - Limit the number of episodes to keep plots readable

Usage:
    # Single task
    python plot_action_curves.py --task_dir /path/to/data/turn_switch

    # Limit the number of episodes shown
    python plot_action_curves.py --task_dir /path/to/data/turn_switch --max_episodes 20

    # Batch mode
    python plot_action_curves.py --root_dir /path/to/data
"""

import os
import argparse
import numpy as np
import h5py
from glob import glob


def collect_task_episodes(task_dir, data_mode="both"):
    """Collect action data from all episodes in a task directory, keeping episodes separate."""
    hdf5_files = []
    if data_mode in ["clean", "both"]:
        hdf5_files.extend(sorted(glob(os.path.join(task_dir, "demo_clean", "data", "*.hdf5"))))
    if data_mode in ["randomized", "both"]:
        hdf5_files.extend(sorted(glob(os.path.join(task_dir, "demo_randomized", "data", "*.hdf5"))))

    if not hdf5_files:
        print(f"  Warning: No HDF5 files found in {task_dir}")
        return []

    episodes = []
    for fp in hdf5_files:
        try:
            with h5py.File(fp, 'r') as f:
                if 'joint_action/vector' in f:
                    action = f['joint_action']['vector'][:]  # (T, D)
                    name = os.path.splitext(os.path.basename(fp))[0]
                    episodes.append({"name": name, "action": action})
        except Exception as e:
            print(f"  Skipping {fp}: {e}")

    return episodes


def plot_task_curves(task_name, episodes, save_dir, max_episodes=50):
    """Plot action trajectories for a single task."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    if not episodes:
        return

    D = episodes[0]["action"].shape[1]
    PI = np.pi

    # Limit the number of episodes.
    if len(episodes) > max_episodes:
        step = len(episodes) // max_episodes
        episodes_show = episodes[::step][:max_episodes]
        print(f"  Showing {len(episodes_show)}/{len(episodes)} episodes (sampling interval: {step})")
    else:
        episodes_show = episodes

    n_eps = len(episodes_show)

    # Dimension labels
    dim_labels = []
    for i in range(D):
        if i == 6:
            dim_labels.append(f"D{i} (L gripper)")
        elif i == 13:
            dim_labels.append(f"D{i} (R gripper)")
        elif i < 7:
            dim_labels.append(f"D{i} (L joint {i})")
        else:
            dim_labels.append(f"D{i} (R joint {i-7})")

    # Color mapping
    cmap = get_cmap('tab20' if n_eps <= 20 else 'hsv')
    colors = [cmap(i / max(n_eps - 1, 1)) for i in range(n_eps)]

    # Layout: one subplot per dimension, with seven rows and two columns
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.0 * n_rows))
    axes = axes.flatten()

    fig.suptitle(
        f"Action Trajectories: {task_name}\n"
        f"({n_eps} episodes shown, red dashed = ±π)",
        fontsize=14, fontweight='bold'
    )

    for d in range(D):
        ax = axes[d]
        is_gripper = (d in [6, 13])
        has_exceed = False

        for ei, ep in enumerate(episodes_show):
            traj = ep["action"][:, d]
            t = np.arange(len(traj))

            # Detect out-of-range values.
            if not is_gripper and (traj.min() < -PI or traj.max() > PI):
                has_exceed = True

            ax.plot(t, traj, color=colors[ei], alpha=0.5, linewidth=0.6)

        # ±π reference lines
        if not is_gripper:
            ax.axhline(y=PI, color='red', linestyle='--', alpha=0.5, linewidth=1.0)
            ax.axhline(y=-PI, color='red', linestyle='--', alpha=0.5, linewidth=1.0)

        # Zero line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2, linewidth=0.5)

        title = dim_labels[d]
        if has_exceed:
            title += "  ⚠ EXCEEDS ±π"
        ax.set_title(title, fontsize=10,
                     fontweight='bold' if has_exceed else 'normal',
                     color='red' if has_exceed else 'black')
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2)

    for d in range(D, len(axes)):
        axes[d].set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"action_curves_{task_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Action trajectory visualization")
    parser.add_argument("--task_dir", type=str,
                        default='/home/yf/Desktop/Code/VLA/RoboTwin/RoboTwin/data/turn_switch',
                        help="Path to a single task directory")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="Dataset root directory (batch mode)")
    parser.add_argument("--data_mode", type=str, default="both",
                        choices=["clean", "randomized", "both"])
    parser.add_argument("--save_dir", type=str, default="./action_curve_plots")
    parser.add_argument("--max_episodes", type=int, default=100,
                        help="Maximum number of episodes to display per task")
    args = parser.parse_args()

    if args.task_dir is None and args.root_dir is None:
        parser.error("Either --task_dir or --root_dir must be specified")

    tasks = []
    if args.task_dir:
        task_name = os.path.basename(args.task_dir.rstrip('/'))
        tasks.append((task_name, args.task_dir))
    elif args.root_dir:
        for d in sorted(os.listdir(args.root_dir)):
            full = os.path.join(args.root_dir, d)
            if os.path.isdir(full):
                has_data = (os.path.exists(os.path.join(full, "demo_clean")) or
                            os.path.exists(os.path.join(full, "demo_randomized")))
                if has_data:
                    tasks.append((d, full))
        print(f"Found {len(tasks)} tasks")

    for task_name, task_dir in tasks:
        print(f"\n{'='*50}")
        print(f"  Task: {task_name}")
        print(f"{'='*50}")

        episodes = collect_task_episodes(task_dir, args.data_mode)
        if not episodes:
            continue

        print(f"  Total: {len(episodes)} episodes, "
              f"length range: {min(e['action'].shape[0] for e in episodes)}"
              f"-{max(e['action'].shape[0] for e in episodes)} steps")

        plot_task_curves(task_name, episodes, args.save_dir, args.max_episodes)


if __name__ == "__main__":
    main()