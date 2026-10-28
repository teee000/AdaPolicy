"""
Compare loss curves from multiple training CSV logs (log/loss/train_loss_*.csv).

Usage:
    python utils/plot_loss_curves.py log/loss/a.csv log/loss/b.csv
    python utils/plot_loss_curves.py log/loss/*.csv --labels run1 run2 \
        --skip 100 --smooth 0.9 --out log/loss_vis/compare.png

Each CSV must have the header: Epoch,Step,Global_Step,Loss
(the format written by LossLogger in the training scripts).
"""

import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Times New Roman preferred; fall back to metric-compatible serif fonts
# (Liberation Serif / Nimbus Roman) when it is not installed.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"


def load_csv(path):
    """Load a loss CSV, return (global_step, loss) arrays."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.size == 0:
        raise ValueError(f"Empty CSV: {path}")
    data = np.atleast_1d(data)
    return data["Global_Step"], data["Loss"]


def ema_smooth(values, weight):
    """Exponential moving average smoothing. weight=0 → no smoothing."""
    if weight <= 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1.0 - weight) * values[i]
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Compare loss curves from training CSV logs")
    parser.add_argument("csv_files", nargs="+", help="Paths to train_loss_*.csv files")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Legend labels, one per CSV (default: file timestamp)")
    parser.add_argument("--skip", type=int, default=100,
                        help="Drop the first N global steps (avoids a stretched y-axis)")
    parser.add_argument("--smooth", type=float, default=0.9,
                        help="EMA smoothing weight in [0, 1); 0 disables smoothing")
    parser.add_argument("--out", type=str, default=None,
                        help="Output image path (default: log/loss_vis/loss_compare_<timestamp>.png)")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.csv_files):
        raise ValueError(f"--labels count ({len(args.labels)}) != csv count ({len(args.csv_files)})")

    if args.out is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.out = os.path.join("log/loss_vis", f"loss_compare_{timestamp}.png")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    smoothed_min, smoothed_max = float("inf"), float("-inf")

    for idx, path in enumerate(args.csv_files):
        steps, losses = load_csv(path)

        # Drop the first N steps
        keep = steps > args.skip
        steps, losses = steps[keep], losses[keep]
        if len(steps) == 0:
            print(f"[warn] {path}: nothing left after --skip {args.skip}, skipped")
            continue

        label = args.labels[idx] if args.labels else \
            os.path.basename(path).replace("train_loss_", "").replace(".csv", "")
        color = colors[idx % len(colors)]

        smoothed = ema_smooth(losses, args.smooth)
        ax.plot(steps, losses, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smoothed, color=color, linewidth=1.8, label=label)
        smoothed_min = min(smoothed_min, float(np.min(smoothed)))
        smoothed_max = max(smoothed_max, float(np.max(smoothed)))

    # y-axis range follows the smoothed curves only, so raw spikes do not
    # stretch the axis and differences between curves stay visible.
    if smoothed_max > smoothed_min:
        margin = (smoothed_max - smoothed_min) * 0.05
        ax.set_ylim(smoothed_min - margin, smoothed_max + margin)

    ax.set_xlabel("Global Step", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training Loss Comparison", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
