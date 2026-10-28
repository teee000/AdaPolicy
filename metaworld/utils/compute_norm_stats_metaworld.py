"""
Compute per-dimension min/max normalization stats for the MetaWorld VLA dataset.

Scans:  <dataset_dir>/<task>/demo_*/data/*.hdf5
Reads:  joint_action/vector  (T, action_dim)   and   proprio/state (T, state_dim)
Writes: a JSON compatible with VLAWrapper.load_norm_stats:

    {
      "metaworld": {
        "action": {"min": [...], "max": [...]},
        "state":  {"min": [...], "max": [...]}
      }
    }

Usage:
    python compute_norm_stats_metaworld.py --dataset-dir ./expert_data \
        --out ./utils/stat-metaworld.json
"""
import os
import glob
import json
import argparse
import numpy as np
import h5py


def update_minmax(cur_min, cur_max, arr):
    a_min = arr.min(axis=0)
    a_max = arr.max(axis=0)
    if cur_min is None:
        return a_min, a_max
    return np.minimum(cur_min, a_min), np.maximum(cur_max, a_max)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="../data")
    p.add_argument("--out", default="./stat-metaworld.json")
    p.add_argument("--key", default="metaworld", help="Top-level JSON key.")
    p.add_argument("--clip", type=float, default=None,
                   help="Clip actions to [-clip, clip] BEFORE computing min/max. "
                        "Must match dataset.action_clip used at training time "
                        "(MetaWorld envs execute clipped-to-[-1,1] actions, so 1.0 "
                        "keeps the norm range compact and train/eval aligned).")
    args = p.parse_args()

    pattern = os.path.join(args.dataset_dir, "*", "demo_*", "data", "*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under pattern: {pattern}")
    print(f"Found {len(files)} episode files.")

    act_min = act_max = None
    st_min = st_max = None
    total_T = 0

    for fp in files:
        with h5py.File(fp, "r") as f:
            actions = np.asarray(f["joint_action"]["vector"], dtype=np.float64)  # (T, A)
            if "proprio" in f and "state" in f["proprio"]:
                state = np.asarray(f["proprio"]["state"], dtype=np.float64)       # (T, S)
            else:
                raise KeyError(f"{fp} missing proprio/state (is this MetaWorld data?)")
        if args.clip is not None:
            actions = np.clip(actions, -args.clip, args.clip)
        act_min, act_max = update_minmax(act_min, act_max, actions)
        st_min, st_max = update_minmax(st_min, st_max, state)
        total_T += actions.shape[0]

    stats = {
        args.key: {
            "action": {"min": act_min.tolist(), "max": act_max.tolist()},
            "state":  {"min": st_min.tolist(),  "max": st_max.tolist()},
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Transitions: {total_T}")
    print(f"action dim={len(act_min)}  min={np.round(act_min,3)}  max={np.round(act_max,3)}")
    print(f"state  dim={len(st_min)}  min={np.round(st_min,3)}  max={np.round(st_max,3)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
