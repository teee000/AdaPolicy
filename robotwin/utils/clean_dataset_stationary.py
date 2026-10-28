"""
Data cleaning script: remove stationary segments from HDF5 training data
=============================================
Core logic:
  1. Detect stationary segments based on joint_action/vector (frame-to-frame diff < threshold on all dims)
  2. Apply a safety margin to each detected stationary interval [n, m]; actually delete [n+margin, m-margin]
  3. Synchronously delete the corresponding frames from all time-aligned datasets in the HDF5 file
  4. Overwrite the processed data back into the original HDF5 file

Notes:
  - All datasets with shape=(T, ...) are trimmed synchronously
  - Compressed image streams (variable-length bytes) are also handled correctly
  - The original file is automatically backed up before deletion (can be disabled via --no_backup)
  - Supports dry-run mode to preview what would be deleted
"""

import os
import sys
import json
import h5py
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Data scanning
# ============================================================

def scan_tasks(dataset_dir, data_mode="both"):
    """Scan the dataset directory, return {task_name: [episode_info, ...]}"""
    root = Path(dataset_dir)
    splits = ["demo_clean", "demo_randomized"] if data_mode == "both" else [f"demo_{data_mode}"]

    task_episodes = {}
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        episodes = []
        for split in splits:
            data_dir = task_dir / split / "data"
            if not data_dir.exists():
                continue
            for hdf5_path in sorted(data_dir.glob("*.hdf5")):
                episodes.append({
                    'hdf5_path': str(hdf5_path),
                    'ep_name': hdf5_path.stem,
                    'split': split,
                    'task': task_name,
                })
        if episodes:
            task_episodes[task_name] = episodes
    return task_episodes


# ============================================================
# Stationary segment detection
# ============================================================

def detect_stationary_segments(actions, threshold=1e-6, min_duration=1):
    """
    Detect stationary segments in an action trajectory.

    Args:
        actions:      (T, D) action sequence
        threshold:    frame-to-frame diff threshold; frames where all dims fall below this are considered stationary
        min_duration: minimum duration in frames

    Returns:
        segments: list of (start, end) stationary segments (half-open, end exclusive)
    """
    T = actions.shape[0]
    if T < 2:
        return []

    diffs = np.abs(np.diff(actions, axis=0))           # (T-1, D)
    max_diff_per_frame = np.max(diffs, axis=1)          # (T-1,)
    still_flags = max_diff_per_frame < threshold        # (T-1,)

    segments = []
    i = 0
    while i < len(still_flags):
        if still_flags[i]:
            j = i
            while j < len(still_flags) and still_flags[j]:
                j += 1
            # still_flags[i:j] all True -> actions[i : j+1] is stationary
            duration = j - i + 1
            if duration >= min_duration:
                segments.append((i, j + 1))
            i = j
        else:
            i += 1

    return segments


def compute_delete_mask(T, segments, margin=3):
    """
    Compute the frame indices to delete based on stationary segments and the safety margin.

    For each stationary interval [n, m), actually delete [n+margin, m-margin).
    If the interval becomes empty after applying the margin (i.e. the stationary segment is too short), skip it.

    Returns:
        keep_mask: (T,) bool array, True = keep, False = delete
        delete_segments: list of (start, end) intervals actually deleted
    """
    delete_mask = np.zeros(T, dtype=bool)
    delete_segments = []

    for seg_start, seg_end in segments:
        actual_start = seg_start + margin
        actual_end = seg_end - margin
        if actual_start >= actual_end:
            # interval is empty after margin, skip
            continue
        # clamp to the valid range
        actual_start = max(0, actual_start)
        actual_end = min(T, actual_end)
        delete_mask[actual_start:actual_end] = True
        delete_segments.append((actual_start, actual_end))

    keep_mask = ~delete_mask
    return keep_mask, delete_segments


# ============================================================
# HDF5 dataset traversal and trimming
# ============================================================

def collect_dataset_paths(hdf5_file):
    """Recursively collect the paths of all datasets in an HDF5 file"""
    paths = []

    def _visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    hdf5_file.visititems(_visitor)
    return paths


def clean_single_file(hdf5_path, threshold, margin, dry_run=False, backup=True):
    """
    Perform stationary-segment cleaning on a single HDF5 file.

    Returns:
        stats dict or None (when the file fails to open)
    """
    try:
        f = h5py.File(hdf5_path, 'r')
    except Exception as e:
        logger.warning(f"Cannot open {hdf5_path}: {e}")
        return None

    # Read joint_action/vector as the basis for stationary detection
    if 'joint_action/vector' not in f:
        logger.warning(f"No joint_action/vector in {hdf5_path}, skipping")
        f.close()
        return None

    actions = f['joint_action/vector'][:].astype(np.float64)
    T_original = actions.shape[0]

    # Detect stationary segments
    segments = detect_stationary_segments(actions, threshold=threshold, min_duration=1)

    # Compute the intervals to actually delete
    keep_mask, delete_segments = compute_delete_mask(T_original, segments, margin=margin)
    n_keep = int(np.sum(keep_mask))
    n_delete = T_original - n_keep

    stats = {
        'file': hdf5_path,
        'T_original': T_original,
        'T_after': n_keep,
        'n_deleted': n_delete,
        'n_raw_segments': len(segments),
        'n_actual_delete_segments': len(delete_segments),
        'delete_segments': delete_segments,
        'delete_ratio': n_delete / T_original if T_original > 0 else 0,
    }

    if n_delete == 0:
        f.close()
        stats['status'] = 'no_change'
        return stats

    if dry_run:
        f.close()
        stats['status'] = 'dry_run'
        return stats

    # ============================================================
    # Actually perform cleaning: read all data -> trim -> write back
    # ============================================================

    # 1. Collect all dataset paths and data
    dataset_paths = collect_dataset_paths(f)
    all_data = {}
    dataset_meta = {}  # record the original attributes of each dataset

    keep_indices = np.where(keep_mask)[0]

    for ds_path in dataset_paths:
        ds = f[ds_path]
        shape = ds.shape
        dtype = ds.dtype

        # Check whether this dataset is aligned along the time axis (first dim == T_original)
        if len(shape) > 0 and shape[0] == T_original:
            raw_data = ds[:]
            trimmed = raw_data[keep_indices]
            all_data[ds_path] = trimmed
        else:
            # Data not aligned along the time axis is kept as-is
            all_data[ds_path] = ds[:]

        # Save meta info (used to recreate the dataset)
        dataset_meta[ds_path] = {
            'dtype': dtype,
            'original_shape': shape,
            'chunks': ds.chunks,
            'compression': ds.compression,
            'compression_opts': ds.compression_opts,
            'maxshape': ds.maxshape,
        }

    # Collect group-level attributes
    group_attrs = {}

    def _collect_attrs(name, obj):
        if isinstance(obj, h5py.Group) and dict(obj.attrs):
            group_attrs[name] = dict(obj.attrs)

    f.visititems(_collect_attrs)
    # root-level attrs
    root_attrs = dict(f.attrs)

    f.close()

    # 2. Back up the original file
    if backup:
        backup_path = hdf5_path + '.bak'
        if not os.path.exists(backup_path):
            shutil.copy2(hdf5_path, backup_path)

    # 3. Overwrite and write back
    with h5py.File(hdf5_path, 'w') as f_out:
        # Restore root attrs
        for k, v in root_attrs.items():
            f_out.attrs[k] = v

        for ds_path, data in all_data.items():
            meta = dataset_meta[ds_path]

            # Ensure the parent group exists
            parent = '/'.join(ds_path.split('/')[:-1])
            if parent and parent not in f_out:
                f_out.require_group(parent)

            # Decide how to create the dataset based on the original dataset's characteristics
            # Variable-length string/bytes types (e.g. compressed image streams)
            if h5py.check_vlen_dtype(meta['dtype']):
                vlen_type = h5py.check_vlen_dtype(meta['dtype'])
                dt = h5py.vlen_dtype(vlen_type)
                f_out.create_dataset(ds_path, data=data, dtype=dt)
            else:
                # Regular numeric types
                create_kwargs = {}
                if meta['compression'] is not None:
                    create_kwargs['compression'] = meta['compression']
                    if meta['compression_opts'] is not None:
                        create_kwargs['compression_opts'] = meta['compression_opts']
                if meta['chunks'] is not None:
                    # Adjust chunk size (first dim must not exceed the new data length)
                    new_chunks = list(meta['chunks'])
                    if len(new_chunks) > 0 and len(data.shape) > 0:
                        new_chunks[0] = min(new_chunks[0], data.shape[0])
                    if all(c > 0 for c in new_chunks):
                        create_kwargs['chunks'] = tuple(new_chunks)

                f_out.create_dataset(ds_path, data=data, **create_kwargs)

        # Restore group attrs
        for grp_path, attrs in group_attrs.items():
            if grp_path in f_out:
                for k, v in attrs.items():
                    f_out[grp_path].attrs[k] = v

    stats['status'] = 'cleaned'
    return stats


# ============================================================
# Main function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Clean stationary segments from HDF5 training data (based on joint_action/vector)")

    parser.add_argument("--config", type=str, default="./configs/robotwin_3.2_ema_4.yaml",
                        help="Path to the dataset config file")
    parser.add_argument("--dataset_dir", type=str,
                        default='/root/autodl-tmp/modelscope_cache/datasets/yangfan97--LiLa-WAM_RoboTwin2.0_50_task/snapshots/master',
                        help="Directly specify the dataset directory (overrides config)")
    parser.add_argument("--data_mode", type=str, default='both',
                        help="Data mode: clean / randomized / both")

    # Cleaning parameters
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="Stationary detection threshold: frame-to-frame diff < threshold on all dims means stationary")
    parser.add_argument("--margin", type=int, default=1,
                        help="Safety margin: for stationary interval [n,m) actually delete [n+margin, m-margin)")

    # Control options
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview mode: only collect statistics without modifying files")
    parser.add_argument("--no_backup", action="store_true", default=True,
                        help="Do not back up the original file (a .bak file is created next to the original by default)")
    parser.add_argument("--report_dir", type=str, default="./analysis_results/cleaning_report",
                        help="Output directory for the cleaning report")

    args = parser.parse_args()

    # --- Resolve dataset path ---
    dataset_dir = args.dataset_dir
    data_mode = args.data_mode or "both"

    # --- Scan the dataset ---
    task_episodes = scan_tasks(dataset_dir, data_mode)
    total_files = sum(len(eps) for eps in task_episodes.values())
    logger.info(f"Found {len(task_episodes)} tasks, {total_files} files total")

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE — no files will be modified")
        logger.info("=" * 60)

    # ============================================================
    # Process files one by one
    # ============================================================
    all_stats = []
    task_summaries = {}

    for task_name, episodes in task_episodes.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {task_name} ({len(episodes)} files)")
        logger.info(f"{'='*60}")

        task_stats_list = []

        for ep_info in tqdm(episodes, desc=f"[{task_name}]"):
            stats = clean_single_file(
                ep_info['hdf5_path'],
                threshold=args.threshold,
                margin=args.margin,
                dry_run=args.dry_run,
                backup=not args.no_backup,
            )

            if stats is None:
                continue

            task_stats_list.append(stats)
            all_stats.append(stats)

            # Detailed log
            if stats['n_deleted'] > 0:
                seg_strs = [f"[{s},{e})" for s, e in stats['delete_segments']]
                logger.info(
                    f"  {Path(stats['file']).stem}: "
                    f"{stats['T_original']} → {stats['T_after']} frames "
                    f"(deleted {stats['n_deleted']}, {stats['delete_ratio']*100:.1f}%) "
                    f"segments: {', '.join(seg_strs)} "
                    f"[{stats['status']}]"
                )
            else:
                logger.info(
                    f"  {Path(stats['file']).stem}: "
                    f"{stats['T_original']} frames, no stationary segments"
                )

        # Task summary
        if task_stats_list:
            total_orig = sum(s['T_original'] for s in task_stats_list)
            total_after = sum(s['T_after'] for s in task_stats_list)
            total_del = sum(s['n_deleted'] for s in task_stats_list)
            n_modified = sum(1 for s in task_stats_list if s['n_deleted'] > 0)

            task_summaries[task_name] = {
                'n_files': len(task_stats_list),
                'n_modified': n_modified,
                'total_frames_before': total_orig,
                'total_frames_after': total_after,
                'total_frames_deleted': total_del,
                'delete_ratio': total_del / total_orig if total_orig > 0 else 0,
            }

    # ============================================================
    # Global summary
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 60)

    header = (f"{'Task':<30} {'Files':>6} {'Modified':>8} "
              f"{'Before':>8} {'After':>8} {'Deleted':>8} {'Ratio':>8}")
    logger.info(header)
    logger.info('-' * len(header))

    grand_before = 0
    grand_after = 0
    grand_deleted = 0
    grand_files = 0
    grand_modified = 0

    for task_name in sorted(task_summaries.keys()):
        s = task_summaries[task_name]
        logger.info(
            f"{task_name:<30} {s['n_files']:>6} {s['n_modified']:>8} "
            f"{s['total_frames_before']:>8} {s['total_frames_after']:>8} "
            f"{s['total_frames_deleted']:>8} {s['delete_ratio']*100:>7.1f}%"
        )
        grand_before += s['total_frames_before']
        grand_after += s['total_frames_after']
        grand_deleted += s['total_frames_deleted']
        grand_files += s['n_files']
        grand_modified += s['n_modified']

    logger.info('-' * len(header))
    grand_ratio = grand_deleted / grand_before if grand_before > 0 else 0
    logger.info(
        f"{'*** TOTAL ***':<30} {grand_files:>6} {grand_modified:>8} "
        f"{grand_before:>8} {grand_after:>8} "
        f"{grand_deleted:>8} {grand_ratio*100:>7.1f}%"
    )

    if args.dry_run:
        logger.info("\n[DRY RUN] No files were modified. Remove --dry_run to execute.")

    # ============================================================
    # Save the report
    # ============================================================
    os.makedirs(args.report_dir, exist_ok=True)

    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'threshold': args.threshold,
            'margin': args.margin,
            'dry_run': args.dry_run,
            'dataset_dir': dataset_dir,
        },
        'global_summary': {
            'total_files': grand_files,
            'files_modified': grand_modified,
            'total_frames_before': grand_before,
            'total_frames_after': grand_after,
            'total_frames_deleted': grand_deleted,
            'delete_ratio': grand_ratio,
        },
        'per_task': task_summaries,
        'per_file': [
            {
                'file': s['file'],
                'T_original': s['T_original'],
                'T_after': s['T_after'],
                'n_deleted': s['n_deleted'],
                'delete_ratio': s['delete_ratio'],
                'delete_segments': s['delete_segments'],
                'status': s['status'],
            }
            for s in all_stats
        ],
    }

    report_path = os.path.join(args.report_dir, "cleaning_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved to: {report_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
