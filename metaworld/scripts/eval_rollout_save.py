"""
Eval rollout with training-format data saving (for downstream RL training).

Runs the trained policy in MetaWorld and saves every rollout episode in the
SAME hdf5 schema as gen_data_hdf5.py (RoboTwin2-compatible), so the data can
be mixed with / loaded by the existing training pipeline:

    joint_action/vector (T, 4)      executed actions (clipped to [-1, 1])
    proprio/state       (T, 4)      [eef_xyz, gripper]
    reward              (T,)        env dense reward per step
    observation/<cam>/rgb           compressed RGB frames (corner + behind)
    attrs: task_name, num_steps, success (ACTUAL outcome), episode_seed,
           reset_rng_mt/pos (global np.random state before reset -> exact
           initial-state reproduction), checkpoint, ...

RL extras appended per episode (extra keys do not affect the IL loader):
    success_step        (T,) float32  per-step success flags
    terminated          (T,) float32
    truncated           (T,) float32

Layout:
    rollout/data/<task>/demo_rollout/data/<task>_ep_XXXX.hdf5
    rollout/videos/<task>/<task>_ep_XXXX.mp4
Summary (per-task success rates) is printed and appended to
    log/eval/rollout_<tag>_<timestamp>.log

Usage:
    python eval_rollout_save.py --task assembly-v3 --num-episodes 20 \
        --checkpoint checkpoints_vla/ig_assembly-v3_xxx/checkpoint_epoch_50.pt
    python eval_rollout_save.py --task all --num-episodes 50 --checkpoint <ckpt>
"""
import os
import sys
import json
import argparse
from datetime import datetime

import h5py
import numpy as np
import metaworld
import torch

from gen_data_hdf5 import save_episode_hdf5

# running from scripts/ requires adding the repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.robotwin_infer import RobotWinInference
from eval_metaworld_parallel import MWRenderWrapper, VideoRecorder, combined_bgr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


def rollout_one_episode(agent, env, task, seed, model_cam, cameras, max_steps):
    """
    Run one episode. Returns (ep_dict, extras) with strict t-alignment:
    (obs_t, img_t, action_t) recorded before env.step, reward/success flags after.
    """
    # capture global RNG state BEFORE reset -> exact initial-state reproduction
    rng_state = np.random.get_state()

    obs, info = env.reset(seed=seed)
    agent.reset()

    ep = {"obs": [], "actions": [], "rewards": [], "task_name": task,
          "episode_seed": int(seed),
          "reset_rng_mt": np.asarray(rng_state[1], dtype=np.uint32),
          "reset_rng_pos": int(rng_state[2]),
          "success": False}
    for ck in cameras:
        ep[f"img_{ck}"] = []
    extras = {"success_step": [], "terminated": [], "truncated": []}
    frames_bgr = []

    for _ in range(max_steps):
        observation = {
            "full_state": np.asarray(obs, dtype=np.float32),
            "observation": {model_cam: {"rgb": info[f"img_{model_cam}"]}},
        }
        action = agent.step(observation, instruction=task)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # record (obs_t, img_t, action_t) BEFORE stepping
        ep["obs"].append(np.asarray(obs, dtype=np.float32))
        ep["actions"].append(action)
        for ck in cameras:
            ep[f"img_{ck}"].append(info[f"img_{ck}"])
        frames_bgr.append(combined_bgr({c: info[f"img_{c}"] for c in cameras
                                        if f"img_{c}" in info}))

        obs, reward, terminated, truncated, info = env.step(action)

        # env feedback AFTER stepping (aligned with action_t)
        ep["rewards"].append(float(reward))
        succ = bool(info.get("success", False))
        extras["success_step"].append(succ)
        extras["terminated"].append(bool(terminated))
        extras["truncated"].append(bool(truncated))

        if succ:
            ep["success"] = True
            break
        if terminated or truncated:
            break

    return ep, extras, frames_bgr


def save_rollout_hdf5(path, ep, extras, cameras, img_format, jpeg_quality, image_hw,
                      checkpoint_path):
    """Write the training-schema hdf5, then append RL extras."""
    save_episode_hdf5(path, ep, cameras=cameras, img_format=img_format,
                      jpeg_quality=jpeg_quality, image_hw=image_hw)
    with h5py.File(path, "a") as f:
        f.create_dataset("success_step",
                         data=np.asarray(extras["success_step"], dtype=np.float32))
        f.create_dataset("terminated",
                         data=np.asarray(extras["terminated"], dtype=np.float32))
        f.create_dataset("truncated",
                         data=np.asarray(extras["truncated"], dtype=np.float32))
        f.attrs["checkpoint"] = str(checkpoint_path)


def main():
    p = argparse.ArgumentParser(description="Eval rollout -> training-format data saving.")
    p.add_argument("--config", default="configs/metaworld.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--norm-stats", default="./utils/stat-metaworld.json")
    p.add_argument("--task", required=True, help="Task name, or 'all' for the MT50 suite.")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--output-dir", default="./rollout/data",
                   help="<output-dir>/<task>/demo_<split>/data/*.hdf5")
    p.add_argument("--split", default="rollout")
    p.add_argument("--video-dir", default="./rollout/videos")
    p.add_argument("--no-video", action="store_true", help="Disable mp4 saving.")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--num-videos", type=int, default=0,
                   help="Max videos per task (0 = save every episode).")
    p.add_argument("--img-format", choices=["jpg", "png"], default="png")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--mujoco-gl", default="egl")
    args = p.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    tasks = sorted(metaworld.MT50().train_classes.keys()) if args.task == "all" else [args.task]

    agent = RobotWinInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        device=device, dtype=dtype,
        task_name=None if args.task == "all" else args.task,
    )

    model_cam = list(agent.config.dataset.camera_names)[0]
    W, H = agent.image_size
    cameras = ["corner", "behind"]          # same keys as the training data

    results = {}
    for task in tasks:
        if getattr(agent, "use_task_cond", False):
            agent.set_task(task)

        task_dir = os.path.join(args.output_dir, task, f"demo_{args.split}", "data")
        os.makedirs(task_dir, exist_ok=True)
        video_dir = os.path.join(args.video_dir, task)
        if not args.no_video:
            os.makedirs(video_dir, exist_ok=True)

        mt1 = metaworld.MT1(task)
        env = MWRenderWrapper(mt1.train_classes[task](render_mode="rgb_array"),
                              cameras=cameras, width=W, height=H)
        task_pool = [t for t in mt1.train_tasks if t.env_name == task]

        successes = 0
        for ep_idx in range(args.num_episodes):
            env.unwrapped.set_task(task_pool[np.random.randint(len(task_pool))])
            if hasattr(env.unwrapped, "_freeze_rand_vec"):
                env.unwrapped._freeze_rand_vec = False

            seed = args.seed + ep_idx
            ep, extras, frames_bgr = rollout_one_episode(
                agent, env, task, seed, model_cam, cameras, args.max_steps)

            fname = f"{task}_ep_{ep_idx:04d}.hdf5"
            save_rollout_hdf5(os.path.join(task_dir, fname), ep, extras,
                              cameras=cameras, img_format=args.img_format,
                              jpeg_quality=args.jpeg_quality, image_hw=(H, W),
                              checkpoint_path=args.checkpoint)

            if not args.no_video and (args.num_videos == 0 or ep_idx < args.num_videos):
                rec = VideoRecorder(os.path.join(
                    video_dir, f"{task}_ep_{ep_idx:04d}.mp4"), fps=args.video_fps)
                for fr in frames_bgr:
                    rec.add(fr)
                rec.close()

            successes += int(ep["success"])
            tag = "\033[92mSUCCESS\033[0m" if ep["success"] else "\033[91mFAIL\033[0m"
            print(f"[{task}][{ep_idx+1:>3}/{args.num_episodes}] seed={seed} -> {tag} "
                  f"T={len(ep['actions'])} return={sum(ep['rewards']):.2f} "
                  f"(running {successes}/{ep_idx+1} = {successes/(ep_idx+1)*100:.1f}%)")

        env.close()
        rate = successes / args.num_episodes
        results[task] = (successes, args.num_episodes, rate)
        print(f"Task: {task} | Success: {successes}/{args.num_episodes} ({rate*100:.2f}%)")

    avg_rate = float(np.mean([r for _, _, r in results.values()])) if results else 0.0
    print("=" * 56)
    for task, (succ, total, rate) in results.items():
        print(f"  {task:<32} {succ:>3}/{total:<3} {rate*100:6.2f}%")
    print("-" * 56)
    print(f"MEAN SUCCESS over {len(results)} tasks: {avg_rate*100:.2f}%")
    print("=" * 56)

    # --- summary log ---
    log_dir = os.path.join(PROJECT_ROOT, "log", "eval")
    os.makedirs(log_dir, exist_ok=True)
    tag = "all" if len(tasks) > 1 else tasks[0]
    log_path = os.path.join(log_dir, f"rollout_{tag}_{datetime.now():%Y%m%d_%H%M%S}.log")
    with open(log_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\nepisodes/task: {args.num_episodes} | "
                f"max_steps: {args.max_steps} | seed: {args.seed} | "
                f"output: {args.output_dir} (demo_{args.split})\n\n")
        for task, (succ, total, rate) in results.items():
            f.write(f"{task},{succ},{total},{rate:.4f}\n")
        f.write(f"\nMEAN_SUCCESS,{avg_rate:.4f}\n")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
