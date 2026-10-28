"""
MetaWorld evaluation harness for the DINOv3 VLA.

Two modes:
  --mode serial    : one env at a time (simple, deterministic, easy to debug)
  --mode parallel  : N envs via gym.vector.AsyncVectorEnv (throughput), batched
                     policy inference. References test_env_parallel.py for the
                     per-worker dual-camera render-into-info pattern.

Video saving (--save-video) writes MP4s to:
    <video-dir>/<run_id>/epoch_<epoch>/...
where <run_id> is the checkpoint's run directory (e.g. ig_2026-07-09_03-44-56)
and <epoch> is parsed from the checkpoint filename (checkpoint_epoch_30.pt -> 30).

Observation contract handed to the single-env agent.step():
    {'full_state': (39,), 'observation': {<cam>: {'rgb': HxWx3 uint8 RGB}}}

Color: MuJoCo renders RGB; the model consumes RGB (ImageNet RGB stats), matching
training -- do NOT convert to BGR for the model. BGR is only used for cv2 video.
The 'corner' camera is rotated 180deg to match data generation.

Usage:
    # serial + video
    python eval_metaworld.py --mode serial --save-video \
        --checkpoint ./checkpoints_vla/ig_2026-07-09_03-44-56/checkpoint_epoch_30.pt \
        --task window-open-v3 --num-episodes 20

    # parallel (8 workers) + video
    python eval_metaworld.py --mode parallel --num-envs 8 --save-video \
        --checkpoint ./checkpoints_vla/ig_2026-07-09_03-44-56/checkpoint_epoch_30.pt \
        --task window-open-v3 --num-episodes 40
    # --config defaults to config.yaml next to the checkpoint.
"""
import argparse
import logging
import os
import re
import math
import sys
from datetime import datetime

os.environ["MUJOCO_GL"] = "egl"

# running from scripts/ requires adding the repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import gymnasium as gym
import mujoco
import metaworld
import torch

from models.robotwin_infer import RobotWinInference, IMAGENET_MEAN, IMAGENET_STD, bspline_smooth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset camera key -> actual MuJoCo camera name.
CAM_TO_MUJOCO = {"corner": "corner", "behind": "behindGripper"}


# ---------------------------------------------------------------------------
# Render wrapper: attaches info['img_<cam>'] for each requested camera on both
# reset() and step(), using a native mujoco.Renderer. Matches data generation
# (corner rotated 180deg). Works in-process (serial) and inside AsyncVectorEnv
# subprocess workers (parallel).
# ---------------------------------------------------------------------------
class MWRenderWrapper(gym.Wrapper):
    def __init__(self, env, cameras=("corner",), width=256, height=256):
        super().__init__(env)
        self.cameras = list(cameras)
        self.renderer = mujoco.Renderer(self.env.unwrapped.model, height=height, width=width)

    def _frame(self, cam_key):
        mj_cam = CAM_TO_MUJOCO.get(cam_key, cam_key)
        self.renderer.update_scene(self.env.unwrapped.data, camera=mj_cam)
        img = self.renderer.render()                 # RGB uint8
        if cam_key == "corner":
            img = np.rot90(img, 2)                   # same fix as data generation
        return np.ascontiguousarray(img)

    def _attach(self, info):
        for cam in self.cameras:
            info[f"img_{cam}"] = self._frame(cam)
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, self._attach(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._attach(info)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def parse_run_and_epoch(checkpoint_path):
    run_id = os.path.basename(os.path.dirname(checkpoint_path)) or "unknown_run"
    m = re.search(r"epoch_?(\d+)", os.path.basename(checkpoint_path))
    epoch = m.group(1) if m else "unknown"
    return run_id, epoch


def combined_bgr(imgs_by_cam):
    """imgs_by_cam: {cam_key: HxWx3 RGB}. Concatenate corner|behind -> BGR for cv2."""
    order = [c for c in ("corner", "behind") if c in imgs_by_cam]
    order += [c for c in imgs_by_cam if c not in order]
    if not order:
        return None
    frames = [np.asarray(imgs_by_cam[c]) for c in order]
    combined = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
    return cv2.cvtColor(np.ascontiguousarray(combined.astype(np.uint8)), cv2.COLOR_RGB2BGR)


class VideoRecorder:
    """Lazily-opened MP4 writer (size inferred from first frame)."""
    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps
        self.writer = None

    def add(self, bgr_frame):
        if bgr_frame is None:
            return
        if self.writer is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            h, w = bgr_frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
            )
        self.writer.write(bgr_frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def video_root(args, checkpoint_path):
    run_id, epoch = parse_run_and_epoch(checkpoint_path)
    root = os.path.join(args.video_dir, run_id, f"epoch_{epoch}")
    return root


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------
def make_single_env(task_name, cameras, W, H, seed=None):
    mt1 = metaworld.MT1(task_name)
    env_cls = mt1.train_classes[task_name]
    env = env_cls(render_mode="rgb_array")
    task_pool = [t for t in mt1.train_tasks if t.env_name == task_name]
    rng = np.random.RandomState(seed if seed is not None else 0)
    env.set_task(task_pool[rng.randint(len(task_pool))])
    if hasattr(env.unwrapped, "_freeze_rand_vec"):
        env.unwrapped._freeze_rand_vec = False
    env = MWRenderWrapper(env, cameras=cameras, width=W, height=H)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_env_fn(task_name, cameras, W, H, seed):
    def _init():
        return make_single_env(task_name, cameras, W, H, seed=seed)
    return _init


# ---------------------------------------------------------------------------
# Batched policy inference (parallel mode). Reuses the agent's frozen encoder +
# action expert; numerically identical pipeline to RobotWinInference._predict_chunk.
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_chunk_batch(agent, proprio_np, frames_np):
    """
    proprio_np : (N, proprio_dim)     current proprioception per env
    frames_np  : (N, H, W, 3) uint8   model-camera RGB per env
    returns    : (N, action_chunk_size, action_dim) denormalized actions
    """
    device, dtype = agent.device, agent.dtype
    N = proprio_np.shape[0]

    # Images -> (N, 3, H, W) ImageNet-normalized (RGB).
    imgs = frames_np.astype(np.float32) / 255.0
    imgs = (imgs - IMAGENET_MEAN) / IMAGENET_STD
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    pixel_values = torch.from_numpy(np.ascontiguousarray(imgs)).to(device, dtype)

    # State history (proprio_len). state_indices default [0] -> history_len 1;
    # for >1 we replicate current proprio, mirroring the processor's initial fill.
    hist = agent.processor.history_len
    state = np.repeat(proprio_np[:, None, :].astype(np.float32), hist, axis=1)  # (N, hist, dim)
    state_t = torch.from_numpy(state).to(device, dtype)
    qpos = agent.model.normalize_state(state_t)

    dino_features_list = agent.model.get_vision_features(pixel_values)

    # Task condition (disabled in single-task config -> None).
    task_cond = None
    if agent.task_cond is not None:
        task_cond = agent.task_cond.expand(N, -1)

    L = agent.config.common.action_chunk_size
    Dact = agent.config.common.action_dim
    x_t = torch.randn((N, L, Dact), device=device, dtype=dtype)
    steps = torch.linspace(0, 1, agent.num_inference_steps + 1, device=device, dtype=dtype)

    for i in range(agent.num_inference_steps):
        t_curr = steps[i]
        dt = steps[i + 1] - t_curr
        t_input = t_curr.repeat(N)  # (N,)

        preds = agent.model.action_model(
            t=t_input,
            noisy_actions=x_t,
            qpos_history=qpos,
            dino_features_list=dino_features_list,
            task_cond=task_cond,
        )
        pred_v = preds["final_pred"]

        x_t = x_t + pred_v * dt

    action = agent.model.denormalize_action(x_t).float().cpu().numpy()  # (N, L, Dact)

    if agent.smooth_actions:
        action = np.stack([bspline_smooth(action[n], degree=3, num_ctrl_pts=8)
                           for n in range(N)], axis=0)
    return action


def _get_success_array(infos, N):
    """Robust per-env success extraction across gymnasium vector-info variants."""
    succ = np.zeros(N, dtype=bool)
    s = infos.get("success", None)
    if s is not None:
        s = np.asarray(s).reshape(-1)
        m = min(N, s.shape[0])
        succ[:m] |= (s[:m].astype(np.float32) > 0.5)
    fi = infos.get("final_info", None)      # terminal-step info on auto-reset
    if fi is not None:
        for i in range(min(N, len(fi))):
            d = fi[i]
            if isinstance(d, dict) and float(d.get("success", 0)) > 0.5:
                succ[i] = True
    return succ


# ---------------------------------------------------------------------------
# Serial evaluation
# ---------------------------------------------------------------------------
def run_serial(agent, args, model_cam, cameras, W, H, vroot):
    mt1 = metaworld.MT1(args.task)
    env_cls = mt1.train_classes[args.task]
    env = env_cls(render_mode="rgb_array")
    task_pool = [t for t in mt1.train_tasks if t.env_name == args.task]
    env = MWRenderWrapper(env, cameras=cameras, width=W, height=H)

    successes = 0
    for ep in range(args.num_episodes):
        env.unwrapped.set_task(task_pool[np.random.randint(len(task_pool))])
        if hasattr(env.unwrapped, "_freeze_rand_vec"):
            env.unwrapped._freeze_rand_vec = False

        seed = args.seed + ep
        obs, info = env.reset(seed=seed)
        agent.reset()

        record = args.save_video and (args.num_videos == 0 or ep < args.num_videos)
        recorder = VideoRecorder(os.path.join(vroot, f"{args.task}_ep{ep:03d}.mp4"),
                                 fps=args.video_fps) if record else None

        ep_success = False
        for _ in range(args.max_steps):
            observation = {
                "full_state": np.asarray(obs, dtype=np.float32),
                "observation": {model_cam: {"rgb": info[f"img_{model_cam}"]}},
            }
            action = agent.step(observation, instruction=args.task)
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

            if recorder is not None:
                imgs = {c: info[f"img_{c}"] for c in cameras if f"img_{c}" in info}
                recorder.add(combined_bgr(imgs))

            obs, reward, terminated, truncated, info = env.step(action)

            if bool(info.get("success", False)):
                ep_success = True
                break
            if terminated or truncated:
                break

        if recorder is not None:
            recorder.close()

        successes += int(ep_success)
        tag = "\033[92mSUCCESS\033[0m" if ep_success else "\033[91mFAIL\033[0m"
        print(f"[{ep+1:>3}/{args.num_episodes}] seed={seed} -> {tag}  "
              f"(running {successes}/{ep+1} = {successes/(ep+1)*100:.1f}%)")

    env.close()
    return successes, args.num_episodes


# ---------------------------------------------------------------------------
# Parallel evaluation (AsyncVectorEnv), batched inference
# ---------------------------------------------------------------------------
def run_parallel(agent, args, model_cam, cameras, W, H, vroot):
    N = args.num_envs
    horizon = agent.action_execution_horizon
    proprio_dim = agent.config.common.state_dim
    video_idx = min(args.video_env_index, N - 1)

    num_batches = math.ceil(args.num_episodes / N)
    total_success = 0
    total_episodes = 0

    for b in range(num_batches):
        base = args.seed + b * N
        env_fns = [make_env_fn(args.task, cameras, W, H, seed=base + i) for i in range(N)]
        envs = gym.vector.AsyncVectorEnv(env_fns)

        cur_obs, cur_info = envs.reset(seed=[base + i for i in range(N)])
        agent.reset()  # processor buffers are unused in batched path; harmless

        active = np.ones(N, dtype=bool)
        success = np.zeros(N, dtype=bool)

        record = args.save_video and (args.num_videos == 0 or b < args.num_videos)
        recorder = VideoRecorder(
            os.path.join(vroot, f"{args.task}_batch{b:02d}_env{video_idx}.mp4"),
            fps=args.video_fps) if record else None

        chunk = None
        for t in range(args.max_steps):
            if t % horizon == 0:
                proprio = cur_obs[:, :proprio_dim].astype(np.float32)         # (N, dim)
                frames = np.asarray(cur_info[f"img_{model_cam}"])             # (N, H, W, 3)
                chunk = predict_chunk_batch(agent, proprio, frames)          # (N, L, dim)

            k = t % horizon
            actions = np.clip(chunk[:, k, :], -1.0, 1.0).astype(np.float32)

            # record BEFORE stepping (frame corresponds to current state)
            if recorder is not None and active[video_idx]:
                imgs = {c: np.asarray(cur_info[f"img_{c}"])[video_idx]
                        for c in cameras if f"img_{c}" in cur_info}
                recorder.add(combined_bgr(imgs))

            cur_obs, rewards, terms, truncs, cur_info = envs.step(actions)

            succ = _get_success_array(cur_info, N)
            success |= (active & succ)

            ended = np.asarray(terms, bool) | np.asarray(truncs, bool) | succ
            active = active & (~ended)

            if not active.any():
                break

        if recorder is not None:
            recorder.close()
        envs.close()

        # Count this batch's N episodes (cap at requested total).
        take = min(N, args.num_episodes - total_episodes)
        total_success += int(success[:take].sum())
        total_episodes += take

        print(f"[batch {b+1}/{num_batches}] episodes so far {total_episodes}: "
              f"{total_success}/{total_episodes} = {total_success/max(total_episodes,1)*100:.1f}%")

    return total_success, total_episodes


# ---------------------------------------------------------------------------
def run_eval(args):
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl  # set before env subprocesses spawn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    # --task all -> evaluate the full MT50 suite one task at a time
    if args.task == "all":
        tasks = sorted(metaworld.MT50().train_classes.keys())
    else:
        tasks = [args.task]

    agent = RobotWinInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        device=device,
        dtype=dtype,
        task_name=None if args.task == "all" else args.task,  # per-task set_task below
    )
    if args.num_inference_steps:
        agent.num_inference_steps = args.num_inference_steps

    model_cam = list(agent.config.dataset.camera_names)[0]
    W, H = agent.image_size

    # Render the model camera always; add both views for a richer video.
    cameras = [model_cam]
    if args.save_video:
        for c in ("corner", "behind"):
            if c not in cameras:
                cameras.append(c)

    vroot = video_root(args, args.checkpoint) if args.save_video else None
    if args.save_video:
        os.makedirs(vroot, exist_ok=True)
        logger.info(f"Saving videos under: {vroot}")

    results = {}   # task -> (success, total, rate)
    for task in tasks:
        args.task = task
        if getattr(agent, "use_task_cond", False):
            agent.set_task(task)

        if args.mode == "parallel":
            succ, total = run_parallel(agent, args, model_cam, cameras, W, H, vroot)
        else:
            succ, total = run_serial(agent, args, model_cam, cameras, W, H, vroot)

        rate = succ / total if total else 0.0
        results[task] = (succ, total, rate)
        print(f"Task: {task} | Success: {succ}/{total} ({rate*100:.2f}%)")

    avg_rate = float(np.mean([r for _, _, r in results.values()])) if results else 0.0

    print("=" * 56)
    for task, (succ, total, rate) in results.items():
        print(f"  {task:<32} {succ:>3}/{total:<3} {rate*100:6.2f}%")
    print("-" * 56)
    print(f"MEAN SUCCESS over {len(results)} tasks: {avg_rate*100:.2f}%  "
          f"(ckpt: {args.checkpoint})")
    if args.save_video:
        print(f"Videos: {vroot}")
    print("=" * 56)

    # --- log to file ---
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "log", "eval")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "all" if len(tasks) > 1 else tasks[0]
    log_path = os.path.join(log_dir, f"eval_{tag}_{timestamp}.log")
    with open(log_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\nconfig: {args.config}\n"
                f"norm_stats: {args.norm_stats}\nmode: {args.mode} | "
                f"num_envs: {args.num_envs} | episodes/task: {args.num_episodes} | "
                f"max_steps: {args.max_steps} | seed: {args.seed}\n\n")
        for task, (succ, total, rate) in results.items():
            f.write(f"{task},{succ},{total},{rate:.4f}\n")
        f.write(f"\nMEAN_SUCCESS,{avg_rate:.4f}\n")
    print(f"Log saved to: {log_path}")
    return avg_rate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/metaworld.yaml",
                   help="Path to the run's config.yaml. Default: config.yaml beside --checkpoint.")
    p.add_argument("--checkpoint", default="./checkpoints_vla/ig_2026-07-09_03-44-56/checkpoint_epoch_30.pt")
    p.add_argument("--norm-stats", default="./utils/stat-metaworld.json")
    p.add_argument("--task", default="window-open-v3",
                   help="Task name, or 'all' to evaluate the full MT50 suite "
                        "one task at a time (per-task rates + mean logged to log/eval/).")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-inference-steps", type=int, default=None,
                   help="Override common.num_inference_steps (Euler ODE steps).")

    # --- evaluation mode ---
    p.add_argument("--mode", choices=["serial", "parallel"], default="parallel")
    p.add_argument("--num-envs", type=int, default=50, help="Parallel workers (parallel mode).")
    p.add_argument("--mujoco-gl", default="egl",
                   help="MUJOCO_GL backend for headless rendering (egl/osmesa/glfw). "
                        "Set empty to leave unchanged.")

    # --- video control ---
    p.add_argument("--save-video", default=False, help="Enable MP4 saving during eval.")
    p.add_argument("--video-dir", default="rollout/videos",
                   help="Root dir; videos go to <video-dir>/<run_id>/epoch_<epoch>/.")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--num-videos", type=int, default=5,
                   help="Max clips to record (serial: first K episodes; parallel: first K batches). 0 = all.")
    p.add_argument("--video-env-index", type=int, default=1,
                   help="Which parallel worker to record (parallel mode).")

    args = p.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

    def _abs(pth):
        if pth is None or os.path.isabs(pth):
            return pth
        return os.path.normpath(os.path.join(BASE_DIR, pth))

    args.checkpoint = _abs(args.checkpoint)
    args.norm_stats = _abs(args.norm_stats)
    args.video_dir = _abs(args.video_dir)

    if args.config is None:
        args.config = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
    else:
        args.config = _abs(args.config)

    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"Config YAML not found: {args.config}\n"
            f"Pass --config <run>/config.yaml (NOT the norm-stats JSON)."
        )
    return args


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass 
    run_eval(parse_args())
