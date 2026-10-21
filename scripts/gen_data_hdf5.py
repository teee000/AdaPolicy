"""
MetaWorld expert-trajectory generator for VLA imitation learning.

Refined from the original test_gen_data.py:
  * one folder per task, laid out for direct reuse by the RoboTwin2 loader:
        <output>/<task>/demo_<mode>/data/<task>_ep_XXXX.hdf5
  * number of demonstration episodes is configurable (per task)
  * data is saved as HDF5 with a RoboTwin2-compatible group schema
  * images are stored as compressed (JPEG/PNG) byte streams under
        observation/<cam>/rgb          (variable-length uint8)
    matching how the RoboTwin2 dataloader (`_decode` via cv2.imdecode) reads them.

------------------------------------------------------------------------------
COLOR CHANNEL NOTE (important, matches the RoboTwin2 loader):
------------------------------------------------------------------------------
The RoboTwin2 `_decode()` does `cv2.imdecode(...)` and then *does not* apply
BGR->RGB, feeding the array straight into ImageNet-RGB normalization.
cv2.imencode -> cv2.imdecode is an identity round-trip on the numpy array
(channels are never permuted across encode/decode). MuJoCo's renderer already
returns RGB, so we encode the RGB frame DIRECTLY (no cv2.cvtColor). The loader
then reads back true RGB. If we converted RGB->BGR before encoding, the loader
would consume B/R-swapped images. So: DO NOT convert to BGR here.
(Side effect: these .jpg files look R/B-swapped in a normal viewer -- expected.)
------------------------------------------------------------------------------

HDF5 schema written per episode (single-arm, proprioception-only):
    joint_action/vector        (T, 4)   float32   MetaWorld action [dx, dy, dz, gripper]
    proprio/state              (T, 4)   float32   robot proprio ONLY: [eef_xyz(3), gripper(1)]
    observation/<cam>/rgb      (T,)     vlen-uint8  compressed bytes, decode -> RGB
    reward                     (T,)     float32
  + root attrs: task_name, num_steps, action_dim, proprio_dim, cameras,
                image_hw, img_format, success

Design notes:
  * Only the robot's OWN state (end-effector xyz + gripper) is stored as proprio.
    Object poses and the task goal are intentionally NOT stored -- the VLA is
    meant to infer the goal from the images / language instruction.
  * MetaWorld is a single Sawyer arm. Unlike RoboTwin (bimanual), we do NOT pad
    to two arms; there is one proprio vector and one action vector, full stop.
  * The full 39-dim observation is still used at collection time (the scripted
    expert policy needs it), but only obs[0:4] is persisted.
"""

import os
import argparse
import importlib

import numpy as np
import cv2
import h5py
import gymnasium as gym
import mujoco
import metaworld
import random


# ---------------------------------------------------------------------------
# Full MT50 task list (for --tasks all)
# ---------------------------------------------------------------------------
def get_all_task_names():
    mt50 = metaworld.MT50()
    return sorted(mt50.train_classes.keys())


# ---------------------------------------------------------------------------
# 1. Dynamic Expert Policy Loader
# ---------------------------------------------------------------------------
# Env names whose expert policy module uses a different spelling.
_POLICY_NAME_ALIASES = {
    "peg-insert-side-v3": "peg-insertion-side-v3",
}


def get_expert_policy(env_name):
    """
    'drawer-close-v3' -> module 'sawyer_drawer_close_v3_policy'
                      -> class  'SawyerDrawerCloseV3Policy'
    """
    policy_name = _POLICY_NAME_ALIASES.get(env_name, env_name)
    module_name = f"sawyer_{policy_name.replace('-', '_')}_policy"
    class_name = "Sawyer" + "".join(w.capitalize() for w in policy_name.split('-')) + "Policy"
    try:
        policy_module = importlib.import_module(f"metaworld.policies.{module_name}")
        policy_cls = getattr(policy_module, class_name)
        return policy_cls()
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Failed to load expert policy '{class_name}' for task '{env_name}'."
        ) from e


# ---------------------------------------------------------------------------
# 2. Multi-View Render Wrapper
# ---------------------------------------------------------------------------
class MultiViewRenderWrapper(gym.Wrapper):
    def __init__(self, env, width=256, height=256):
        super().__init__(env)
        self.mj_model = self.env.unwrapped.model
        self.mj_data = self.env.unwrapped.data
        # Native MuJoCo renderer -> full control over resolution & camera.
        self.custom_renderer = mujoco.Renderer(self.mj_model, height=height, width=width)

    def _get_cam_frame(self, camera_name):
        """Returns an RGB uint8 (H, W, 3) frame (MuJoCo renders RGB)."""
        try:
            self.custom_renderer.update_scene(self.mj_data, camera=camera_name)
            img = self.custom_renderer.render()          # RGB
            if camera_name == 'corner':
                # Fix upside-down 'corner' camera in v3 (geometry only, not color).
                img = np.rot90(img, 2)
            return img
        except Exception as e:
            print(f"Error rendering camera '{camera_name}': {e}")
            return np.zeros(
                (self.custom_renderer.height, self.custom_renderer.width, 3),
                dtype=np.uint8,
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['img_corner'] = self._get_cam_frame('corner')
        info['img_behind'] = self._get_cam_frame('behindGripper')
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['img_corner'] = self._get_cam_frame('corner')
        info['img_behind'] = self._get_cam_frame('behindGripper')
        return obs, reward, terminated, truncated, info

    def set_task(self, task):
        self.env.set_task(task)


# ---------------------------------------------------------------------------
# 3. Image encoding (RGB-preserving; see COLOR CHANNEL NOTE at top of file)
# ---------------------------------------------------------------------------
def encode_rgb(frame_rgb, img_format="jpg", jpeg_quality=95):
    """
    Encode an RGB frame to compressed bytes WITHOUT converting to BGR.

    The RoboTwin2 loader decodes with cv2.imdecode and treats the result as RGB
    (no BGR2RGB). Because imencode/imdecode round-trip the array identically,
    encoding the RGB frame directly makes the loader read back true RGB.
    """
    frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    if img_format == "jpg":
        ok, buf = cv2.imencode(".jpg", frame_rgb,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    elif img_format == "png":
        ok, buf = cv2.imencode(".png", frame_rgb)   # lossless
    else:
        raise ValueError(f"Unsupported img_format: {img_format}")
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf  # uint8 ndarray


# ---------------------------------------------------------------------------
# 4. HDF5 writer (RoboTwin2-compatible schema)
# ---------------------------------------------------------------------------
def save_episode_hdf5(path, ep, cameras, img_format, jpeg_quality, image_hw):
    """
    ep: dict with keys obs, actions, rewards, and one img list per camera
        (ep['img_<cam>'] -> list of RGB frames).
    """
    obs = np.asarray(ep["obs"], dtype=np.float32)          # (T, 39) full obs (policy input)
    actions = np.asarray(ep["actions"], dtype=np.float32)  # (T, 4)
    rewards = np.asarray(ep["rewards"], dtype=np.float32)  # (T,)
    T = actions.shape[0]

    # Proprioception ONLY: end-effector xyz (3) + gripper openness (1) = 4 dims.
    # Object poses (obs[4:18]) and goal (obs[36:39]) are deliberately dropped;
    # the model infers the goal from images / language.
    proprio = obs[:, 0:4].astype(np.float32)               # (T, 4)

    # vlen uint8 dtype for compressed image byte streams
    try:
        vlen_u8 = h5py.vlen_dtype(np.uint8)
    except AttributeError:  # older h5py
        vlen_u8 = h5py.special_dtype(vlen=np.uint8)

    with h5py.File(path, "w") as f:
        # --- action (single arm, 4-dim) ---
        g_act = f.create_group("joint_action")
        g_act.create_dataset("vector", data=actions)       # (T, 4)

        # --- proprioception (single arm; NO dual-arm padding) ---
        g_pro = f.create_group("proprio")
        g_pro.create_dataset("state", data=proprio)        # (T, 4) [eef_xyz, gripper]

        # --- rewards ---
        f.create_dataset("reward", data=rewards)           # (T,)

        # --- cameras: compressed RGB byte streams ---
        g_obs = f.create_group("observation")
        for cam in cameras:
            frames = ep[f"img_{cam}"]
            g_cam = g_obs.create_group(cam)
            ds = g_cam.create_dataset("rgb", shape=(T,), dtype=vlen_u8)
            for i in range(T):
                ds[i] = encode_rgb(frames[i], img_format, jpeg_quality)

        # --- metadata ---
        f.attrs["task_name"] = ep["task_name"]
        f.attrs["episode_seed"] = int(ep["episode_seed"])
        # Global np.random state right before env.reset():
        #   restore = np.random.set_state(('MT19937', mt, pos, 0, 0.0))
        #   then env._freeze_rand_vec = False; env.reset(seed=episode_seed)
        f.attrs["reset_rng_mt"] = ep["reset_rng_mt"]       # (624,) uint32
        f.attrs["reset_rng_pos"] = ep["reset_rng_pos"]
        f.attrs["num_steps"] = T
        f.attrs["action_dim"] = actions.shape[1]
        f.attrs["proprio_dim"] = proprio.shape[1]
        f.attrs["cameras"] = np.array(cameras, dtype="S")
        f.attrs["image_hw"] = np.array(image_hw, dtype=np.int32)  # (H, W)
        f.attrs["img_format"] = img_format
        f.attrs["success"] = bool(ep.get("success", True))


# ---------------------------------------------------------------------------
# 5. Per-task collection loop
# ---------------------------------------------------------------------------
def collect_task(env_name, num_episodes, output_root,
                 split="clean", width=256, height=256,
                 img_format="jpg", jpeg_quality=95,
                 max_steps=500, post_success_steps=15,
                 base_seed=42, cameras=("corner", "behind"),
                 cam_render_map=None):
    """
    Collect `num_episodes` successful expert demos for a single task and write
    them to  <output_root>/<env_name>/demo_<split>/data/<env_name>_ep_XXXX.hdf5
    """
    # camera key in the hdf5  ->  MuJoCo camera name / wrapper info key
    if cam_render_map is None:
        cam_render_map = {"corner": "img_corner", "behind": "img_behind"}

    task_dir = os.path.join(output_root, env_name, f"demo_{split}", "data")
    os.makedirs(task_dir, exist_ok=True)

    # Skip tasks that already have enough episodes (resume after interruption).
    existing = len([f for f in os.listdir(task_dir) if f.endswith(".hdf5")])
    if existing >= num_episodes:
        print(f"Task: {env_name} | already has {existing} eps (>= {num_episodes}), skipping.")
        return 0

    mt1 = metaworld.MT1(env_name)
    env_cls = mt1.train_classes[env_name]
    env = env_cls(render_mode="rgb_array")
    env = MultiViewRenderWrapper(env, width=width, height=height)

    expert_policy = get_expert_policy(env_name)
    task_pool = [t for t in mt1.train_tasks if t.env_name == env_name]

    print("=" * 62)
    print(f"Task: {env_name} | Target episodes: {num_episodes} | "
          f"res: {width}x{height} | fmt: {img_format}")
    print(f"Output: {task_dir}")
    print("=" * 62)

    successful = 0
    total_steps = 0

    while successful < num_episodes:
        task = random.choice(task_pool)
        env.unwrapped.set_task(task)
        if hasattr(env.unwrapped, "_freeze_rand_vec"):
            env.unwrapped._freeze_rand_vec = False

        episode_seed = base_seed + successful * 1000 + random.randint(0, 999)
        # Capture the global RNG state BEFORE reset: some envs draw the rand_vec
        # multiple times per reset (e.g. rejection loops), so storing only
        # _last_rand_vec is NOT enough to reproduce the initial state.
        rng_state = np.random.get_state()
        obs, info = env.reset(seed=episode_seed)

        cur = {ck: info[ik] for ck, ik in cam_render_map.items()}

        ep = {"obs": [], "actions": [], "rewards": [], "task_name": env_name,
              "episode_seed": episode_seed,
              "reset_rng_mt": np.asarray(rng_state[1], dtype=np.uint32),
              "reset_rng_pos": int(rng_state[2])}
        for ck in cam_render_map:
            ep[f"img_{ck}"] = []

        is_success = False
        post_success = 0
        step_count = 0

        for _ in range(max_steps):
            action = expert_policy.get_action(obs)

            # Save (obs_t, img_t, action_t) with strict t-alignment.
            ep["obs"].append(obs)
            ep["actions"].append(action)
            for ck in cam_render_map:
                ep[f"img_{ck}"].append(cur[ck])

            next_obs, reward, terminated, truncated, sinfo = env.step(action)
            ep["rewards"].append(reward)

            obs = next_obs
            cur = {ck: sinfo[ik] for ck, ik in cam_render_map.items()}
            step_count += 1

            if bool(sinfo.get("success", False)):
                is_success = True
                post_success += 1
            if post_success >= post_success_steps:
                break
            if terminated or truncated:
                break

        if is_success:
            fname = f"{env_name}_ep_{successful:04d}.{'hdf5'}"
            fpath = os.path.join(task_dir, fname)
            save_episode_hdf5(
                fpath, ep, cameras=list(cam_render_map.keys()),
                img_format=img_format, jpeg_quality=jpeg_quality,
                image_hw=(height, width),
            )
            total_steps += step_count
            successful += 1
            print(f"  [{successful:>3}/{num_episodes}] saved  steps={step_count:<4} -> {fname}")
        else:
            print("  episode failed by expert, discarding...")

    env.close()
    print(f"--- {env_name} done: {successful} eps, {total_steps} transitions ---\n")
    return total_steps


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="MetaWorld expert data generator (HDF5).")
    p.add_argument("--tasks", nargs="+",
                   default=["door-lock-v3", "box-close-v3", "assembly-v3", "basketball-v3"],
                   help="Task name(s), or 'all' for the full MT50 suite.")
    p.add_argument("--num-episodes", type=int, default=50,
                   help="Number of successful demos PER task.")
    p.add_argument("--output-dir", default="./data")
    p.add_argument("--split", default="clean",
                   help="Split name -> folder demo_<split> (loader data_mode).")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--img-format", choices=["jpg", "png"], default="png")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--post-success-steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tasks = get_all_task_names() if args.tasks == ["all"] else args.tasks

    random.seed(args.seed)
    np.random.seed(args.seed)

    grand_total = 0
    for env_name in tasks:
        grand_total += collect_task(
            env_name=env_name,
            num_episodes=args.num_episodes,
            output_root=args.output_dir,
            split=args.split,
            width=args.width, height=args.height,
            img_format=args.img_format, jpeg_quality=args.jpeg_quality,
            max_steps=args.max_steps, post_success_steps=args.post_success_steps,
            base_seed=args.seed,
        )

    print("=" * 62)
    print(f"ALL DONE. Tasks: {len(tasks)} | Total transitions: {grand_total}")
    print("=" * 62)


if __name__ == "__main__":
    main()
