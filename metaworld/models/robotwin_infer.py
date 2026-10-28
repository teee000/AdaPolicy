import os
import torch
import numpy as np
import logging
from typing import Dict, Any, List
from omegaconf import OmegaConf
from collections import deque
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline

from .model_runner import ModelFactory, VLAWrapper


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def bspline_smooth(action_seq: np.ndarray, degree: int = 3, num_ctrl_pts: int = 8) -> np.ndarray:
    N, D = action_seq.shape
    if N <= num_ctrl_pts:
        return action_seq

    x = np.arange(N)
    num_internal_knots = num_ctrl_pts - degree
    internal_knots = np.linspace(0, N - 1, num_internal_knots + 2)[1:-1]
    knots = np.concatenate([
        [0] * (degree + 1),
        internal_knots,
        [N - 1] * (degree + 1),
    ])
    spline = make_lsq_spline(x, action_seq, knots, k=degree)
    return spline(x)


def normalize_image_np(img_np: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB → 3xHxW float32 ImageNet normalized."""
    img = img_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return img


class RobotWinInference:
    """
    RobotWin VLA infer
    """
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        norm_stats_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        task_name: str = None,
    ):
        self.device = device
        self.dtype = dtype

        # 1. Config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        self.config = OmegaConf.load(config_path)

        self.num_inference_steps = self.config.common.num_inference_steps
        self.action_execution_horizon = self.config.common.action_execution_horizon

        self.smooth_actions = self.config.inference.smooth_actions
        self.smooth_sigma = self.config.inference.smooth_sigma

        time_sampler = self.config.training.time_sampler

        self.image_size = tuple(OmegaConf.to_container(self.config.dataset.image_size, resolve=True))   # (W, H)

        self.action_queue = deque()

        # 2. Models
        logger.info(f"Loading Model from {checkpoint_path}...")

        vision_encoder, dino_hidden_size, num_register_tokens, patch_size = ModelFactory.create_vision_encoder(
            self.config.model.vision_encoder.checkpoint_path,
            dtype, device,
        )

        feat_layers = list(self.config.model.vision_encoder.feat_layers)
        num_dino_layers = len(feat_layers)

        self.use_task_cond = self.config.model.get('use_task_cond', False)
        self.task_cond_dim = dino_hidden_size if self.use_task_cond else None
        self.task_cond_dir = self.config.dataset.get('task_cond_dir', None) if self.use_task_cond else None
        self.task_cond = None  

        action_model = ModelFactory.create_action_model(
            self.config,
            dino_hidden_size=dino_hidden_size,
            num_dino_layers=num_dino_layers,
            task_cond_dim=self.task_cond_dim,
            patch_size=patch_size,
        )

        ff_cfg = self.config.model.get('future_feat', {})
        self.model = VLAWrapper(
            vision_encoder=vision_encoder,
            action_model=action_model,
            time_sampler=time_sampler,
            feat_layers=feat_layers,
            include_cls_register=self.config.model.vision_encoder.include_cls_register,
            num_register_tokens=num_register_tokens,
            device=device,
            dtype=dtype,
            norm_stats_path=norm_stats_path,
            train_config=None,
            future_feat_target_layer=ff_cfg.get('target_layer', -1) if ff_cfg else -1,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        msg = self.model.action_model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded Action Model weights. Missing: {len(msg.missing_keys)}, "
                    f"Unexpected: {len(msg.unexpected_keys)}")

        self.model.eval()
        self.model.to(device, dtype)

        # Processor
        indices_config = self.config.dataset.indices_config
        self.processor = RobotWinInferenceProcessor(
            indices_config=indices_config,
            camera_name=list(self.config.dataset.camera_names)[0],
            image_size=self.image_size,
            device=device,
            dtype=dtype,
            proprio_dim=self.config.common.state_dim,   # MetaWorld: 4
        )

        if self.use_task_cond and task_name is not None:
            self.set_task(task_name)

        logger.info(f"Inference Engine Ready. "
                    f"Smooth: {self.smooth_actions}; "
                    f"feat_layers={feat_layers}, image_size={self.image_size}; "
                    f"TaskCond: {self.use_task_cond}")

    def set_task(self, task_name: str):
        if not self.use_task_cond:
            return
        if self.task_cond_dir is None:
            raise ValueError("use_task_cond=True but dataset.task_cond_dir not defined")
        npy_path = os.path.join(self.task_cond_dir, task_name, "task_cond.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Task condition vector not found for '{task_name}': {npy_path}")
        vec = np.load(npy_path).astype(np.float32)
        self.task_cond = torch.from_numpy(vec).to(self.device, self.dtype).unsqueeze(0)  # (1, D)
        logger.info(f"Loaded task condition for '{task_name}' (dim={vec.shape[0]}, |v|={np.linalg.norm(vec):.3f})")

    def reset(self):
        self.processor.reset()
        self.action_queue.clear()

    @torch.no_grad()
    def _predict_chunk(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        if self.use_task_cond and self.task_cond is None:
            raise RuntimeError("use_task_cond=True but not set_task(task_name)")
            
        batch = self.processor.process(observation)

        # 2. Conditioning
        qpos_cond = self.model.normalize_state(batch['state'])
        dino_features_list = self.model.get_vision_features(batch['pixel_values'])

        # 3. Flow Matching Sampling 
        B = 1
        action_len = self.config.common.action_chunk_size
        action_dim = self.config.common.action_dim

        x_t = torch.randn((B, action_len, action_dim), device=self.device, dtype=self.dtype)
        steps = torch.linspace(0, 1, self.num_inference_steps + 1, device=self.device, dtype=self.dtype)

        # 4. ODE Solver
        for i in range(self.num_inference_steps):
            t_curr = steps[i]
            dt = steps[i+1] - t_curr
            t_input = t_curr.unsqueeze(0)

            preds = self.model.action_model(
                t=t_input,
                noisy_actions=x_t,
                qpos_history=qpos_cond,
                dino_features_list=dino_features_list,
                task_cond=self.task_cond,
            )

            pred_v = preds["final_pred"]

            x_t = x_t + pred_v * dt

        # 5. Denormalize
        action_seq = self.model.denormalize_action(x_t)
        action_np = action_seq[0].float().cpu().numpy()

        if self.smooth_actions:
            action_np = bspline_smooth(action_np, degree=3, num_ctrl_pts=8)

        return action_np

    def step(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        self.processor.update_state_buffer(observation)

        if len(self.action_queue) == 0:
            full_chunk = self._predict_chunk(observation, instruction)
            valid_actions = full_chunk[:self.action_execution_horizon]
            for act in valid_actions:
                self.action_queue.append(act)

        return self.action_queue.popleft()


class RobotWinInferenceProcessor:
    def __init__(
        self,
        indices_config: Dict[str, List[int]] = None,
        camera_name: str = 'corner',
        image_size=(320, 240),    # (W, H)
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        proprio_dim: int = 4,
    ):
        self.device = device
        self.dtype = dtype
        self.camera_name = camera_name
        self.image_size = tuple(image_size)
        self.proprio_dim = proprio_dim   # MetaWorld single-arm proprio = [eef_xyz(3), gripper(1)]

        self.state_indices = indices_config['state_indices']
        self.history_len = 1 + abs(min(self.state_indices))

        self.state_buffer = deque(maxlen=self.history_len)

    def reset(self):
        self.state_buffer.clear()

    def update_state_buffer(self, observation: Dict[str, Any]):
        current_state = self._parse_state_from_obs(observation)
        if len(self.state_buffer) == 0:
            for _ in range(self.history_len):
                self.state_buffer.append(current_state)
        else:
            self.state_buffer.append(current_state)

    def _parse_state_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        """MetaWorld single-arm proprioception -> (proprio_dim,) = [eef_xyz(3), gripper(1)].

        Accepts, in priority order:
          * obs['proprio']    : already-sliced proprio vector
          * obs['state']      : proprio vector (non-dict)
          * obs['full_state'] : the raw 39-dim MetaWorld observation; we take the
                                first `proprio_dim` dims, matching how the training
                                data was generated (proprio = obs[0:4]).
        This deliberately drops object/goal state -- the goal is inferred from vision.
        """
        if 'proprio' in obs:
            vec = np.asarray(obs['proprio'], dtype=np.float32)
        elif 'state' in obs and not isinstance(obs['state'], dict):
            vec = np.asarray(obs['state'], dtype=np.float32)
        elif 'full_state' in obs:
            vec = np.asarray(obs['full_state'], dtype=np.float32)
        else:
            raise KeyError(
                "MetaWorld observation must contain one of 'proprio' / 'state' / 'full_state'."
            )
        return vec[:self.proprio_dim].astype(np.float32)

    def process(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # State
        state_seq_np = np.stack(list(self.state_buffer), axis=0)
        state_tensor = torch.from_numpy(state_seq_np).to(self.device, self.dtype).unsqueeze(0)

        # Image
        pixel_values = None
        if self.camera_name in observation['observation']:
            import cv2
            img_np = observation['observation'][self.camera_name]['rgb']    # HxWx3 RGB uint8

            # Resize to (W, H)
            if (img_np.shape[1], img_np.shape[0]) != self.image_size:
                img_np = cv2.resize(img_np, self.image_size, interpolation=cv2.INTER_LINEAR)

            normed = normalize_image_np(img_np)   # (3, H, W)
            pixel_values = torch.from_numpy(normed).to(self.device, self.dtype).unsqueeze(0)
        else:
            logger.warning(f"Camera {self.camera_name} not found in observation!")

        return {
            'state': state_tensor,
            'pixel_values': pixel_values,
        }


class ActionRecorder:
    def __init__(self):
        self.actions = []

    def record(self, action):
        if hasattr(action, 'cpu'):
            action = action.cpu().detach().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        self.actions.append(action)

    def plot_and_save(self, save_dir, episode_id):
        if not self.actions:
            return

        actions_np = np.array(self.actions)
        T, D = actions_np.shape

        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        axes = axes.flatten()

        for d in range(min(D, 14)):
            axes[d].plot(actions_np[:, d], color='b')
            axes[d].set_title(f'Action Dimension {d} (Joint Angle)')
            axes[d].set_xlabel('Step')
            axes[d].set_ylabel('Value')
            axes[d].grid(True)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'action_episode_{episode_id}.png')
        plt.savefig(save_path)
        plt.close(fig)
        self.actions = []


if __name__ == "__main__":
    # For a full MetaWorld rollout, use eval_metaworld.py. This is just a smoke test.
    agent = RobotWinInference(
        config_path="./configs/metaworld.yaml",
        checkpoint_path="./checkpoints_vla/ig_XXXX/checkpoint_epoch_60.pt",
        norm_stats_path="./utils/stat-metaworld.json",
    )
    # obs = env.reset()
    # agent.reset()
    # for i in range(100):
    #     action = agent.step(obs)
    #     obs, _, _, _ = env.step(action)
