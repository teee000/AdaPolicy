# Robotwin2 Dataset Loader for VLA-DINOv3 (HDF5 Version)
# Supports Robotwin2 data with HDF5 storage and flexible indexing
import os
import random
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
import torchvision.transforms as T
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)

NUM_THREADS = os.cpu_count() or 4

# ImageNet normalization (DINOv3 uses ImageNet stats)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _decode(buf):
    """Decode bytes to an RGB numpy array.

    cv2.imencode/imdecode round-trip the array identically (channels are never
    permuted across encode->decode). Both RoboTwin and our MetaWorld generator
    encode the RGB frame directly, so imdecode returns true RGB and we must NOT
    apply a BGR2RGB conversion here.
    """
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # NOTE: data saved RGB-style on purpose -> do NOT BGR2RGB here.
    return img


def _normalize_image(img_np: np.ndarray) -> np.ndarray:
    """
    HxWx3 uint8 (RGB) → 3xHxW float32, ImageNet normalized.
    """
    img = img_np.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


class RobotWinTaskDataset(data.Dataset):
    def __init__(self, dataset_dir, data_mode="clean",
                 indices_config=None, camera_names=None, image_size=(320, 240),
                 val=False, image_aug=False, use_next_chunk=False,
                 task_cond_dir=None,
                 use_future_feat=False, future_frame_offset=None,
                 action_clip=None, task_names=None):
        """
        RobotWin Dataset for DINOv3-based VLA (no language input).

        Args:
            dataset_dir
            data_mode: "clean" / "randomized" / "both"
            indices_config: {state_indices, action_indices, camera_indices}
            camera_names: camera list
            image_size: (W, H) DINOv3 patch_size(=16)
            use_next_chunk: action chunk (next-chunk prediction)
            task_cond_dir: task condition feature
            use_future_feat: (future-feature prediction for training)
            future_frame_offset: future frame offset; None → = chunk_size
            action_clip: clip loaded actions to [-action_clip, action_clip];
                None disables. MetaWorld envs clip to [-1, 1] internally, so
                1.0 keeps train targets aligned with eval-time execution.
            task_names: whitelist of task folder names to load; None = all tasks
                found under dataset_dir (joint multi-task training).
        """
        if indices_config is None:
            raise ValueError("indices_config is required")
        if 'state_indices' not in indices_config or 'action_indices' not in indices_config:
            raise ValueError("indices_config missing required keys")

        self.dataset_dirs = [Path(dataset_dir)] if isinstance(dataset_dir, str) else [Path(p) for p in dataset_dir]
        self.data_mode = data_mode
        self.all_episodes = []
        self.use_next_chunk = use_next_chunk

        self.use_future_feat = use_future_feat
        self.future_frame_offset = future_frame_offset if future_frame_offset is not None else len(indices_config['action_indices'])

        self.state_offsets = torch.tensor(indices_config['state_indices'], dtype=torch.long)
        self.action_offsets = torch.tensor(indices_config['action_indices'], dtype=torch.long)
        self.chunk_size = len(self.action_offsets)
        self.indices_config = indices_config
        self.camera_names = camera_names
        # image_size: (W, H)
        self.image_size = tuple(image_size)
        assert self.image_size[0] % 16 == 0 and self.image_size[1] % 16 == 0, \
            f"DINOv3 requires H/W to be multiples of 16, got {self.image_size}"
        self.val = val
        self.image_aug = image_aug

        self.task_cond_dir = task_cond_dir
        self.task_cond_cache = {}   # task_name -> torch.Tensor(D,)
        self.use_task_cond = task_cond_dir is not None

        self.action_clip = action_clip
        self.task_names = set(task_names) if task_names else None

        # ColorJitter
        self.aug_pool = []
        if self.image_aug and not self.val:
            logger.info("Initializing Image Augmentation (Randomly picking 1-2 ops)...")
            self.aug_pool = [
                T.ColorJitter(brightness=0.05),
                T.ColorJitter(contrast=0.05),
                T.ColorJitter(saturation=0.05),
                T.ColorJitter(hue=0.05),
            ]

        self._load_episodes()
        
        if self.use_task_cond:
            self._load_task_cond_vectors()

        logger.info("Building Index Map for dataset...")
        self._build_index_map()

        if self.use_next_chunk:
            logger.info(f"Next-Chunk mode ENABLED: will return next {self.chunk_size}-step action chunk per sample.")
        if self.use_future_feat:
            logger.info(f"Future-Feat mode ENABLED: will return future frame at offset={self.future_frame_offset}.")

    def _build_index_map(self):
        self.valid_indices = []
        self.episode_metadata = []

        current_offset = 0
        valid_ep_count = 0

        for ep_info in tqdm(self.all_episodes, desc="Scanning episode lengths"):
            path = ep_info['hdf5_path']
            with h5py.File(path, 'r') as f:
                length = f['joint_action']['vector'].shape[0]

            if length < 2:
                continue

            self.episode_metadata.append({
                'hdf5_path': path,
                'task_name': ep_info['task_name'],
                'length': length,
                'global_start': current_offset,
                'global_end': current_offset + length,
            })

            ep_start = current_offset
            ep_end = current_offset + length
            curr_indices = np.arange(ep_start, ep_end, dtype=np.int64)
            self.valid_indices.append(curr_indices)

            current_offset += length
            valid_ep_count += 1

        self.valid_indices = np.concatenate(self.valid_indices)
        self._ep_end_bounds = np.array([ep['global_end'] for ep in self.episode_metadata])

        logger.info(f"Index map built: {valid_ep_count} valid episodes, {len(self.valid_indices)} total searchable frames.")

    def _load_task_cond_vectors(self):
        cond_root = Path(self.task_cond_dir)
        if not cond_root.exists():
            raise FileNotFoundError(f"task_cond_dir not found: {cond_root}")

        # gather all task_name
        task_names = set(ep['task_name'] for ep in self.all_episodes)
        loaded = 0
        for tn in sorted(task_names):
            npy_path = cond_root / tn / "task_cond.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"Task condition vector missing for task '{tn}': {npy_path}\n")
            vec = np.load(npy_path).astype(np.float32)
            self.task_cond_cache[tn] = torch.from_numpy(vec)
            loaded += 1
        dim = next(iter(self.task_cond_cache.values())).shape[0]
        logger.info(f"Loaded task condition vectors for {loaded} tasks (dim={dim}) from {cond_root}")

    def _scan_task_folder(self, task_path: Path, split_name: str) -> List[Dict[str, Any]]:
        data_dir = task_path / "data"
        if not data_dir.exists():
            return []
        valid_episodes = []
        for hdf5_path in data_dir.glob("*.hdf5"):
            valid_episodes.append({
                'episode_name': hdf5_path.stem,
                'task_name': task_path.parent.name if task_path.name in ['demo_clean', 'demo_randomized'] else task_path.name,
                'hdf5_path': str(hdf5_path),
                'split': split_name,
            })
        return valid_episodes

    def _load_episodes(self):
        logger.info("Scanning dataset folders for all tasks..."
                    if self.task_names is None else
                    f"Scanning dataset folders for tasks: {sorted(self.task_names)}")
        data_splits = ["demo_clean", "demo_randomized"] if self.data_mode == "both" else [f"demo_{self.data_mode}"]

        for root_dir in self.dataset_dirs:
            if not root_dir.exists():
                continue
            for task_dir in [d for d in root_dir.iterdir() if d.is_dir()]:
                if self.task_names is not None and task_dir.name not in self.task_names:
                    continue
                for split in data_splits:
                    split_path = task_dir / split
                    if split_path.exists():
                        episodes = self._scan_task_folder(split_path, split)
                        self.all_episodes.extend(episodes)

        if not self.all_episodes:
            raise ValueError(f"No valid episodes found in: {self.dataset_dirs}")

        logger.info(f"Successfully scanned {len(self.all_episodes)} total episode files.")

    def _get_query_indices(self, query_idx: int, episode_len: int) -> Tuple[Dict[str, List[int]], Dict[str, torch.Tensor]]:
        ep_start, ep_end = 0, episode_len
        query_indices, padding_mask = {}, {}
        keys_to_process = {
            'state': self.indices_config['state_indices'],
            'action': self.indices_config['action_indices'],
        }
        for cam_name in self.camera_names:
            keys_to_process[cam_name] = self.indices_config['camera_indices']

        for key, delta_list in keys_to_process.items():
            abs_indices = [query_idx + delta for delta in delta_list]
            query_indices[key] = [max(ep_start, min(ep_end - 1, idx)) for idx in abs_indices]

            if key == 'action':
                valid_mask = [(idx < ep_end) for idx in abs_indices]
                padding_mask[f"{key}_mask"] = torch.from_numpy(np.array(valid_mask, dtype=bool))
            else:
                padding_mask[f"{key}_mask"] = torch.ones(len(abs_indices), dtype=torch.bool)

        return query_indices, padding_mask

    def _load_hdf5_data(self, hdf5_path: str, query_indices: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        data_batch = {}
        with h5py.File(hdf5_path, 'r') as root:
            # Actions. MetaWorld envs clip actions to [-1, 1] internally, so the
            # raw expert outputs beyond that range carry no executable meaning;
            # clip at load time to keep train/eval semantics aligned with the env.
            t_idx = np.array(query_indices['action'])
            h5_idx = np.unique(t_idx)
            actions = root['joint_action']['vector'][h5_idx][np.searchsorted(h5_idx, t_idx)]
            if self.action_clip is not None:
                actions = np.clip(actions, -self.action_clip, self.action_clip)
            data_batch['action_sequence'] = torch.from_numpy(actions).float()

            # State (proprioception).
            # MetaWorld: single-arm proprio stored at /proprio/state -> (T, 4) = [eef_xyz(3), gripper(1)].
            # (Falls back to the old RoboTwin bimanual /endpose layout if present, so this
            #  loader can read both dataset formats.)
            t_idx = np.array(query_indices['state'])
            h5_idx = np.unique(t_idx)
            if 'proprio' in root and 'state' in root['proprio']:
                state_data = root['proprio']['state'][h5_idx]          # (n, state_dim)
            else:
                # Legacy RoboTwin bimanual endpose format.
                l_pose = root['endpose']['left_endpose'][h5_idx]
                l_grip = root['endpose']['left_gripper'][h5_idx]
                r_pose = root['endpose']['right_endpose'][h5_idx]
                r_grip = root['endpose']['right_gripper'][h5_idx]
                if l_grip.ndim == 1:
                    l_grip = l_grip[:, None]
                if r_grip.ndim == 1:
                    r_grip = r_grip[:, None]
                state_data = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)
            data_batch['state'] = torch.from_numpy(state_data[np.searchsorted(h5_idx, t_idx)]).float()

            # Next Action Chunk
            if self.use_next_chunk and 'next_action' in query_indices:
                t_idx_next = np.array(query_indices['next_action'])
                h5_idx_next = np.unique(t_idx_next)
                next_actions = root['joint_action']['vector'][h5_idx_next][np.searchsorted(h5_idx_next, t_idx_next)]
                if self.action_clip is not None:
                    next_actions = np.clip(next_actions, -self.action_clip, self.action_clip)
                data_batch['next_action_sequence'] = torch.from_numpy(next_actions).float()

            # Cameras
            data_batch['frame'] = {}
            for cam in self.camera_names:
                t_idx = np.array(query_indices[cam])
                if cam not in root['observation']:
                    continue
                h5_idx = np.unique(t_idx)
                comp_imgs = root['observation'][cam]['rgb'][h5_idx]

                decoded = np.stack([_decode(img) for img in comp_imgs])
                final_img = np.ascontiguousarray(decoded[np.searchsorted(h5_idx, t_idx)])
                if (final_img.shape[2], final_img.shape[1]) != self.image_size:
                    final_img = np.stack([
                        cv2.resize(i, self.image_size, interpolation=cv2.INTER_LINEAR) for i in final_img
                    ])
                data_batch['frame'][cam] = final_img

            # Future Frame (future-feature prediction): one frame
            if self.use_future_feat and 'future_frame' in query_indices:
                cam0 = self.camera_names[0]
                if cam0 in root['observation']:
                    t_idx_f = np.array(query_indices['future_frame'])
                    h5_idx_f = np.unique(t_idx_f)
                    comp_f = root['observation'][cam0]['rgb'][h5_idx_f]
                    decoded_f = np.stack([_decode(img) for img in comp_f])
                    fut_img = np.ascontiguousarray(decoded_f[np.searchsorted(h5_idx_f, t_idx_f)])
                    if (fut_img.shape[2], fut_img.shape[1]) != self.image_size:
                        fut_img = np.stack([
                            cv2.resize(i, self.image_size, interpolation=cv2.INTER_LINEAR) for i in fut_img
                        ])
                    data_batch['future_frame'] = fut_img   # (1, H, W, 3) uint8 RGB
        return data_batch

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        global_curr_idx = self.valid_indices[idx]
        ep_idx = np.searchsorted(self._ep_end_bounds, global_curr_idx, side='right')
        ep_meta = self.episode_metadata[ep_idx]

        abs_start = ep_meta['global_start']

        try:
            # local index
            local_anchor_idx = global_curr_idx - abs_start
            total_frames = ep_meta['length']

            query_indices, padding_mask = self._get_query_indices(local_anchor_idx, total_frames)

            # Next action chunk index
            if self.use_next_chunk:
                next_action_deltas = [d + self.chunk_size for d in self.indices_config['action_indices']]
                next_abs_indices = [local_anchor_idx + d for d in next_action_deltas]
                query_indices['next_action'] = [
                    max(0, min(total_frames - 1, idx)) for idx in next_abs_indices
                ]

            # Future frame index (one frame, anchor + offset)
            if self.use_future_feat:
                future_abs_idx = local_anchor_idx + self.future_frame_offset
                query_indices['future_frame'] = [
                    max(0, min(total_frames - 1, future_abs_idx))
                ]

            data_batch = self._load_hdf5_data(ep_meta['hdf5_path'], query_indices)
            data_batch.update(padding_mask)

            # Primary camera → ImageNet Norm pixel_values
            primary_cam = self.camera_names[0]
            pixel_values = None
            if primary_cam in data_batch['frame']:
                imgs_np = data_batch['frame'][primary_cam]   # (T, H, W, 3) uint8 RGB, T=1 for camera_indices=[0]

                # augmentation
                if self.aug_pool:
                    num_ops = random.choice([1, 2])
                    active_ops = random.sample(self.aug_pool, num_ops)
                    new_imgs = []
                    for img_np in imgs_np:
                        pil_img = Image.fromarray(img_np)
                        for op in active_ops:
                            pil_img = op(pil_img)
                        new_imgs.append(np.array(pil_img))
                    imgs_np = np.stack(new_imgs, axis=0)

                # norm
                normed = np.stack([_normalize_image(img) for img in imgs_np], axis=0)   # (T, 3, H, W)
                pixel_values = torch.from_numpy(normed).float()
                # camera_indices default [0] → T=1, squeeze to (3, H, W)
                if pixel_values.shape[0] == 1:
                    pixel_values = pixel_values.squeeze(0)

            result = {
                'state': data_batch['state'],
                'action_sequence': data_batch['action_sequence'],
                'pixel_values': pixel_values,
                'state_mask': data_batch['state_mask'],
                'action_mask': data_batch['action_mask'],
            }

            if self.use_next_chunk and 'next_action_sequence' in data_batch:
                result['next_action_sequence'] = data_batch['next_action_sequence']

            # future frame pixel_values (no color aug)
            if self.use_future_feat and 'future_frame' in data_batch:
                fut_np = data_batch['future_frame']                        # (1, H, W, 3)
                fut_normed = np.stack([_normalize_image(img) for img in fut_np], axis=0)  # (1, 3, H, W)
                future_pixel_values = torch.from_numpy(fut_normed).float()
                if future_pixel_values.shape[0] == 1:
                    future_pixel_values = future_pixel_values.squeeze(0)   # (3, H, W)
                result['future_pixel_values'] = future_pixel_values

            # task cond emb
            if self.use_task_cond:
                task_name = ep_meta['task_name']
                result['task_cond'] = self.task_cond_cache[task_name]   # (D,)

            return result

        except Exception as e:
            logger.warning(f"Error loading idx {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


def create_dataset(config: Any, val: bool = False, use_next_chunk: bool = False):
    from omegaconf import OmegaConf
    indices_config = OmegaConf.to_container(config.dataset.indices_config, resolve=True)

    image_size = tuple(OmegaConf.to_container(config.dataset.image_size, resolve=True))

    task_cond_dir = None
    if config.model.get('use_task_cond', False):
        task_cond_dir = config.dataset.get('task_cond_dir', None)
        if task_cond_dir is None:
            raise ValueError("model.use_task_cond=True, dataset.task_cond_dir not defined")

    ff_cfg = config.model.get('future_feat', {})
    use_future_feat = ff_cfg.get('enabled', False) if ff_cfg else False
    future_frame_offset = config.dataset.get('future_frame_offset', None)

    params = {
        'dataset_dir': config.dataset.dataset_dir,
        'indices_config': indices_config,
        'val': val,
        'image_aug': config.dataset.image_aug and not val,
        'camera_names': list(config.dataset.camera_names),
        'data_mode': config.dataset.data_mode,
        'image_size': image_size,
        'use_next_chunk': use_next_chunk,
        'task_cond_dir': task_cond_dir,
        'use_future_feat': use_future_feat,
        'future_frame_offset': future_frame_offset,
        'action_clip': config.dataset.get('action_clip', None),
        'task_names': config.dataset.get('task_names', None),
    }

    return RobotWinTaskDataset(**params)


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    result = {}
    keys = batch[0].keys()

    for key in keys:
        val = batch[0][key]
        if isinstance(val, torch.Tensor):
            result[key] = torch.stack([sample[key] for sample in batch])
        elif val is None:
            result[key] = None
        else:
            result[key] = [sample[key] for sample in batch]

    return result
