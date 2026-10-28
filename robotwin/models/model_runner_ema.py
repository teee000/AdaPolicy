"""
Model factory, VLA wrapper, and fp32 EMA parameter tracking.

EMAParams maintains shadow parameters for the second forward pass used by
next-chunk prediction. Shadows are stored and updated in fp32, cast to the model
dtype for use, and saved and restored through state_dict and load_state_dict.
"""
import torch
import torch.nn as nn
import logging
import json
from transformers import AutoModel, AutoConfig

from .model_fm_ema import VLAModel, calc_flow_matching_loss

logger = logging.getLogger(__name__)


class ModelFactory:
    @staticmethod
    def create_vision_encoder(checkpoint_path, dtype=torch.bfloat16, device="cuda"):
        """
        load & freeze DINOv3 ViT.

        Returns:
            model: DINOv3 model (eval mode, frozen)
            hidden_size: int, model hidden dimension
            num_register_tokens: int, register token num
            patch_size: int, ViT patch size
        """
        logger.info(f"Loading frozen DINOv3 vision encoder from {checkpoint_path}...")
        
        config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)

        model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)

        # Freeze
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        hidden_size = getattr(config, "hidden_size", None)
        num_register_tokens = getattr(config, "num_register_tokens", 4)
        patch_size = getattr(config, "patch_size", 16)

        if hidden_size is None:
            # fallback: dummy forward
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
                out = model(pixel_values=dummy)
                hidden_size = out.last_hidden_state.shape[-1]

        logger.info(
            f"DINOv3 loaded: hidden_size={hidden_size}, "
            f"num_register_tokens={num_register_tokens}, patch_size={patch_size}"
        )
        return model, hidden_size, num_register_tokens, patch_size

    @staticmethod
    def create_action_model(config, dino_hidden_size, num_dino_layers, task_cond_dim=None,
                            patch_size=16):
        """create VLAModel"""
        logger.info("Initializing VLAModel...")

        model_cfg = config.model
        ae_cfg = model_cfg.action_expert
        ve_cfg = model_cfg.vision_encoder
        
        dino_feat_dims = tuple([dino_hidden_size] * num_dino_layers)

        # ---- multi layer feature fusion mode ----
        fusion_mode = ve_cfg.get('fusion_mode', 'per_layer')
        concat_cfg = ve_cfg.get('concat', {})
        concat_proj_type = concat_cfg.get('proj_type', 'linear') if concat_cfg else 'linear'
        concat_pre_norm = concat_cfg.get('pre_norm', True) if concat_cfg else True
        concat_out_dim = concat_cfg.get('out_dim', None) if concat_cfg else None
        if concat_out_dim is None:
            concat_out_dim = dino_hidden_size   

        # ---- task condition insertion point ----
        task_cond_insert = model_cfg.get('task_cond_insert', 'action_head')

        # ---- future feature prediction ----
        ff_cfg = model_cfg.get('future_feat', {})
        use_future_feat = ff_cfg.get('enabled', False) if ff_cfg else False
        future_feat_attach = ff_cfg.get('attach', 'cond_tokens') if ff_cfg else 'cond_tokens'
        future_feat_depth = ff_cfg.get('num_decoder_layers', 2) if ff_cfg else 2
        future_feat_heads = ff_cfg.get('num_heads', 4) if ff_cfg else 4

        # future frame patch token num (query num)
        img_w, img_h = tuple(config.dataset.image_size)
        num_patches = (img_h // patch_size) * (img_w // patch_size)
        logger.info(f"Fusion mode: {fusion_mode} "
                    f"(concat_out_dim={concat_out_dim}, proj={concat_proj_type}, pre_norm={concat_pre_norm})")
        if task_cond_dim is not None:
            logger.info(f"Task-cond insertion point: {task_cond_insert}")
        if use_future_feat:
            logger.info(f"Future-Feat Prediction ENABLED: attach={future_feat_attach}, "
                        f"num_queries(num_patches)={num_patches}, out_dim={dino_hidden_size}")

        model = VLAModel(
            action_dim=config.common.action_dim,
            proprio_dim=config.common.state_dim,
            hidden_dim=ae_cfg.hidden_size,
            action_len=config.common.action_chunk_size,
            proprio_len=config.common.proprio_len,
            depth=ae_cfg.depth,
            num_heads=ae_cfg.num_heads,
            dino_feat_dims=dino_feat_dims,
            vlm_num_queries=ae_cfg.vlm_adapter_num_queries,
            adapter_depth=ae_cfg.adapter_depth,
            num_registers=model_cfg.num_registers,
            state_inject_start=0,
            task_cond_dim=task_cond_dim,
            task_cond_insert=task_cond_insert,
            fusion_mode=fusion_mode,
            concat_proj_type=concat_proj_type,
            concat_pre_norm=concat_pre_norm,
            concat_out_dim=concat_out_dim,
            use_future_feat=use_future_feat,
            future_feat_attach=future_feat_attach,
            future_feat_num_queries=num_patches,
            future_feat_out_dim=dino_hidden_size,
            future_feat_depth=future_feat_depth,
            future_feat_heads=future_feat_heads,
        )
        return model


class EMAParams:
    """
    Maintain fp32 EMA (Exponential Moving Average) shadow copies of Action Model
    parameters for the second forward pass in next-chunk prediction.

    Update formula: shadow = decay * shadow + (1 - decay) * param (in fp32).
    Call update() only after optimizer.step().

    Shadows are stored in fp32 because the relative update from (1-decay) at each
    step can be smaller than bf16 precision, causing EMA updates to stall due to
    rounding. get_params(dtype=...) casts shadows to the model dtype only for use.
    """

    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.named_parameters()}
        logger.info(f"EMAParams initialized: decay={decay}, num_params={len(self.shadow)}")

    @torch.no_grad()
    def update(self, model):
        """Track online parameters in fp32 after optimizer.step()."""
        for k, v in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def get_params(self, dtype=None):
        """Return a parameter dict for torch.func.functional_call without gradient tracking."""
        if dtype is None:
            return dict(self.shadow)
        return {k: v.to(dtype) for k, v in self.shadow.items()}

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict.get('decay', self.decay)
        for k, v in state_dict['shadow'].items():
            if k in self.shadow:
                self.shadow[k].copy_(v.float())
            else:
                self.shadow[k] = v.float().clone()


class VLAWrapper(nn.Module):
    def __init__(self,
                 vision_encoder,
                 action_model,
                 time_sampler,
                 feat_layers,
                 include_cls_register,
                 num_register_tokens,
                 device,
                 dtype,
                 norm_stats_path,
                 train_config=None,
                 future_feat_target_layer=-1,
                 ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.action_model = action_model
        self.time_sampler = time_sampler
        self.feat_layers = list(feat_layers)
        self.include_cls_register = include_cls_register
        self.num_register_tokens = num_register_tokens
        self.future_feat_target_layer = future_feat_target_layer

        self.device = device
        self.dtype = dtype

        # param only for training
        self.train_config = train_config
        if self.train_config is not None:
            self.time_mu = train_config['time_mu']
            self.time_sigma = train_config['time_sigma']
            self.use_next_chunk_pred = train_config['use_next_chunk_pred']
            self.lambda_next_chunk = train_config['lambda_next_chunk']
            self.use_future_feat = train_config.get('use_future_feat', False)
            self.future_feat_attach = train_config.get('future_feat_attach', 'cond_tokens')
            self.lambda_future_feat = train_config.get('lambda_future_feat', 0.0)
            self.use_ema = train_config.get('use_ema', False)
            self.ema_decay = train_config.get('ema_decay', 0.999)
        else:
            # default param for infer
            self.time_mu = 0.0
            self.time_sigma = 1.0
            self.use_next_chunk_pred = False
            self.lambda_next_chunk = 0.0
            self.use_future_feat = False
            self.future_feat_attach = 'cond_tokens'
            self.lambda_future_feat = 0.0
            self.use_ema = False
            self.ema_decay = 0.999

        # EMA of the action-head params for the Next-Chunk second forward.
        # EMAParams instance; assigned externally after construction
        # (train script: model.ema_helper = EMAParams(action_model, ...)).
        self.ema_helper = None

        logger.info(f"VLAWrapper initialized. feat_layers={self.feat_layers}, "
                    f"include_cls_register={self.include_cls_register}")
        if self.use_next_chunk_pred:
            logger.info(f"Next-Chunk Prediction ENABLED (lambda={self.lambda_next_chunk})")
        if self.use_ema:
            logger.info(f"EMA for Next-Chunk second forward ENABLED (decay={self.ema_decay})")
            
        self.load_norm_stats(norm_stats_path)

    def load_norm_stats(self, path):
        logger.info(f"Loading normalization stats from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)

        stats = data['robotwin2']
        action_stats = stats['action']
        state_stats = stats['state']

        act_min = torch.tensor(action_stats['min'], dtype=torch.float32)
        act_max = torch.tensor(action_stats['max'], dtype=torch.float32)
        self.register_buffer('action_min', act_min)
        self.register_buffer('action_max', act_max)
        logger.info(f"Loaded Action stats - Dim: {len(action_stats['min'])}")

        state_min = torch.tensor(state_stats['min'], dtype=torch.float32)
        state_max = torch.tensor(state_stats['max'], dtype=torch.float32)
        self.register_buffer('state_min', state_min)
        self.register_buffer('state_max', state_max)
        logger.info(f"Loaded State stats - Dim: {len(state_stats['min'])}")

    @torch.no_grad()
    def get_vision_features(self, pixel_values):
        """
        get DINOv3 hidden states.

        Args:
            pixel_values: (B, 3, H, W)

        Returns:
            List[Tensor(B, N, hidden_size)] length = len(feat_layers)
            if include_cls_register=True, N = 1 + num_register_tokens + num_patches
            else N = num_patches (excluding the CLS and register tokens)
        """
        pixel_values = pixel_values.to(self.device, self.dtype)
        outputs = self.vision_encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states

        feats_list = []
        for layer_idx in self.feat_layers:
            h = hidden_states[layer_idx]   # (B, 1+R+P, D)
            if not self.include_cls_register:
                skip = 1 + self.num_register_tokens
                h = h[:, skip:, :]
            feats_list.append(h)
        return feats_list

    @torch.no_grad()
    def get_future_target_features(self, future_pixel_values):
        """
        get future frame DINO patch token feature, freeze, no gradient

        Args:
            future_pixel_values: (B, 3, H, W)

        Returns:
            Tensor(B, P, hidden_size): only patch tokens (no 1 CLS + R register)
        """
        future_pixel_values = future_pixel_values.to(self.device, self.dtype)
        outputs = self.vision_encoder(
            pixel_values=future_pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        h = outputs.hidden_states[self.future_feat_target_layer]   # (B, 1+R+P, D)
        skip = 1 + self.num_register_tokens
        h = h[:, skip:, :]                                          # (B, P, D) only patch tokens
        return h

    def _normalize_tensor(self, x, min_val, max_val):
        min_v = min_val.to(device=x.device, dtype=x.dtype)
        max_v = max_val.to(device=x.device, dtype=x.dtype)

        denominator = max_v - min_v
        denominator[denominator < 1e-6] = 1.0

        norm_x = 2 * (x - min_v) / denominator - 1
        return norm_x

    def normalize_action(self, action):
        return self._normalize_tensor(action, self.action_min, self.action_max)

    def normalize_state(self, state):
        return self._normalize_tensor(state, self.state_min, self.state_max)

    def denormalize_action(self, norm_action):
        action_min = self.action_min.to(device=norm_action.device, dtype=norm_action.dtype)
        action_max = self.action_max.to(device=norm_action.device, dtype=norm_action.dtype)

        denominator = action_max - action_min
        denominator[denominator < 1e-6] = 1.0

        action = (norm_action + 1) / 2 * denominator + action_min
        return action

    def forward(self, batch):
        # 1. Vision feature
        pixel_values = batch['pixel_values']      # (B, 3, H, W)
        dino_features_list = self.get_vision_features(pixel_values)

        # 2. Action / State
        x1_raw = batch['action_sequence'].to(self.device, self.dtype)
        qpos_raw = batch['state'].to(self.device, self.dtype)

        if qpos_raw.dim() == 2:
            qpos_raw = qpos_raw.unsqueeze(1)

        # 3. Normalize
        x1 = self.normalize_action(x1_raw)
        qpos = self.normalize_state(qpos_raw)
        qpos_history = qpos

        # 4. Next-Chunk GT
        x1_next = None
        if self.use_next_chunk_pred and batch.get('next_action_sequence') is not None:
            x1_next_raw = batch['next_action_sequence'].to(self.device, self.dtype)
            x1_next = self.normalize_action(x1_next_raw)

        # 4b. Task Condition 
        task_cond = None
        if batch.get('task_cond') is not None:
            task_cond = batch['task_cond'].to(self.device, self.dtype)

        # 4c. Future Feature Target
        future_feat_target = None
        if self.use_future_feat and batch.get('future_pixel_values') is not None:
            future_feat_target = self.get_future_target_features(batch['future_pixel_values'])

        # print('task_cond:', task_cond)

        # 4d. EMA params for the second forward (fp32 shadow, cast to model dtype;
        #     None makes the loss fall back to detached online parameters)
        ema_params = None
        if self.use_ema and self.use_next_chunk_pred and x1_next is not None \
                and self.ema_helper is not None:
            ema_params = self.ema_helper.get_params(dtype=self.dtype)

        # 5. Flow Matching Loss
        loss, info_dic = calc_flow_matching_loss(
            self.action_model,
            x1=x1,
            dino_features_list=dino_features_list,
            qpos_history=qpos_history,
            task_cond=task_cond,
            time_sampler=self.time_sampler,
            time_mu=self.time_mu,
            time_sigma=self.time_sigma,
            x1_next=x1_next,
            use_next_chunk_pred=self.use_next_chunk_pred,
            lambda_next_chunk=self.lambda_next_chunk,
            future_feat_target=future_feat_target,
            use_future_feat=self.use_future_feat,
            future_feat_attach=self.future_feat_attach,
            lambda_future_feat=self.lambda_future_feat,
            ema_params=ema_params,
        )

        return loss, info_dic
