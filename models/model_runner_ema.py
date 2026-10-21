"""
EMA model runner (the _ema variant of models/model_runner.py).

- ModelFactory: identical to the original, re-exported directly
- EMAParams: fp32 EMA shadow copy of the action-model parameters, used for the
  second forward of next-chunk prediction (following the train_two_stage8_ema.py
  / 8_ema design)
- VLAWrapper: subclasses the original; passes the EMA params to the _ema
  calc_flow_matching_loss in forward; ema_helper can be assigned after
  construction (model.ema_helper = EMAParams(...))

Note: the shadow is stored in fp32. Storing it in bf16 directly can make the
per-step (1-decay) relative update smaller than bf16 rounding precision,
leaving the EMA stuck in a rounding dead zone.
"""
import logging

import torch

from .model_runner import ModelFactory, VLAWrapper as _VLAWrapperBase  # noqa: F401
from .vla_model_fm_ema import calc_flow_matching_loss as calc_flow_matching_loss_ema

logger = logging.getLogger(__name__)


class EMAParams:
    """
    Maintains an EMA (Exponential Moving Average) copy of the action-model
    parameters, used for the second forward of next-chunk prediction.

    Update rule: shadow = decay * shadow + (1 - decay) * param
    Call update() only after optimizer.step().
    """

    def __init__(self, model, decay=0.98):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.named_parameters()}
        logger.info(f"EMAParams initialized: decay={decay}, num_params={len(self.shadow)}")

    @torch.no_grad()
    def update(self, model):
        """Call after optimizer.step() to track the online parameters"""
        for k, v in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def get_params(self, dtype=None):
        """Return a parameter dict usable with torch.func.functional_call (no grad)"""
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


class VLAWrapper(_VLAWrapperBase):
    """Original VLAWrapper + EMA next-chunk support."""

    def __init__(self, *args, ema_helper=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ema_helper = ema_helper

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

        # 4d. EMA params for the second forward (None -> original detach fallback)
        ema_params = None
        if self.use_next_chunk_pred and x1_next is not None and self.ema_helper is not None:
            ema_params = self.ema_helper.get_params(dtype=self.dtype)

        # 5. Flow Matching Loss
        loss, info_dic = calc_flow_matching_loss_ema(
            self.action_model,
            x1=x1,
            dino_features_list=dino_features_list,
            qpos_history=qpos_history,
            task_cond=task_cond,
            time_sampler=self.time_sampler,
            time_mu=self.time_mu,
            time_sigma=self.time_sigma,
            use_velocity_weighting=self.use_vel_weight,
            alpha=self.vel_weight_alpha,
            sigma=self.vel_weight_sigma,
            x1_next=x1_next,
            use_next_chunk_pred=self.use_next_chunk_pred,
            lambda_next_chunk=self.lambda_next_chunk,
            ema_params=ema_params,
            future_feat_target=future_feat_target,
            use_future_feat=self.use_future_feat,
            future_feat_attach=self.future_feat_attach,
            lambda_future_feat=self.lambda_future_feat,
        )

        return loss, info_dic
