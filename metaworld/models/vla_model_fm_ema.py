"""
EMA next-chunk prediction (the _ema variant of models/vla_model_fm.py).

The only difference: the second forward for next-chunk uses
torch.func.functional_call with **EMA parameters** (ema_params, maintained by
model_runner_ema.EMAParams as an fp32 shadow copy) instead of the original's
on-the-fly detach of the online parameters. EMA params carry no gradient, so
the loss_next_chunk gradient flows back to the first forward's computation
graph only through obs_tokens_evolved, training the online parameters.

The model structure (VLAModel etc.) is identical to the original and reused
unchanged.
"""
import torch
import torch.nn.functional as F

from .vla_model_fm import (  # noqa: F401  (re-export, structure identical to the original)
    VLAModel,
    future_feature_cosine_loss,
)


def calc_flow_matching_loss(
    model,
    x1,
    dino_features_list,
    qpos_history,
    task_cond=None,
    time_sampler="uniform",
    time_mu=0.0,
    time_sigma=1.0,
    use_velocity_weighting=False,
    alpha=0.2,
    sigma=0.01,
    # Next-Chunk Prediction
    x1_next=None,
    use_next_chunk_pred=False,
    lambda_next_chunk=0.5,
    ema_params=None,               # EMA parameter dict for the second forward
    # Future Feature Prediction
    future_feat_target=None,
    use_future_feat=False,
    future_feat_attach="cond_tokens",
    lambda_future_feat=0.5,
):
    """
    Flow Matching Loss (Next-Chunk Prediction + Task Condition + Future-Feature Prediction)
    """
    device = x1.device
    bs = x1.shape[0]

    # 1. noise sample x0
    x0 = torch.randn_like(x1)

    # 2. timestamp t
    if time_sampler == "uniform":
        t = torch.rand(bs, device=device)
    elif time_sampler == "logit_normal":
        normal_samples = torch.randn(bs, device=device)
        normal_samples = normal_samples * time_sigma + time_mu
        t = torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unsupported time_sampler: {time_sampler}")

    # 3. t
    t_expand = t.view(bs, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1

    # 4. target v
    target_v = x1 - x0

    # 5. first forward
    preds = model(t,
                  noisy_actions=x_t,
                  dino_features_list=dino_features_list,
                  task_cond=task_cond,
                  qpos_history=qpos_history)

    pred_v_final = preds["final_pred"]
    obs_tokens_evolved = preds["obs_tokens_evolved"]

    # ==================== Velocity Weighting (sample-level) ====================
    if use_velocity_weighting:
        velocities = torch.diff(x1[:, :, :-1], dim=1)
        mean_vel = torch.mean(torch.abs(velocities), dim=(1, 2))
        weights = alpha + (1.0 - alpha) * torch.exp(-(mean_vel ** 2) / (2 * sigma ** 2))
    else:
        weights = torch.ones(bs, device=device)
        mean_vel = torch.zeros(bs, device=device)

    # ==================== Final Loss ====================
    loss_final_unreduced = F.mse_loss(pred_v_final, target_v, reduction='none')
    loss_final_per_sample = torch.mean(loss_final_unreduced, dim=(1, 2))
    loss_mse = torch.mean(loss_final_per_sample * weights)

    # ==================== Next-Chunk Prediction Loss ====================
    loss_next_chunk = torch.tensor(0.0, device=device)

    if use_next_chunk_pred and x1_next is not None:
        x0_next = torch.randn_like(x1_next)

        if time_sampler == "uniform":
            t_next = torch.rand(bs, device=device)
        elif time_sampler == "logit_normal":
            normal_samples_next = torch.randn(bs, device=device)
            normal_samples_next = normal_samples_next * time_sigma + time_mu
            t_next = torch.sigmoid(normal_samples_next)
        else:
            t_next = torch.rand(bs, device=device)

        t_next_expand = t_next.view(bs, 1, 1)
        x_t_next = (1 - t_next_expand) * x0_next + t_next_expand * x1_next
        target_v_next = x1_next - x0_next

        # second forward: EMA params (requires_grad=False) receive no direct
        # gradient from loss_next_chunk; obs_tokens_evolved stays in the graph,
        # so gradients flow back to the first forward
        if ema_params is None:
            # fallback: use a detached copy of the current params when no EMA
            # params are provided (same behavior as the original)
            ema_params = {k: v.detach() for k, v in model.named_parameters()}

        preds_next = torch.func.functional_call(
            model,
            ema_params,
            args=(t_next,),
            kwargs={
                'noisy_actions': x_t_next,
                'obs_tokens_evolved': obs_tokens_evolved,
            }
        )

        pred_v_next = preds_next["final_pred"]
        loss_next_unreduced = F.mse_loss(pred_v_next, target_v_next, reduction='none')
        loss_next_chunk = torch.mean(loss_next_unreduced)

    # ==================== Future Feature Prediction Loss ====================
    loss_future_feat = torch.tensor(0.0, device=device)

    if use_future_feat and future_feat_target is not None:
        if future_feat_attach == "adapter":
            cond_tokens = preds["adapter_cond_tokens"]
            assert cond_tokens is not None, \
                "future_feat_attach='adapter' forward without adapter_cond_tokens (dino_features_list null?)"
        else:  # "cond_tokens"
            cond_tokens = obs_tokens_evolved

        # gradient back to adapter / DiT; target from DINO
        pred_future = model.future_feat_decoder(cond_tokens)        # (B, M, D_dino)
        loss_future_feat = future_feature_cosine_loss(pred_future, future_feat_target)

    # ==================== Total Loss ====================
    loss = loss_mse + lambda_next_chunk * loss_next_chunk + lambda_future_feat * loss_future_feat

    return loss, {
        "pred_v": pred_v_final,
        "target_v": target_v,
        "mean_chunk_vel": mean_vel.mean().item(),
        "mean_loss_weight": weights.mean().item(),
        "loss_mse": loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse,
        "loss_next_chunk": loss_next_chunk.item() if isinstance(loss_next_chunk, torch.Tensor) else loss_next_chunk,
        "loss_future_feat": loss_future_feat.item() if isinstance(loss_future_feat, torch.Tensor) else loss_future_feat,
    }
