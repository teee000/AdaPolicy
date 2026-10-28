"""
AdaPolicy imitation-learning training entry.

Next-chunk prediction: in addition to flow-matching denoising of the first
chunk, a second forward predicts the next chunk; the second forward uses EMA
parameters (EMAParams, updated after every optimizer step, fp32 shadow copy).
The checkpoint additionally saves/restores ema_state_dict.

Usage:
    python train_metaworld_ema.py --config configs/metaworld.yaml
"""
import os
import sys
import torch
import logging
import argparse
import time
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader.dataset import collate_fn, create_dataset
from utils.train_utils import count_parameters
from models.model_runner_ema import ModelFactory, VLAWrapper, EMAParams


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


class LossLogger:
    def __init__(self, log_dir="log/loss"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"train_loss_{timestamp}.csv")
        with open(self.log_file, 'w') as f:
            f.write("Epoch,Step,Global_Step,Loss\n")

    def log(self, epoch, step, global_step, loss):
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{step},{global_step},{loss:.6f}\n")


def build_train_config_from_yaml(cfg):
    """get VLAWrapper config params"""
    t = cfg.training
    ff = cfg.model.get('future_feat', {})
    return {
        'time_mu': t.time_mu,
        'time_sigma': t.time_sigma,
        'use_vel_weight': t.use_vel_weight,
        'vel_weight_alpha': t.vel_weight_alpha,
        'vel_weight_sigma': t.vel_weight_sigma,
        'use_next_chunk_pred': t.use_next_chunk_pred,
        'lambda_next_chunk': t.lambda_next_chunk,
        'use_future_feat': ff.get('enabled', False) if ff else False,
        'future_feat_attach': ff.get('attach', 'cond_tokens') if ff else 'cond_tokens',
        'lambda_future_feat': t.get('lambda_future_feat', 0.0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA Training Script (DINOv3, EMA Next-Chunk)")
    parser.add_argument("--config", type=str, default="./configs/metaworld.yaml",
                        help="Path to config file")
    parser.add_argument("--norm_stats_path", type=str, default="./utils/stat-metaworld.json",
                        help="Path to normalization stats")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vla",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from "
                             "(restores model + optimizer + scheduler + epoch + ema)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Task subset to train on, e.g. --tasks assembly-v3 "
                             "or --tasks assembly-v3 push-v3. Omit or pass 'all' "
                             "for joint training on every task in the dataset dir.")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="EMA decay for the next-chunk second forward "
                             "(default: training.ema_decay in config, else 0.98)")
    parser.add_argument("--stop_after_epoch", type=int, default=0,
                        help="Early stop after this epoch (checkpoint is still saved). "
                             "0 = disabled. LR schedule still uses training.epochs, "
                             "so stopping early keeps the 60-epoch cosine decay.")
    args = parser.parse_args()

    # =========================================================================
    # 1. Config
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    logger.info(f"Using device: {device}, Precision: {dtype}")

    config = load_config(args.config)

    # Task subset selection: CLI overrides; None / 'all' = joint training on all tasks.
    task_names = None
    if args.tasks and args.tasks != ["all"]:
        task_names = sorted(args.tasks)
        config.dataset.task_names = task_names

    epochs = config.training.epochs
    stop_after_epoch = args.stop_after_epoch
    if stop_after_epoch and stop_after_epoch >= epochs:
        logger.warning(f"--stop_after_epoch={stop_after_epoch} >= epochs={epochs}, ignored")
        stop_after_epoch = 0
    grad_accum_steps = config.training.grad_accum_steps
    save_interval_epoch = config.training.save_interval_epoch
    batch_size = config.training.batch_size
    grad_clip_norm = config.training.grad_clip_norm
    lr = config.training.learning_rate
    lr_min = config.training.lr_min
    use_next_chunk_pred = config.training.use_next_chunk_pred
    ema_decay = args.ema_decay
    if ema_decay is None:
        ema_decay = float(config.training.get('ema_decay', 0.98))

    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | "
                f"grad_accum: {grad_accum_steps} | save_every: {save_interval_epoch} ep"
                + (f" | early stop after epoch {stop_after_epoch}" if stop_after_epoch else ""))
    logger.info(f"Tasks: {'ALL (joint multi-task)' if task_names is None else task_names}")
    logger.info(f"DINOv3 feat_layers: {list(config.model.vision_encoder.feat_layers)} | "
                f"include_cls_register: {config.model.vision_encoder.include_cls_register}")
    logger.info(f"Next-Chunk Pred: {'ENABLED' if use_next_chunk_pred else 'DISABLED'} "
                f"(lambda={config.training.lambda_next_chunk})")
    logger.info(f"EMA for next-chunk second forward: decay={ema_decay}")
    logger.info(f"Vel-Weight: {'ENABLED' if config.training.use_vel_weight else 'DISABLED'}")

    # =========================================================================
    # 2. Dataset / DataLoader
    # =========================================================================
    train_dataset = create_dataset(config, val=False, use_next_chunk=use_next_chunk_pred)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"Dataset Size: {len(train_dataset)} | Batches per Epoch: {len(train_dataloader)}")

    # =========================================================================
    # 3. Model
    # =========================================================================
    logger.info(">>> Initializing VLA (DINOv3)")

    vision_encoder, dino_hidden_size, num_register_tokens, patch_size = ModelFactory.create_vision_encoder(
        config.model.vision_encoder.checkpoint_path,
        dtype, device,
    )

    feat_layers = list(config.model.vision_encoder.feat_layers)
    num_dino_layers = len(feat_layers)

    use_task_cond = config.model.get('use_task_cond', False)
    task_cond_dim = dino_hidden_size if use_task_cond else None
    logger.info(f"Task Condition: {'ENABLED' if use_task_cond else 'DISABLED'}"
                + (f" (dim={task_cond_dim}, dir={config.dataset.get('task_cond_dir', None)})" if use_task_cond else ""))

    ff_cfg = config.model.get('future_feat', {})
    use_future_feat = ff_cfg.get('enabled', False) if ff_cfg else False
    logger.info(f"Future-Feat Pred: {'ENABLED' if use_future_feat else 'DISABLED'}"
                + (f" (attach={ff_cfg.get('attach', 'cond_tokens')}, "
                   f"target_layer={ff_cfg.get('target_layer', -1)}, "
                   f"lambda={config.training.get('lambda_future_feat', 0.0)})" if use_future_feat else ""))

    action_model = ModelFactory.create_action_model(
        config,
        dino_hidden_size=dino_hidden_size,
        num_dino_layers=num_dino_layers,
        task_cond_dim=task_cond_dim,
        patch_size=patch_size,
    )

    action_model.to(device, dtype=dtype)
    action_model.train()

    count_parameters(action_model, model_name="Action Model (Trainable)")

    train_config_dict = build_train_config_from_yaml(config)

    model = VLAWrapper(
        vision_encoder=vision_encoder,
        action_model=action_model,
        time_sampler=config.training.time_sampler,
        feat_layers=feat_layers,
        include_cls_register=config.model.vision_encoder.include_cls_register,
        num_register_tokens=num_register_tokens,
        device=device,
        dtype=dtype,
        norm_stats_path=args.norm_stats_path,
        train_config=train_config_dict,
        future_feat_target_layer=ff_cfg.get('target_layer', -1) if ff_cfg else -1,
    )

    # =========================================================================
    # 4. Optimizer / Scheduler
    # =========================================================================
    optimizer = AdamW(
        action_model.parameters(),
        lr=lr,
        betas=tuple(config.training.betas),
        weight_decay=config.training.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_dataloader),
        eta_min=lr_min,
    )

    # =========================================================================
    # 5. Resume
    # =========================================================================
    start_epoch = 0
    global_step = 0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        logger.info(f">>> Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)

        msg = action_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        logger.info(f"Model loaded. missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            steps_done = ckpt['epoch'] * len(train_dataloader)
            for _ in range(steps_done):
                scheduler.step()
            logger.warning("Old checkpoint without scheduler_state_dict; "
                           "scheduler advanced manually (lr may drift slightly).")

        start_epoch = ckpt['epoch']
        global_step = ckpt.get('global_step',
                               start_epoch * len(train_dataloader) // grad_accum_steps)
        logger.info(f"Resumed at epoch={start_epoch}, global_step={global_step}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # [EMA] Create the EMA parameter copy (must be after weight loading / resume
    # so the initial values are correct), used for the second forward of
    # next-chunk prediction. With training.use_ema=false no ema_helper is
    # attached and the second forward falls back to detached online params
    # (ablation switch).
    use_ema = bool(config.training.get('use_ema', True))
    if use_ema:
        ema_helper = EMAParams(action_model, decay=ema_decay)
        if args.resume and 'ema_state_dict' in ckpt:
            ema_helper.load_state_dict(ckpt['ema_state_dict'])
            logger.info("Loaded EMA params from checkpoint.")
        else:
            logger.info("Initialized EMA params from current model weights.")
        model.ema_helper = ema_helper
    else:
        ema_helper = None
        logger.info("EMA DISABLED (use_ema=false): second forward uses detached online params.")

    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_tag = "all" if task_names is None else (
        task_names[0] if len(task_names) == 1 else f"{len(task_names)}tasks")
    run_save_dir = os.path.join(args.save_dir, f"ema_ig_{task_tag}_{timestamp}")
    os.makedirs(run_save_dir, exist_ok=True)

    OmegaConf.save(config, os.path.join(run_save_dir, "config.yaml"))
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    loss_logger = LossLogger(log_dir="log/loss")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        start_time = time.time()

        for step, batch in enumerate(train_dataloader):
            with torch.amp.autocast('cuda', dtype=dtype):
                loss, info_dic = model(batch)
                loss = loss / grad_accum_steps

            loss.backward()

            current_step_loss = loss.item() * grad_accum_steps
            epoch_loss += current_step_loss

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # [EMA] Update the EMA copy after the parameter update
                if ema_helper is not None:
                    ema_helper.update(action_model)

                loss_logger.log(epoch + 1, step + 1, global_step, current_step_loss)

                if global_step % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    log_msg = (
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Step [{step+1}/{len(train_dataloader)}] "
                        f"Loss: {info_dic['loss_mse'] * grad_accum_steps:.4f} "
                        f"LR: {current_lr:.2e} "
                    )
                    if use_next_chunk_pred:
                        log_msg += f"NextChunk: {info_dic.get('loss_next_chunk', 0.0):.4f} "
                    if use_future_feat:
                        log_msg += f"FutureFeat: {info_dic.get('loss_future_feat', 0.0):.4f} "
                    logger.info(log_msg)

        avg_loss = epoch_loss / len(train_dataloader)
        elapsed = time.time() - start_time
        logger.info(f"=== Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s ===")

        if (epoch + 1) % save_interval_epoch == 0 or (epoch + 1) == epochs \
                or (stop_after_epoch and (epoch + 1) == stop_after_epoch):
            ckpt_name = f"checkpoint_epoch_{epoch+1}.pt"
            ckpt_path = os.path.join(run_save_dir, ckpt_name)
            save_dict = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': action_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema_helper.state_dict() if ema_helper is not None else None,
                'loss': avg_loss,
            }
            torch.save(save_dict, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

        if stop_after_epoch and (epoch + 1) >= stop_after_epoch:
            logger.info(f"Early stop: reached stop_after_epoch={stop_after_epoch} "
                        f"(LR schedule was computed for {epochs} epochs).")
            break

    logger.info("Training Complete.")
