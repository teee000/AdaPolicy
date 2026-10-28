#!/bin/bash
# Entry script for EMA imitation-learning training/evaluation
# (train_metaworld_ema.py + configs/metaworld.yaml)
#
# Usage:
#   ./run_train_ema.sh smoke            # single-task (dial-turn-v3) smoke test, verifies the pipeline
#   ./run_train_ema.sh all              # joint training on 50 tasks (default, ~3h on RTX 5090)
#   ./run_train_ema.sh all 40           # joint training, early stop after epoch 40
#                                       # (LR still follows the 60-epoch cosine decay; a checkpoint is saved at epoch 40)
#   ./run_train_ema.sh eval <ckpt> [task] [n_episodes] [exp_name]
#                                       # evaluate a checkpoint (model structure is identical, reuse the eval script)
#                                       # when exp_name is given, results are appended to
#                                       #   docs/il_ablation_future_feat.xlsx (scripts/log_eval_to_excel.py)
#   ./run_train_ema.sh resume <ckpt>    # resume training (incl. EMA state)
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:-all}
STOP_AFTER_EPOCH=${STOP_AFTER_EPOCH:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}

mkdir -p log
LOG="log/train_ema_${MODE}_$(date +%Y%m%d_%H%M%S).log"

case "$MODE" in
  smoke)
    echo "[smoke] dial-turn-v3 single-task smoke test, log: $LOG"
    python -u train_metaworld_ema.py \
        --config configs/metaworld.yaml \
        --tasks dial-turn-v3 2>&1 | tee "$LOG"
    ;;
  all)
    # The second positional arg or the STOP_AFTER_EPOCH env var sets early stopping
    STOP_AFTER_EPOCH=${2:-$STOP_AFTER_EPOCH}
    EXTRA_ARGS=()
    if [ -n "$STOP_AFTER_EPOCH" ]; then
      EXTRA_ARGS+=(--stop_after_epoch "$STOP_AFTER_EPOCH")
      echo "[all] joint training on 50 tasks, early stop after epoch $STOP_AFTER_EPOCH, log: $LOG"
    else
      echo "[all] joint training on 50 tasks, log: $LOG"
    fi
    python -u train_metaworld_ema.py \
        --config configs/metaworld.yaml \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
    ;;
  resume)
    CKPT=${2:?usage: ./run_train_ema.sh resume <checkpoint.pt>}
    echo "[resume] resuming from $CKPT, log: $LOG"
    python -u train_metaworld_ema.py \
        --config configs/metaworld.yaml \
        --resume "$CKPT" 2>&1 | tee "$LOG"
    ;;
  eval)
    CKPT=${2:?usage: ./run_train_ema.sh eval <checkpoint.pt> [task] [n_episodes] [exp_name]}
    TASK=${3:-all}
    NEP=${4:-50}
    EXP_NAME=${5:-}
    # Prefer the config.yaml saved next to the checkpoint (the actual training config);
    # otherwise fall back to configs/metaworld.yaml (structure, e.g. chunk size, must match the checkpoint)
    CKPT_DIR=$(dirname "$CKPT")
    if [ -f "$CKPT_DIR/config.yaml" ]; then CFG="$CKPT_DIR/config.yaml"; else CFG="configs/metaworld.yaml"; fi
    echo "[eval] ckpt=$CKPT task=$TASK episodes=$NEP config=$CFG exp_name=${EXP_NAME:-<not-logged>}"
    python -u scripts/eval_metaworld_parallel.py \
        --config "$CFG" \
        --checkpoint "$CKPT" \
        --task "$TASK" --num-episodes "$NEP" --num-envs "$NEP" 2>&1 | tee "$LOG"
    if [ -n "$EXP_NAME" ]; then
      EVAL_LOG=$(grep -oP 'Log saved to: \K\S+' "$LOG" | tail -1 || true)
      if [ -n "$EVAL_LOG" ]; then
        python scripts/log_eval_to_excel.py \
            --eval-log "$EVAL_LOG" --name "$EXP_NAME" --checkpoint "$CKPT"
      else
        echo "[eval] eval log path not found, skipping Excel logging" >&2
      fi
    fi
    ;;
  *)
    echo "unknown mode: $MODE (choose from: smoke | all | resume | eval)" >&2
    exit 1
    ;;
esac
