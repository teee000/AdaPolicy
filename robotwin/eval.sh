#!/bin/bash
# eval.sh - RoboTwin 2.0 evaluation entry point for DINOV3-HDF5-Myself-3-ADWM
#
# How it works: the task list is split into NUM_WORKERS shards (round-robin).
# Each shard is evaluated sequentially by one independent subprocess running
# RoboTwin-side eval_vla_bridge_adwm.py (cwd = RoboTwin root, so RoboTwin's
# internal relative paths work without modifying any RoboTwin code).
# Finally the per-worker result JSONs are aggregated.
#
# The bridge script eval_vla_bridge_adwm.py lives in THIS project (single
# source of truth) and is synced into $ROBOTWIN_ROOT automatically on every
# run - no manual copy needed.
#
# Usage:
#   conda activate RoboTwin          # or: PYTHON=/path/to/python bash eval.sh
#   bash eval.sh
#
# Logs:    eval_result/logs/worker_<i>.log
# Results: eval_result/bridge_results/worker_<i>.json and result.txt in each
#          task's output directory
set -u

VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="/home/yf/Desktop/Code/VLA/RoboTwin/RoboTwin"
BRIDGE_SCRIPT="$ROBOTWIN_ROOT/eval_robotwin_bridge.py"
PYTHON="${PYTHON:-python}"

# ===== Eval config =====
CKPT_SETTING="ig_2026-10-22_09-19-21"
CHECKPOINT_EP="16"
INFER_MODULE="robotwin_infer"   

USE_EMA_WEIGHTS=0                   # checkpoint EMA 
TASK_CONFIG="demo_clean"            # demo_clean / demo_randomized
INSTRUCTION_TYPE="unseen"           # unseen / seen
NORM_STATS_PATH="utils/stat-500-all.json"
CONFIG_PATH="configs/robotwin_ema.yaml"   # use config.yaml of checkpoint file first
SEED=0
TEST_NUM=50                          # 
NUM_WORKERS=4                        # parallel processes (each holds its own GPU memory)

# ===== Task list (commented-out tasks are skipped) =====
TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock

    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic

    hanging_mug
    lift_pot

    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop

    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right

    place_bread_basket
    place_bread_skillet
    place_burger_fries

    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup

    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand

    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin

    put_object_cabinet

    rotate_qrcode
    scan_object
    shake_bottle
    shake_bottle_horizontally
    stack_blocks_three

    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

# ===== Sync the bridge script into the RoboTwin root =====
if [ ! -d "$ROBOTWIN_ROOT/envs" ]; then
    echo "ROBOTWIN_ROOT=$ROBOTWIN_ROOT does not look like a RoboTwin root (no envs/), fix it in this script"
    exit 1
fi
cp "$VLA_ROOT/eval_vla_bridge_adwm.py" "$BRIDGE_SCRIPT"
echo "[sync] eval_vla_bridge_adwm.py -> $BRIDGE_SCRIPT"

SAVE_ROOT="$VLA_ROOT/eval_result"
LOG_DIR="$SAVE_ROOT/logs"
RESULT_DIR="$SAVE_ROOT/bridge_results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
rm -f "$RESULT_DIR"/worker_*.json "$RESULT_DIR"/*_status.json

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "TASKS is empty, uncomment at least one task in this script"
    exit 1
fi

EXTRA_ARGS=()
if [ "$USE_EMA_WEIGHTS" = "1" ]; then
    EXTRA_ARGS+=(--use_ema_weights)
fi

# ===== Round-robin sharding and parallel launch =====
pids=()
worker_ids=()
for ((w = 0; w < NUM_WORKERS; w++)); do
    chunk=()
    for ((i = w; i < ${#TASKS[@]}; i += NUM_WORKERS)); do
        chunk+=("${TASKS[i]}")
    done
    [ ${#chunk[@]} -eq 0 ] && continue

    echo "[worker $w] tasks: ${chunk[*]}"
    (
        cd "$ROBOTWIN_ROOT" || exit 1
        "$PYTHON" "$BRIDGE_SCRIPT" \
            --task_names "${chunk[@]}" \
            --task_config "$TASK_CONFIG" \
            --instruction_type "$INSTRUCTION_TYPE" \
            --ckpt_setting "$CKPT_SETTING" \
            --checkpoint_ep "$CHECKPOINT_EP" \
            --infer_module "$INFER_MODULE" \
            --model_base_path "$VLA_ROOT" \
            --norm_stats_path "$NORM_STATS_PATH" \
            --config_path "$CONFIG_PATH" \
            --vla_root "$VLA_ROOT" \
            --save_root "$SAVE_ROOT" \
            --result_json "$RESULT_DIR/worker_$w.json" \
            --seed "$SEED" \
            --test_num "$TEST_NUM" \
            "${EXTRA_ARGS[@]}" \
            > "$LOG_DIR/worker_$w.log" 2>&1
    ) &
    pids+=($!)
    worker_ids+=($w)
done

# ===== Wait for all workers =====
fail=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "\033[91m[worker ${worker_ids[$idx]}] exited abnormally, check $LOG_DIR/worker_${worker_ids[$idx]}.log\033[0m"
        fail=1
    fi
done

# ===== Aggregate results from all workers =====
"$PYTHON" - "$RESULT_DIR" <<'EOF'
import glob, json, os, sys

files = sorted(glob.glob(os.path.join(sys.argv[1], "worker_*.json")))
results = []
for fp in files:
    with open(fp) as f:
        results.extend(json.load(f)["results"])

print("\n" + "=" * 50)
total_suc, total_tests = 0, 0
for r in results:
    total_suc += r["success"]
    total_tests += r["test_num"]
    print(f" - {r['task_name']}: {r['success']}/{r['test_num']} ({r['success_rate']*100:.1f}%)")

if total_tests > 0:
    print(f"\n>>> Avg Success Rate: {total_suc}/{total_tests} "
          f"({total_suc/total_tests*100:.2f}%) <<<")
else:
    print("No task produced results, check the worker logs")
print("=" * 50 + "\n")
EOF

[ $fail -ne 0 ] && echo "WARNING: some workers exited abnormally, the summary above may be incomplete (logs in $LOG_DIR)"
exit 0
