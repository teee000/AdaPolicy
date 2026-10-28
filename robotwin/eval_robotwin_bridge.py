# eval_vla_bridge_adwm.py
# RoboTwin-side evaluation bridge script for DINOV3-HDF5-Myself-3-ADWM
# (a NEW file on the RoboTwin side - no existing RoboTwin code is modified).
# It is launched as a subprocess by eval.sh in the VLA project (cwd = RoboTwin
# root), evaluates the tasks given by --task_names sequentially within a single
# process, and imports the VLA model (robotwin_infer_3_ema_2.RobotWinInference)
# from the VLA project via --vla_root.
#
# Differences from the LiLa-WAM eval_vla_bridge.py:
#   - inference module is selectable via --infer_module (a module in vla_root
#     exposing RobotWinInference, e.g. robotwin_infer_3_ema_2 for
#     train_dinov3_3_2_ema_2/ema_3 checkpoints, robotwin_infer_3_ema_4 for
#     train_dinov3_3_2_ema_4 checkpoints); default robotwin_infer_3_ema_2
#   - new --use_ema_weights flag: load the checkpoint's ema_state_dict shadow
#     weights for inference instead of model_state_dict
import sys
import os
import subprocess
import matplotlib.pyplot as plt

# RoboTwin internally uses many paths relative to its root directory
# (./assets/..., ./task_config/...), so make sure the cwd is always the
# directory containing this script (the RoboTwin root), no matter where
# the process is launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
import torch
import numpy as np
from pathlib import Path

import json
import yaml
from datetime import datetime
import importlib
import argparse

from description.utils.generate_episode_instructions import *

import logging
logging.getLogger("curobo").setLevel(logging.WARNING)


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "./task_config/_camera_config.yml")
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    checkpoint_ep = usr_args["checkpoint_ep"]
    model_base_path = usr_args["model_base_path"]

    policy_name = 'test_policy'
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    print('args:', args)
    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(usr_args["save_root"]) / task_name / policy_name / task_config / ckpt_setting / current_time
    save_dir.mkdir(parents=True, exist_ok=True)

    args["save_dir"] = str(save_dir)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + ", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + ", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)

    checkpoint_path = os.path.join(
        model_base_path,
        f"checkpoints_vla/{ckpt_setting}/checkpoint_epoch_{checkpoint_ep}.pt",
    )

    auto_config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if os.path.exists(auto_config_path):
        config_path = auto_config_path
        print(f"\033[92m[Config] Auto load checkpoint config: {config_path}\033[0m")
    else:
        config_path = os.path.join(model_base_path, usr_args["config_path"])
        print(f"\033[93m[Config] not fine {auto_config_path}, return to: {config_path}\033[0m")

    norm_stats_path = os.path.join(model_base_path, usr_args["norm_stats_path"])

    # =========================================================================
    # infer init (VLA-project-side model, imported lazily after sys.path setup)
    # =========================================================================
    if usr_args["vla_root"] not in sys.path:
        sys.path.append(usr_args["vla_root"])
    infer_module = importlib.import_module(usr_args["infer_module"])
    RobotWinInference = infer_module.RobotWinInference

    print(f"\033[94m[Model] infer_module: {usr_args['infer_module']}\033[0m")
    print(f"\033[94m[Model] checkpoint: {checkpoint_path}\033[0m")
    print(f"\033[94m[Model] use_ema_weights: {usr_args['use_ema_weights']}\033[0m")
    model = RobotWinInference(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        norm_stats_path=norm_stats_path,
        use_ema_weights=usr_args["use_ema_weights"],
    )

    if hasattr(model, "set_task"):
        model.set_task(task_name)

    # live status file for the monitor script (one per worker, atomically
    # rewritten after every episode)
    if usr_args.get("result_json"):
        args["status_file"] = usr_args["result_json"].replace(".json", "_status.json")

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=usr_args["test_num"],
                                   video_size=video_size,
                                   instruction_type=instruction_type)

    file_path = save_dir / "result.txt"
    with open(file_path, "w") as f:
        f.write(f"Timestamp: {current_time}\n\n")
        f.write(f"Instruction Type: {instruction_type}\n\n")
        f.write(f"Success Rate: {suc_num / usr_args['test_num']}\n")

    return suc_num


def visualize_cameras(obs):
    target_cameras = ['front_camera', 'head_camera']
    images_to_show = []

    if 'observation' not in obs:
        print("\033[91mError: Input dictionary does not contain 'observation' key.\033[0m")
        return

    cam_data = obs['observation']
    for cam_name in target_cameras:
        if cam_name in cam_data and 'rgb' in cam_data[cam_name]:
            img = cam_data[cam_name]['rgb']
            images_to_show.append((cam_name, img))
        else:
            print(f"\033[93mWarning: {cam_name} or its RGB data not found.\033[0m")

    if not images_to_show:
        return

    num_imgs = len(images_to_show)
    fig, axes = plt.subplots(1, num_imgs, figsize=(6 * num_imgs, 5))
    if num_imgs == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, images_to_show):
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if hasattr(img, 'numpy'):
            img = img.numpy()
        ax.imshow(img)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def inspect_observation(obs, indent=0):
    prefix = " " * indent
    keys = sorted(obs.keys())
    for key in keys:
        value = obs[key]
        if isinstance(value, dict):
            print(f"{prefix}Key: \033[1m{key}\033[0m | Type: Dict")
            inspect_observation(value, indent + 4)
        elif isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                min_val = f"{value.min():.4f}"
                max_val = f"{value.max():.4f}"
            else:
                min_val = "N/A"
                max_val = "N/A"
            print(f"{prefix}Key: {key:<20} | Type: np.ndarray | Shape: {str(value.shape):<15} "
                  f"| Min: {min_val:<10} | Max: {max_val:<10} | Dtype: {value.dtype}")
        elif isinstance(value, torch.Tensor):
            if torch.is_floating_point(value) or torch.is_complex(value) or \
               value.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                min_val = f"{value.min().item():.4f}"
                max_val = f"{value.max().item():.4f}"
            else:
                min_val = "N/A"
                max_val = "N/A"
            print(f"{prefix}Key: {key:<20} | Type: torch.Tensor | Shape: {str(list(value.shape)):<15} "
                  f"| Min: {min_val:<10} | Max: {max_val:<10} | Dtype: {value.dtype}")
        elif isinstance(value, list):
            print(f"{prefix}Key: {key:<20} | Type: List (len={len(value)})")
            if len(value) > 0:
                print(f"{prefix}    Element Type: {type(value[0])}")
        else:
            print(f"{prefix}Key: {key:<20} | Type: {type(value).__name__:<10} | Value: {value}")


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        print('instruction:', instruction)

        TASK_ENV.set_instruction(instruction=instruction)

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", video_size,
                    "-framerate", "10",
                    "-i", "-",
                    "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264", "-crf", "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        model.reset()
        print('TASK_ENV.step_lim:', TASK_ENV.step_lim)

        step_counter = 0
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()

            current_action = model.step(observation, instruction)
            TASK_ENV.take_action(current_action, action_type='qpos')

            if TASK_ENV.eval_success:
                succ = True
                break

            step_counter += 1

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
            f"\033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )

        # atomic live-status update for monitor_eval.py
        if args.get("status_file"):
            tmp_file = args["status_file"] + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump({
                    "task_name": task_name,
                    "suc": TASK_ENV.suc,
                    "done": TASK_ENV.test_num,
                    "target": test_num,
                }, f)
            os.replace(tmp_file, args["status_file"])

        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_names", type=str, nargs="+", required=True,
                        help="tasks to evaluate sequentially in this process")
    parser.add_argument("--instruction_type", type=str, default='unseen')   # unseen / seen
    parser.add_argument("--task_config", type=str, default='demo_clean')    # demo_clean / demo_randomized

    parser.add_argument("--ckpt_setting", type=str, required=True)
    parser.add_argument("--checkpoint_ep", type=str, required=True)
    parser.add_argument("--model_base_path", type=str, required=True)
    parser.add_argument("--norm_stats_path", type=str, default='utils/stat-500-all.json')
    parser.add_argument("--config_path", type=str, default='configs/robotwin_3.2_ema_2_all.yaml')
    parser.add_argument("--use_ema_weights", action="store_true",
                        help="load the checkpoint's ema_state_dict shadow weights "
                             "instead of model_state_dict for inference")
    parser.add_argument("--infer_module", type=str, default="robotwin_infer_3_ema_2",
                        help="inference module in vla_root exposing RobotWinInference "
                             "(robotwin_infer_3_ema_2 for ema_2/ema_3 checkpoints, "
                             "robotwin_infer_3_ema_4 for ema_4 checkpoints)")

    parser.add_argument("--vla_root", type=str, required=True,
                        help="VLA project root (used to import --infer_module)")
    parser.add_argument("--save_root", type=str, required=True,
                        help="absolute path of the eval output root directory")
    parser.add_argument("--result_json", type=str, default=None,
                        help="absolute path of the machine-readable result output")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_num", type=int, default=50,
                        help="Number of test episodes per task")

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    from script.test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    results = []
    for t_name in usr_args["task_names"]:
        print(f"\n{'='*20} Start: {t_name} {'='*20}")
        usr_args["task_name"] = t_name
        suc_num = main(usr_args)
        results.append({
            "task_name": t_name,
            "success": suc_num,
            "test_num": usr_args["test_num"],
            "success_rate": suc_num / usr_args["test_num"],
        })
        # append this task's result to the shared summary file right away, so
        # finished tasks are recorded even if the run is interrupted later
        summary_path = os.path.join(usr_args["save_root"], "eval_summary.txt")
        os.makedirs(usr_args["save_root"], exist_ok=True)
        ema_tag = " ema" if usr_args["use_ema_weights"] else ""
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{usr_args['ckpt_setting']} ep{usr_args['checkpoint_ep']}{ema_tag} | "
                f"{usr_args['task_config']}/{usr_args['instruction_type']} | "
                f"{t_name}: {suc_num}/{usr_args['test_num']} "
                f"({suc_num / usr_args['test_num'] * 100:.1f}%)\n"
            )
        # release CUDA cache between tasks to lower the peak memory of this
        # long-running process
        del usr_args["task_name"]
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    for r in results:
        print(f" - {r['task_name']}: {r['success']}/{r['test_num']} ({r['success_rate']*100:.1f}%)")
    total_suc = sum(r["success"] for r in results)
    total_tests = sum(r["test_num"] for r in results)
    if total_tests > 0:
        print(f"\n>>> Worker Avg Success Rate: {total_suc}/{total_tests} "
              f"({total_suc/total_tests*100:.2f}%) <<<")
    print("=" * 50 + "\n")

    # machine-readable results for the caller (eval.sh in the VLA project)
    if usr_args.get("result_json"):
        os.makedirs(os.path.dirname(usr_args["result_json"]), exist_ok=True)
        with open(usr_args["result_json"], "w") as f:
            json.dump({"results": results}, f, indent=2)
