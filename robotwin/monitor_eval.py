#!/usr/bin/env python3
"""Live monitor for eval.sh runs.

Reads the per-worker status files written by eval_vla_bridge.py after every
episode and prints a live per-task success-rate table:

    task                    state     suc/done/target   rate
    adjust_bottle           running   16/30/50          53.3%
    scan_object             done      34/50             68.0%

Usage (in a second terminal, while eval.sh is running):

    python monitor_eval.py            # refresh every 2s
    python monitor_eval.py 5          # refresh every 5s
"""
import glob
import json
import os
import sys
import time

RESULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_result", "bridge_results"
)


def collect():
    tasks = {}
    # finished tasks: final per-worker results
    for fp in glob.glob(os.path.join(RESULT_DIR, "worker_*.json")):
        try:
            with open(fp) as f:
                for r in json.load(f)["results"]:
                    tasks[r["task_name"]] = {
                        "state": "done",
                        "suc": r["success"],
                        "done_eps": r["test_num"],
                        "target": r["test_num"],
                    }
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    # in-progress tasks: live status files (worker_<i>_status.json)
    for fp in glob.glob(os.path.join(RESULT_DIR, "*_status.json")):
        try:
            with open(fp) as f:
                s = json.load(f)
            tasks[s["task_name"]] = {
                "state": "running",
                "suc": s["suc"],
                "done_eps": s["done"],
                "target": s["target"],
            }
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return tasks


def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    while True:
        tasks = collect()
        total_suc = sum(t["suc"] for t in tasks.values())
        total_done = sum(t["done_eps"] for t in tasks.values())

        lines = [
            f"=== RoboTwin Eval Progress ({time.strftime('%H:%M:%S')}, "
            f"refresh {interval:g}s, Ctrl-C to quit) ===",
            f"{'task':<28}{'state':<10}{'suc/done/target':<18}rate",
        ]
        for name in sorted(tasks):
            t = tasks[name]
            if t["done_eps"] > 0:
                rate = f"{t['suc']/t['done_eps']*100:.1f}%"
            else:
                rate = "-"
            if t["state"] == "done":
                progress = f"{t['suc']}/{t['done_eps']}"
            else:
                progress = f"{t['suc']}/{t['done_eps']}/{t['target']}"
            lines.append(f"{name:<28}{t['state']:<10}{progress:<18}{rate}")

        if total_done > 0:
            lines.append(
                f"\noverall: {total_suc}/{total_done} "
                f"({total_suc/total_done*100:.1f}%) over finished episodes"
            )
        if not tasks:
            lines.append(f"(no status files in {RESULT_DIR} yet)")

        os.system("clear")
        print("\n".join(lines), flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
