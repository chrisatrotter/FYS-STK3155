# code/project3.py
"""
Master driver for Project 3 – FYS-STK3155 / FYS4155
Run everything with: python code/project3.py --dataset both --part all
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List
from helpers.download_datasets import ensure_datasets

# Import our beautiful divider
from utils import breakpoint

# ──────────────────────────────────────────────────────────────
# Path Setup
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent           # code/
TASKS_DIR = SCRIPT_DIR / "tasks"                       # code/tasks/
ROOT_DIR = SCRIPT_DIR.parent                           # Project root
CODE_DIR = SCRIPT_DIR                                  # code/ → needed for imports

# Build clean environment with correct PYTHONPATH
base_env = os.environ.copy()
current_pythonpath = base_env.get("PYTHONPATH", "")
separator = ":" if current_pythonpath else ""
base_env["PYTHONPATH"] = f"{current_pythonpath}{separator}{CODE_DIR}"


def run_script(script_name: str, extra_args: List[str] = None):
    extra_args = extra_args or []
    script_path = TASKS_DIR / script_name

    if not script_path.exists():
        print(f"Script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)] + extra_args
    cmdprint = [script_name] + extra_args

    breakpoint()
    print(f" RUNNING → {script_name.replace('.py', '').upper()}")
    print(f" Command: {' '.join(cmdprint)}")
    breakpoint()

    result = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        env=base_env,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print("Warnings / Errors:")
        print(result.stderr.rstrip())

    if result.returncode != 0:
        print(f"FAILED: {script_name} (exit code {result.returncode})")
        sys.exit(result.returncode)


def run_dataset(dataset: str, parts: List[str], seed: int, epochs: int, batch: int):
    base_args = [
        f"--dataset={dataset}",
        f"--seed={seed}",
        f"--epochs={epochs}",
        f"--batch={batch}"
    ]
    script_map = {
        "a": "part_a_data_exploration.py",   # Fixed: was wrong name before
        "c": "part_c_training.py",
        "d": "part_d_results.py",
    }

    for part in parts:
        run_script(script_map[part], base_args + [f"--part={part}"])


def main():
    parser = argparse.ArgumentParser(
        description="Project 3 – Applied Data Analysis and Machine Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", choices=["power", "trade", "both"], default="both")
    parser.add_argument("--part", choices=["a", "c", "d", "all"], default="all")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    ensure_datasets()

    datasets = ["power", "trade"] if args.dataset == "both" else [args.dataset]
    parts = ["a", "c", "d"] if args.part == "all" else [args.part]

    breakpoint()
    print(" PROJECT 3 – FYS-STK3155 / FYS4155")
    print(" Applied Data Analysis and Machine Learning")
    print(f" Running → Datasets: {', '.join(d.upper() for d in datasets)}")
    print(f"           Parts:     {', '.join(p.upper() for p in parts)}")
    print(f" Seed: {args.seed} | Epochs: {args.epochs} | Batch: {args.batch}")
    breakpoint()

    for dataset in datasets:
        run_dataset(dataset, parts, args.seed, args.epochs, args.batch)

    breakpoint()
    print(" PROJECT 3 COMPLETED SUCCESSFULLY!")
    print(" All tasks finished. Figures, models, and predictions saved.")
    print(" Ready for your final report!")
    breakpoint()


if __name__ == "__main__":
    main()
