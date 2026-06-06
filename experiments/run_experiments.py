"""
run_experiments.py

Generated using Claude Sonnett 4.6

Wrapper that runs train_hybinn.py for each (model configuration, seed) pair.
Each run is saved to runs/{model_name}/seed_{seed}/.

Usage:
    python run_experiments.py                          # run all configs, all seeds
    python run_experiments.py --configs binn_clinical  # run one config, all seeds
    python run_experiments.py --seeds 0 1 2            # run all configs, specific seeds
"""

import argparse
import os
import subprocess
import sys
import time


# ---- Experiment definitions ----

ALL_CONFIGS = {
    "binn_only":      ["binn"],
    "clinical_only":  ["clinical"],
    "binn_clinical":  ["binn", "clinical"],
    "full_hybinn":    ["binn", "gene", "clinical"],
}

ALL_SEEDS = list(range(10))

BASE_RUN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "runs"))
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train_hybinn.py")


# ---- Runner ----

def run_one(model_name, branches, seed, dry_run=False):
    run_dir = os.path.join(BASE_RUN_DIR, model_name, f"seed_{seed}")

    # Skip if results.json already exists (allows resuming interrupted runs)
    results_path = os.path.join(run_dir, "results.json")
    if os.path.exists(results_path):
        print(f"  [SKIP] {model_name}/seed_{seed} — results.json already exists")
        return True

    os.makedirs(run_dir, exist_ok=True)

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--branches", *branches,
        "--seed",     str(seed),
        "--run_dir",  run_dir,
        "--run_name", model_name,
    ]

    print(f"  [RUN]  {model_name}/seed_{seed}")
    if dry_run:
        print(f"         cmd: {' '.join(cmd)}")
        return True

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [FAIL] {model_name}/seed_{seed} (exit code {result.returncode})")
        return False
    else:
        print(f"  [DONE] {model_name}/seed_{seed} ({elapsed/60:.1f} min)")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs",  nargs="+", default=list(ALL_CONFIGS.keys()),
                        choices=list(ALL_CONFIGS.keys()),
                        help="Which model configs to run (default: all)")
    parser.add_argument("--seeds",    nargs="+", type=int, default=ALL_SEEDS,
                        help="Which seeds to run (default: 0-9)")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    total   = len(args.configs) * len(args.seeds)
    success = 0
    failed  = []

    print(f"\nRunning {total} experiments ({len(args.configs)} configs × {len(args.seeds)} seeds)")
    print(f"Output directory: {BASE_RUN_DIR}\n")

    for model_name in args.configs:
        branches = ALL_CONFIGS[model_name]
        print(f"\n[{model_name}]  branches={branches}")
        for seed in args.seeds:
            ok = run_one(model_name, branches, seed, dry_run=args.dry_run)
            if ok:
                success += 1
            else:
                failed.append(f"{model_name}/seed_{seed}")

    print(f"\n{'='*50}")
    print(f"Finished: {success}/{total} runs succeeded")
    if failed:
        print(f"Failed runs:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()