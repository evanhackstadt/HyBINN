"""
aggregate_results.py

Generated using Claude Sonnett 4.6

Reads results.json from all (model, seed) directories and produces:
  - aggregate_summary.csv   : mean ± std across seeds for each model config
  - aggregate_all_seeds.csv : one row per (model, seed) for inspection / plotting

Usage:
    python aggregate_results.py
    python aggregate_results.py --runs_dir /path/to/runs
    python aggregate_results.py --configs binn_only binn_clinical
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


ALL_CONFIGS = ["binn_only", "clinical_only", "binn_clinical", "full_hybinn"]
DEFAULT_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


# ---- Loader ----

def load_seed_results(runs_dir, model_name):
    """Loads all results.json files for one model config, returns list of dicts."""
    model_dir = os.path.join(runs_dir, model_name)
    if not os.path.isdir(model_dir):
        return []

    records = []
    for seed_dir in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, seed_dir, "results.json")
        if not os.path.exists(path):
            print(f"  [MISSING] {model_name}/{seed_dir}/results.json")
            continue
        with open(path) as f:
            r = json.load(f)
        r["model_name"] = model_name
        r["seed_dir"]   = seed_dir
        records.append(r)

    return records


# ---- Formatting helpers ----

def fmt(mean, std, decimals=4):
    """Format as 'mean ± std' string."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def fmt_ci(lower, upper, decimals=4):
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


# ---- Per-seed flat table ----

def flatten_record(r):
    """Converts one results.json dict to a flat row for the all-seeds CSV."""
    row = {
        "model":             r.get("model_name"),
        "seed":              r.get("seed"),
        "n_trainval":        r.get("n_trainval"),
        "n_test":            r.get("n_test"),
        "cv_mean_cindex":    r.get("cv_mean_cindex"),
        "cv_std_cindex":     r.get("cv_std_cindex"),
        "retrain_mean_cindex": r.get("retrain_mean_cindex"),
        "test_cindex":       r.get("test_cindex"),
        "test_loss":         r.get("test_loss"),
        "mean_best_epoch":   r.get("mean_best_epoch"),
    }

    # Bootstrap CI
    boot = r.get("bootstrap_ci", {})
    row["boot_mean"]  = boot.get("mean")
    row["boot_lower"] = boot.get("lower")
    row["boot_upper"] = boot.get("upper")
    row["boot_std"]   = boot.get("std")

    # Branch weights (variable keys depending on model)
    weights = r.get("branch_weights", {})
    for branch, w in weights.items():
        row[f"weight_{branch}"] = w

    return row


# ---- Summary table ----

def summarize_model(records, model_name):
    """Produces a single summary row (mean ± std across seeds) for one model config."""
    if not records:
        return {"model": model_name, "n_seeds": 0}

    flat = [flatten_record(r) for r in records]
    df   = pd.DataFrame(flat)

    row = {
        "model":   model_name,
        "n_seeds": len(df),
    }

    # Core metrics
    for metric in ["cv_mean_cindex", "test_cindex"]:
        if metric in df.columns and df[metric].notna().any():
            row[metric] = fmt(df[metric].mean(), df[metric].std())

    # Bootstrap CI: average the per-seed lower/upper bounds
    if df["boot_lower"].notna().any():
        row["test_cindex_95ci"] = fmt_ci(
            df["boot_lower"].mean(),
            df["boot_upper"].mean()
        )

    # Branch weights
    weight_cols = [c for c in df.columns if c.startswith("weight_")]
    for col in weight_cols:
        if df[col].notna().any():
            row[col] = fmt(df[col].mean(), df[col].std(), decimals=3)

    return row


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Path to runs directory")
    parser.add_argument("--configs",  nargs="+", default=ALL_CONFIGS,
                        help="Which model configs to aggregate")
    args = parser.parse_args()

    all_records = []
    summary_rows = []

    print(f"\nAggregating results from: {args.runs_dir}\n")

    for model_name in args.configs:
        records = load_seed_results(args.runs_dir, model_name)
        print(f"  {model_name}: {len(records)} seeds loaded")

        all_records.extend(records)
        summary_rows.append(summarize_model(records, model_name))

    if not all_records:
        print("\nNo results found. Check that runs_dir is correct and results.json files exist.")
        return

    # Save all-seeds flat table
    flat_df = pd.DataFrame([flatten_record(r) for r in all_records])
    flat_path = os.path.join(args.runs_dir, "aggregate_all_seeds.csv")
    flat_df.to_csv(flat_path, index=False)
    print(f"\nPer-seed table saved to: {flat_path}")

    # Save and print summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.runs_dir, "aggregate_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to:  {summary_path}\n")

    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()