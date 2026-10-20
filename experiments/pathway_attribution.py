# experiments/pathway_attribution.py
"""
pathway_attribution.py

Computes pathway-level feature importance for a trained BINN branch using
Integrated Gradients (Sundararajan, Taly & Yan, 2017), evaluated POST-HOC
on a frozen, already-trained HyBINN checkpoint. No retraining required.

--------------------------------------------------------------------------
Why not mean activation, why not raw partial derivatives?
--------------------------------------------------------------------------
Mean pathway activation (post-tanh) measures how "on" a pathway node is on
average -- not how much that pathway's signal actually moves the model's
output. Raw partial derivatives (the approach used by CoxPAS-Net, Hao et
al. 2018) fix that "on vs. relevant" conflation, but suffer from gradient
saturation in tanh-heavy architectures like this one: a saturated unit can
be highly influential yet show ~0 *local* gradient at the observed point.

Integrated Gradients accumulates the gradient along the entire straight-line
path from a reference baseline to the observed activation, rather than
evaluating it once at the endpoint. This is immune to endpoint saturation,
and satisfies a completeness axiom (attributions sum exactly to
f(x) - f(baseline)), giving a built-in correctness check via
`convergence_delta`.

This mirrors P-NET (Elmarakeby et al., Nature 2021) -- the closest prior
BINN to this one, also Reactome-pathway-structured -- which used DeepLIFT
for the identical reason (avoiding saturation/discontinuity artifacts in
raw-gradient attribution). IG and DeepLIFT are both baseline-relative
attribution methods; IG is used here because captum ships it as a drop-in
wrapper requiring no custom backward rules.

--------------------------------------------------------------------------
Why this is NOT captum.attr.LayerIntegratedGradients
--------------------------------------------------------------------------
BINNBranch reuses a single nn.Tanh() instance (self.tanh) at three
different points in its forward pass (pathway layer, hidden layer, output
layer). captum's LayerIntegratedGradients hooks a *module instance* and
cannot disambiguate which of the three invocations per forward call you
want attributions for. To avoid this, the frozen model is manually split
into two callables:

    (a) x_mapped              -> pathway_activations   (sc1 + mask + tanh)
    (b) pathway_activations   -> model output           (rest of BINN,
                                                          fusion, survival head)

and vanilla captum.attr.IntegratedGradients is applied to (b), with the
other branches' outputs held fixed as additional forward arguments (they
don't depend on pathway_activations, so holding them fixed is exact, not
an approximation). This is mathematically identical to attributing to an
intermediate layer, without depending on captum's module-hook machinery.

--------------------------------------------------------------------------
Baseline choice
--------------------------------------------------------------------------
The baseline is the all-zero pathway-activation vector -- i.e. "no pathway
signal reaches the rest of the network." This is the standard reference
point for tanh-space attribution (mirrors DeepLIFT's typical all-zero
reference in P-NET).

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    pip install captum --break-system-packages   # if not already installed

    python pathway_attribution.py \
        --run_dir experiments/runs/full_hybinn/seed_0 \
        --out_dir experiments/figures

    python pathway_attribution.py \
        --runs_dir experiments/runs --model_name full_hybinn --both

Requires that <run_dir> contains best_model.pt and config_frozen.yaml,
i.e. it must be a run directory produced by train_hybinn.py.
"""

import os
import argparse
import json

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from utils.seed import set_seed
from utils.config import load_config
from utils.splitting import stratified_train_test_split
from utils.logging import get_logger

from processing.reactome import build_reactome_map, build_mask_matrix
from processing.split_genes import split_genes
from datasets.dataset import SurvivalDataset, get_dataloader

from train_hybinn import instantiate_model


# ---- Reactome ID -> human-readable name decoding (mirrors analyze_results.py) ----

def build_reactome_name_map(reactome_path):
    """
    Reads the Ensembl2Reactome TSV and returns {ReactomePathwayID: PathwayName}.
    """
    df = pd.read_csv(reactome_path, delimiter="\t",
                     names=["EnsemblID", "ReactomePathwayID", "URL",
                            "PathwayName", "Evidence", "Species"])
    df = df[df["Species"] == "Homo sapiens"]
    return df.drop_duplicates("ReactomePathwayID").set_index("ReactomePathwayID")["PathwayName"].to_dict()


# ---- Model surgery: split the BINN branch around its pathway layer ----

def _make_pathway_fns(binn_branch):
    """
    Args:
        binn_branch (BINNBranch): the frozen, trained BINN branch (model.branches['binn'])

    Returns:
        get_pathway_activations(x_mapped) -> pathway_activations
            Reproduces BINNBranch's sc1 + mask + tanh step exactly.
        binn_tail(pathway_activations) -> binn branch output
            Reproduces everything in BINNBranch AFTER the pathway layer.
            Safe to call in eval mode (dropout becomes a no-op).
    """

    def get_pathway_activations(x_mapped):
        masked_weights = binn_branch.sc1.weight * binn_branch.mask.T
        return binn_branch.tanh(F.linear(x_mapped, masked_weights, binn_branch.sc1.bias))

    def binn_tail(pathway_activations):
        x = binn_branch.dropout(pathway_activations)
        x = binn_branch.tanh(binn_branch.fc2(x))
        x = binn_branch.dropout(x)
        x = binn_branch.tanh(binn_branch.fc3(x))
        return x

    return get_pathway_activations, binn_tail


# ---- Main attribution routine ----

def compute_pathway_attributions(model, dataloader, device, n_steps=50, logger=None):
    """
    Runs Integrated Gradients on the BINN branch's pathway layer across an
    entire dataloader (typically the frozen run's own held-out test set).

    Args:
        model (HyBINN): frozen, trained model, already .eval()'d with weights loaded
        dataloader (DataLoader): should be built from the SAME split used to
                                  train/evaluate this checkpoint (see main())
        device (torch.device)
        n_steps (int): number of interpolation steps along the IG path.
                        Increase if convergence_delta is large relative to
                        the risk score's scale.

    Returns:
        mean_abs_attr (ndarray): shape (n_pathways,) -- mean |IG attribution|
                                  per pathway across all patients in dataloader
        all_attrs (ndarray): shape (n_patients, n_pathways) -- per-patient
                              signed attributions, kept for downstream analysis
                              (e.g. do high-risk vs low-risk patients rely on
                              different pathways?)
        convergence_deltas (list of ndarray): per-batch completeness-axiom
                              sanity check; should be small vs. the model's
                              output scale
    """
    model.eval()
    binn_branch = model.branches['binn']
    full_branch_order = list(model.branches.keys())
    non_binn_names = [n for n in full_branch_order if n != 'binn']

    get_pathway_acts, binn_tail = _make_pathway_fns(binn_branch)

    def pathway_to_output(pathway_activations, *other_tensors):
        """
        Recomputes the model's final output as a function of pathway_activations
        only, with every other branch's output supplied as a fixed extra arg.
        captum auto-expands tensor additional_forward_args to match the
        n_steps-interpolated batch, so this composes correctly with IG.
        """
        binn_out = binn_tail(pathway_activations)
        other_dict = dict(zip(non_binn_names, other_tensors))
        outs = [binn_out if name == 'binn' else other_dict[name] for name in full_branch_order]

        if model.emb_dim == 1:
            weights = torch.softmax(model.fusion_logits, dim=0)
            r_final = 0
            for w, r in zip(weights, outs):
                r_final = r_final + w * r
            return r_final
        else:
            z = model.attention_fusion(outs)
            return model.survival_head(z)

    ig = IntegratedGradients(pathway_to_output)

    all_attrs, convergence_deltas = [], []

    for batch in dataloader:
        x_mapped = batch['X_mapped'].to(device)
        x_unmapped = batch['X_unmapped'].to(device)
        x_clinical = batch['X_clinical'].to(device)

        # Other branches don't depend on pathway_activations -> compute once, hold fixed.
        other_tensors = tuple(
            model.branches[name](x_mapped, x_unmapped, x_clinical).detach()
            for name in non_binn_names
        )

        pathway_acts = get_pathway_acts(x_mapped).detach().requires_grad_(True)
        baseline = torch.zeros_like(pathway_acts)

        attributions, delta = ig.attribute(
            inputs=pathway_acts,
            baselines=baseline,
            additional_forward_args=other_tensors,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        all_attrs.append(attributions.detach().cpu().numpy())
        convergence_deltas.append(delta.detach().cpu().numpy())

        if logger:
            logger.info(f"Batch done. mean |convergence delta| = {np.abs(delta.detach().cpu().numpy()).mean():.6f}")

    all_attrs = np.concatenate(all_attrs, axis=0)     # (n_patients, n_pathways)
    mean_abs_attr = np.abs(all_attrs).mean(axis=0)    # (n_pathways,)

    # Sanity guard: a convergence delta of ~0 is good news (IG converged), but if
    # attributions themselves are ALSO exactly 0 everywhere, that's a red flag for
    # a disconnected graph (e.g. pathway_acts not actually reaching model output)
    # rather than genuine convergence. Distinguish the two explicitly.
    if np.allclose(all_attrs, 0.0):
        if logger:
            logger.warning(
                "WARNING: all IG attributions are exactly 0. A convergence delta "
                "of 0 alongside all-zero attributions suggests a disconnected "
                "computation graph, not genuine convergence -- double check that "
                "pathway_activations actually reaches the model output."
            )

    return mean_abs_attr, all_attrs, convergence_deltas


# ---- Single-run core logic (reused by both single-run and multi-seed modes) ----

def run_single_seed(run_dir, config_path=None, reactome_path_override=None,
                    n_steps=50, out_dir=None, save_outputs=True, logger=None, untrained=False):
    """
    Loads one frozen run (a single (model_config, seed) checkpoint), reproduces
    its exact preprocessing + test split, and computes pathway-level Integrated
    Gradients attributions against it.

    Args:
        run_dir (str): frozen run directory (contains best_model.pt, config_frozen.yaml)
        config_path (str): override path to config_frozen.yaml (default: <run_dir>/config_frozen.yaml)
        reactome_path_override (str): override the reactome path stored in the config,
                                       if the file has moved since training
        n_steps (int): IG interpolation steps
        out_dir (str): where to save this seed's outputs (default: run_dir)
        save_outputs (bool): if True, writes pathway_ig_rankings.csv and the
                              per-patient attribution .npy to out_dir
        logger: shared logger, or None to create a fresh one
        untrained (bool): optionally instantiate a fresh model (rather than a pretrained seed)
                          to obtain a baseline for pathway IG stability

    Returns:
        pathway_labels (list[str]): Reactome pathway IDs, in matrix-column order
        mean_abs_attr (ndarray): shape (n_pathways,)
        all_attrs (ndarray): shape (n_patients, n_pathways)
        deltas (list[ndarray]): per-batch IG convergence deltas
        reactome_path (str): the reactome file actually used (for name decoding upstream)
    """
    logger = logger or get_logger(__name__)
    config_path = config_path or os.path.join(run_dir, "config_frozen.yaml")
    out_dir = out_dir or run_dir
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(config_path, os.path.abspath(__file__))
    if reactome_path_override:
        cfg['data']['reactome_path'] = reactome_path_override
    reactome_path = cfg['data']['reactome_path']

    seed = cfg['data']['random_seed']
    set_seed(seed)   # reproduces the EXACT train/val/test split used to train this checkpoint

    logger.info(f"Loading frozen run from {run_dir}")
    logger.info(f"Active branches: {cfg['model']['branches']}")

    if 'binn' not in cfg['model']['branches']:
        raise ValueError(f"{run_dir} has no BINN branch -- nothing to attribute pathways for.")

    # ---- Reproduce preprocessing (deterministic given the same input files) ----
    df = pd.read_csv(cfg['data']['gene_data_path'], index_col=0)
    clinical_df = pd.read_csv(cfg['data']['clinical_data_path'], index_col=0)
    times = df['OS.time'].to_numpy(dtype=np.float32)
    events = df['OS'].to_numpy(dtype=np.float32)
    gene_df = df.drop(columns=['OS.time', 'OS'])

    pathway_map = build_reactome_map(reactome_path)
    mapped, unmapped, valid_pathways = split_genes(gene_df, pathway_map)
    mask, gene_labels, pathway_labels = build_mask_matrix(mapped, pathway_map, valid_pathways)
    logger.info(
        f"Reactome mapping: {len(mapped)} mapped genes, "
        f"{len(unmapped)} unmapped genes, {len(valid_pathways)} pathways"
    )

    x_mapped = gene_df[gene_labels].to_numpy(dtype=np.float32)
    x_unmapped = gene_df[unmapped].to_numpy(dtype=np.float32)
    x_clinical = clinical_df.to_numpy(dtype=np.float32)

    dataset = SurvivalDataset(x_mapped, x_unmapped, x_clinical, times, events)

    # Reproduce the exact test split used at training time (same seed, same StratifiedShuffleSplit)
    _, test_idx = stratified_train_test_split(events, cfg)
    test_dataloader = get_dataloader(dataset, test_idx, cfg, shuffle=False)

    # ---- Rebuild architecture and load frozen weights ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = cfg['model']['embedding_dim']
    
    if untrained:
        set_seed(seed + 1000)    # fresh init; the test split was already drawn from cfg seed
        logger.info(f"Set seed to {seed + 1000} for a fresh untrained model init. Will NOT be loading best_model.pt.")
    
    model = instantiate_model(cfg, mapped, unmapped, clinical_df.columns.to_list(),
                              mask, pathway_labels, embed_dim)
    if not untrained:
        best_model_path = os.path.join(run_dir, "best_model.pt")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    model.to(device)
    model.eval()

    # ---- Run Integrated Gradients ----
    logger.info(f"Running Integrated Gradients (n_steps={n_steps}) on {len(test_idx)} test patients...")
    mean_abs_attr, all_attrs, deltas = compute_pathway_attributions(
        model, test_dataloader, device, n_steps=n_steps, logger=logger
    )

    max_delta = max(np.abs(d).max() for d in deltas)
    logger.info(f"Max |convergence delta| across all batches: {max_delta:.6f}")

    if save_outputs:
        name_map = build_reactome_name_map(reactome_path)
        ranking = pd.DataFrame({
            "pathway_id": pathway_labels,
            "pathway_name": [name_map.get(pid, pid) for pid in pathway_labels],
            "mean_abs_ig_attribution": mean_abs_attr,
        }).sort_values("mean_abs_ig_attribution", ascending=False)

        out_path = os.path.join(out_dir, "pathway_ig_rankings.csv")
        ranking.to_csv(out_path, index=False)
        logger.info(f"Saved pathway IG rankings to {out_path}")

        # Per-patient signed attributions, for follow-up analysis (e.g. do high-risk vs.
        # low-risk patients rely on different pathways? subgroup-specific attribution?)
        np.save(os.path.join(out_dir, "pathway_ig_attributions_per_patient.npy"), all_attrs)

    return pathway_labels, mean_abs_attr, all_attrs, deltas, reactome_path



# ---- Stability analysis (replaces aggregate_over_seeds) ----

def load_pathway_sizes(run_dir):
    """Genes per pathway node (mask column sums). Depends only on data + Reactome, not on seed."""
    cfg = load_config(os.path.join(run_dir, "config_frozen.yaml"), os.path.abspath(__file__))
    df = pd.read_csv(cfg['data']['gene_data_path'], index_col=0)
    gene_df = df.drop(columns=['OS.time', 'OS'])
    pathway_map = build_reactome_map(cfg['data']['reactome_path'])
    mapped, unmapped, valid_pathways = split_genes(gene_df, pathway_map)
    mask, _, pathway_labels = build_mask_matrix(mapped, pathway_map, valid_pathways)
    sizes = pd.Series(mask.sum(axis=0), index=pathway_labels)
    return sizes


def ensure_seed_outputs(runs_dir, model_name, seeds, n_steps, reactome_path_override,
                        untrained, logger):
    """Runs IG only for seeds whose per-seed rankings CSV doesn't already exist."""
    for seed in seeds:
        run_dir = os.path.join(runs_dir, model_name, f"seed_{seed}")
        if not os.path.isdir(run_dir):
            logger.info(f"[MISSING] {run_dir}")
            continue
        out = os.path.join(run_dir, "untrained") if untrained else run_dir
        if os.path.exists(os.path.join(out, "pathway_ig_rankings.csv")):
            continue
        run_single_seed(run_dir=run_dir, reactome_path_override=reactome_path_override,
                        n_steps=n_steps, out_dir=out, untrained=untrained, logger=logger)


def _load_stability_rankings(runs_dir, model_name, seeds, subdir=""):
    """Load per-seed rankings as a pathways-by-seeds matrix."""
    cols, names = {}, {}
    for seed in seeds:
        path = os.path.join(
            runs_dir, model_name, f"seed_{seed}", subdir, "pathway_ig_rankings.csv"
        )
        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            continue
        rankings = pd.read_csv(path).set_index("pathway_id")
        cols[seed] = rankings["mean_abs_ig_attribution"]
        names.update(rankings["pathway_name"].to_dict())

    return pd.DataFrame(cols).dropna(), names


def _pairwise_spearman_rhos(matrix):
    """Calculate pairwise Spearman correlations between seed columns."""
    return np.array([
        stats.spearmanr(matrix[first], matrix[second])[0]
        for first, second in combinations(matrix.columns, 2)
    ])


def summarize_stability(runs_dir, model_name, seeds, out_dir=None, top_k=20,
                        subdir="", tag=""):
    """
    Cross-seed stability of pathway attributions. No filtering, no ranking. 
    top_k is fixed in advance, not tuned.
    subdir="untrained" reads the untrained-network control instead.
    """
    out_dir = out_dir or os.path.join(runs_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    M, names = _load_stability_rankings(runs_dir, model_name, seeds, subdir)
    n_path, n_seeds = M.shape
    print(f"{n_path} pathways x {n_seeds} seeds")

    # 1. Pairwise Spearman between seeds (full vectors)
    rho = _pairwise_spearman_rhos(M)

    # 2. Top-k selection counts vs. chance
    in_topk = pd.DataFrame({s: M[s].rank(ascending=False) <= top_k for s in M.columns})
    n_in_topk = in_topk.sum(axis=1)
    min_freq = int(np.ceil(n_seeds / 2))
    observed = int((n_in_topk >= min_freq).sum())
    expected = n_path * stats.binom.sf(min_freq - 1, n_seeds, top_k / n_path)

    # 3. Size confound
    seed0_dir = os.path.join(runs_dir, model_name, f"seed_{M.columns[0]}")
    sizes = load_pathway_sizes(seed0_dir).reindex(M.index)
    rho_size = np.array([stats.spearmanr(M[s], sizes)[0] for s in M.columns])

    summary = {
        "n_pathways": n_path, "n_seeds": n_seeds, "top_k": top_k,
        "pairwise_rho_mean": float(rho.mean()), "pairwise_rho_min": float(rho.min()),
        "pairwise_rho_max": float(rho.max()), "n_pairs": len(rho),
        "chance_rho_sd": float(1 / np.sqrt(n_path - 1)),
        f"n_pathways_in_topk_of_>= {min_freq}_seeds": observed,
        "n_expected_by_chance": float(expected),
        "rho_attribution_vs_size_mean": float(rho_size.mean()),
        "rho_attribution_vs_size_range": [float(rho_size.min()), float(rho_size.max())],
    }
    with open(os.path.join(out_dir, f"stability_summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Per-pathway table (descriptive; Table 3 = head(10))
    tbl = pd.DataFrame({
        "pathway_id": M.index,
        "pathway_name": [names[p] for p in M.index],
        "n_genes": sizes.values,
        "mean_abs_ig": M.mean(axis=1).values,
        "sd_abs_ig": M.std(axis=1, ddof=1).values,
        f"n_seeds_in_top{top_k}": n_in_topk.values,
    }).sort_values("mean_abs_ig", ascending=False)
    tbl["cv"] = tbl["sd_abs_ig"] / tbl["mean_abs_ig"]
    tbl.insert(0, "rank_by_mean", np.arange(1, n_path + 1))
    tbl.to_csv(os.path.join(out_dir, f"pathway_stability{tag}.csv"), index=False)

    # Two-panel figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(rho, bins=15, color="steelblue", edgecolor="black")
    ax[0].axvline(0, ls="--", c="gray")
    ax[0].set_xlabel("Spearman ρ between seeds (all pathways)")
    ax[0].set_ylabel(f"Seed pairs (n={len(rho)})")
    ax[1].scatter(sizes, M.mean(axis=1), s=8, alpha=0.5, color="steelblue")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Genes in pathway node"); ax[1].set_ylabel("Mean |IG| across seeds")
    ax[1].set_title(f"ρ = {rho_size.mean():.2f}", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"pathway_stability{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_stability_comparison(runs_dir, model_name, seeds, out_dir=None):
    """Save comparison figures for trained and untrained attribution stability."""
    out_dir = out_dir or os.path.join(runs_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    scatter_data = []
    for subdir, label, color in (
        ("", "Trained", "steelblue"),
        ("untrained", "Untrained", "darkorange"),
    ):
        matrix, _ = _load_stability_rankings(runs_dir, model_name, seeds, subdir)
        rho = _pairwise_spearman_rhos(matrix)
        ax.hist(rho, alpha=0.55, color=color, edgecolor="black",
                label=f"{label} (n={len(rho)} seed pairs)")

        seed0_dir = os.path.join(
            runs_dir, model_name, f"seed_{matrix.columns[0]}"
        )
        sizes = load_pathway_sizes(seed0_dir).reindex(matrix.index)
        rho_size = np.array([
            stats.spearmanr(matrix[seed], sizes)[0] for seed in matrix.columns
        ])
        scatter_data.append((label, color, sizes, matrix.mean(axis=1), rho_size.mean()))

    ax.axvline(0, ls="--", c="gray")
    ax.set_xlabel("Spearman ρ between seeds (all pathways)")
    ax.set_ylabel("Seed pairs")
    # ax.set_title("Trained vs. untrained pathway attribution stability")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "pathway_stability_comparison.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=False)
    for axis, (label, color, sizes, mean_attribution, rho_size_mean) in zip(
        ax, scatter_data
    ):
        axis.scatter(sizes, mean_attribution, s=8, alpha=0.5, color=color)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel("Mean |IG| across seeds")
        axis.set_title(f"{label}: ρ = {rho_size_mean:.2f}", fontsize=10)

    ax[-1].set_xlabel("Genes in pathway node")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "pathway_stability_size_comparison.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser()

    # Single-run mode (not recommended)
    parser.add_argument('--run_dir', type=str, default=None,
                        help="Single frozen run directory (contains best_model.pt, config_frozen.yaml)")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config_frozen.yaml (default: <run_dir>/config_frozen.yaml)")

    # Multi-seed mode (use for both trained and untrained IG + summary)
    parser.add_argument('--runs_dir', type=str, default=None,
                        help="Base runs directory, e.g. experiments/runs (for summary)")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Model config subfolder under --runs_dir, e.g. full_hybinn")
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)),
                        help="Seeds to include in aggregation mode (default: 0-9)")
    attribution_mode = parser.add_mutually_exclusive_group()
    attribution_mode.add_argument('--untrained', action='store_true',
                                  help="Control: IG on freshly initialized (untrained) networks")
    attribution_mode.add_argument('--both', action='store_true',
                                  help="Compute trained and untrained stability and overlay their rho histograms")

    # Shared options
    parser.add_argument('--reactome_path', type=str, default=None,
                        help="Override reactome path from config, if it moved since training")
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Where to save outputs (default: run_dir, or <runs_dir>/<model_name> in aggregate mode)")
    args = parser.parse_args()

    logger = get_logger(__name__)

    if args.runs_dir and args.model_name:
        if args.both:
            for untrained in (False, True):
                ensure_seed_outputs(
                    args.runs_dir, args.model_name, args.seeds,
                    args.n_steps, args.reactome_path, untrained, logger
                )
                summarize_stability(
                    args.runs_dir, args.model_name, args.seeds, args.out_dir,
                    subdir="untrained" if untrained else "",
                    tag="_untrained" if untrained else "",
                )
            plot_stability_comparison(
                args.runs_dir, args.model_name, args.seeds, args.out_dir
            )
        else:
            ensure_seed_outputs(args.runs_dir, args.model_name, args.seeds,
                                args.n_steps, args.reactome_path, args.untrained, logger)
            summarize_stability(args.runs_dir, args.model_name, args.seeds, args.out_dir,
                                subdir="untrained" if args.untrained else "",
                                tag="_untrained" if args.untrained else "")
    elif args.run_dir:
        if args.both:
            parser.error("--both requires --runs_dir and --model_name")
        run_single_seed(
            run_dir=args.run_dir,
            config_path=args.config,
            reactome_path_override=args.reactome_path,
            n_steps=args.n_steps,
            out_dir=args.out_dir,
            save_outputs=True,
            logger=logger,
            untrained=args.untrained,
        )
    else:
        parser.error("Provide either --run_dir (single seed) or --runs_dir + --model_name (summarize seeds).")


if __name__ == "__main__":
    main()