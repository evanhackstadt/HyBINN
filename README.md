# What About the Unmapped Genes? Ablation of a Hybrid Biologically-Informed Neural Network for Breast Cancer Survival from RNA-seq

Personal research project under the mentorship of Dr. Hamed Akbari, Santa Clara University.

## Research Questions

Do genes that aren't mapped to biological pathways affect performance of neural networks for cancer survival analysis?

## Neural Network Architecture

HyBINN has 3 input branches:

1. BINN <—— Genes mapped to pathways in Reactome
2. Gene MLP <—— All other genes (not mapped in Reactome)
3. Clinical MLP <—— clinical phenotype (T/N/M stages)

The model can be run using any combination of these branches, enabling an ablation experiment. Each branch produces per-patient risk scores. If multiple branches are used, their risk scores are combined with learned weights into a final risk score per-patient. Interpretability can be performed on the pathway layer of the BINN branch.

## Repo Layout

* `configs/`: universal model architecture and hyperparameters
* `data/`: datasets were not uploaded due to file size; processed pathway names are included
* `experiments/`: master scripts to run experiments across seeds / model configurations, and process the results
  * `figures/`: output figures from `analyze_results.py`
  * `runs/`: logged outputs from all configuration-seed combos (saved .pt model files are excluded due to their large size)
* `notebooks/`: jupyter notebooks for data preprocessing and legacy data visualization
* `src/`: model source code
  * `datasets/`: defines dataset class and dataloader creation
  * `models/`: PyTorch model classes for each branch
  * `processing/`: mapped/unmapped gene split, pathway mask
  * `training/`: cox loss and main training/testing functions
  * `utils/`: cindex bootstrapping, stratified splitting, logging, loading

## Known Issues

Pathway filtering by gene size (≥15 and ≤300 genes) failed to filter an edge case of duplicate pathways. As a result, 51 of the 491 pathway nodes contain 8–14 genes. If fixed, all models should be rerun.
