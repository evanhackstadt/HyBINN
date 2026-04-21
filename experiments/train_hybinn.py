# experiments/train_hybinn.py

import os, shutil
import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from utils.config import load_config
from processing.reactome import build_reactome_map, build_mask_matrix
from processing.split_genes import split_genes
from models.binn import StandaloneBINN
from datasets.dataset import SurvivalDataset, get_dataloader
from utils.splitting import stratified_train_test_split, get_stratified_kfold_indices
from training.trainer import train, test
from utils.logging import get_logger


# Load config file
cfg = load_config("../configs/model_config.yaml")

# Load dataset
data_path = cfg['data']['data_path']
df = pd.read_csv(data_path, index_col=0)

times = df['OS.time'].to_numpy(dtype=np.float32)
events = df['OS'].to_numpy(dtype=np.float32)
gene_df = df.drop(columns=['OS.time', 'OS'])

# Map pathways and split genes
reactome_path = cfg['data']['reactome_path']
pathway_map = build_reactome_map(reactome_path)
mapped, unmapped, valid_pathways = split_genes(gene_df, pathway_map)
mask, gene_labels, pathway_labels = build_mask_matrix(mapped, pathway_map, valid_pathways)

x_mapped = gene_df[mapped].to_numpy(dtype=np.float32)
x_unmapped = gene_df[unmapped].to_numpy(dtype=np.float32)

# Dimensions check
assert x_mapped.shape[1] == mask.shape[0], \
    f"Mapped gene mismatch: {x_mapped.shape[1]} vs {mask.shape[0]}"
    
# Instantiate and split Dataset
dataset = SurvivalDataset(x_mapped, x_unmapped, times, events)

trainval_idx, test_idx = stratified_train_test_split(events, cfg)
test_dataloader = get_dataloader(dataset, test_idx, cfg, shuffle=False)

trainval_dataset = SurvivalDataset(x_mapped[trainval_idx, :],
                                   x_unmapped[trainval_idx, :],
                                   times[trainval_idx],
                                   events[trainval_idx])


# Logging path setup
dir_path = cfg['logging']['run_dir']
run_name = cfg['logging']['run_name']

log_path = os.path.join(dir_path, f"{run_name}.log")
results_path = os.path.join(dir_path, f"{run_name}.csv")
best_model_path = os.path.join(dir_path, "best_model.pt")


# TRAINING LOOP

fold_cindexes = []
results = pd.DataFrame(columns=['fold', 'epoch', 'train_loss',
                                'val_loss', 'cindexes'])

for fold, (train_idx, val_idx) in enumerate(get_stratified_kfold_indices(trainval_dataset.y_event, cfg)):
    print(f"--- Fold {fold+1}/{cfg['training']['folds']} ---")
    
    # get dataloaders
    train_dataloader = get_dataloader(trainval_dataset, train_idx, cfg, shuffle=True)
    val_dataloader = get_dataloader(trainval_dataset, val_idx, cfg, shuffle=False)
    
    # instantiate fresh model
    model = StandaloneBINN(in_nodes=len(gene_labels),
                       pathway_nodes=len(pathway_labels),
                       hidden_nodes=cfg['model']['hidden_nodes'],
                       out_nodes=cfg['model']['out_nodes'],
                       pathway_mask=mask)
    
    # train and log results
    train_loss, val_loss, cindexes = train(model, train_dataloader, val_dataloader, cfg)
    
    fold_cindexes.append(np.mean(cindexes))
    fold_results = pd.DataFrame({
        'fold': [fold] * len(cindexes),
        'epoch': range(1, len(cindexes)+1),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'cindexes': cindexes
    })
    results = pd.concat([results, fold_results], ignore_index=True)
    
# Summarize training results
mean_cindex = np.mean(fold_cindexes)
stdev_cindex = np.std(fold_cindexes)
results.to_csv(results_path)

"""
retrain on full trainval set - use mean epochs across folds as stopping criteria
test(model, test_loader, ...)
log final test C-index
"""

# TESTING
avg_test_loss, test_cindex = test(model=model, test_loader=test_dataloader, 
                                  best_model_file=best_model_path, logfile=log_path)


# Save config used for this run
os.makedirs(dir_path, exist_ok=True)
shutil.copy("configs/model_config.yaml", os.path.join(dir_path, "config.yaml"))


# INTERPRETABILITY
# examine stored pathway layer activations on test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger("interpretability", log_path)

all_activations = []
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        X = batch['X_mapped'].to(device)
        _ = model(X)
        all_activations.append(model.pathway_activations.detach().cpu())

# Average activation per pathway across all test patients
mean_activations = torch.cat(all_activations, dim=0).mean(dim=0).numpy()
top_indices = np.argsort(mean_activations)[::-1][:10]
top_pathways = [pathway_labels[i] for i in top_indices]

logger.info("Top 10 activated pathways:")
for rank, (idx, name) in enumerate(zip(top_indices, top_pathways)):
    logger.info(f"  {rank+1}. {name}  (mean activation: {mean_activations[idx]:.4f})")



# ---- Baseline: CoxPH ----
# CURRENTLY NOT IMPLEMENTED CORRECTLY - TODO?
from lifelines import CoxPHFitter

baseline_df = pd.DataFrame({
    'OS.time': df['OS.time'], 
    'OS': df['OS']
})

cph = CoxPHFitter().fit(baseline_df, duration_col='OS.time', event_col='OS')
cph.print_summary()

# TODO:
#   take in CLI args for runs? or automatically increment?