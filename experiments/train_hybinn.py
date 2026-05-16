# experiments/train_hybinn.py

import os, shutil
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from utils.config import load_config
from processing.reactome import build_reactome_map, build_mask_matrix
from processing.split_genes import split_genes
from datasets.dataset import SurvivalDataset, get_dataloader
from utils.splitting import stratified_train_test_split, get_stratified_kfold_indices

from models.binn_branch import BINNBranch
from models.gene_branch import GeneBranch
from models.clinical_branch import ClinicalBranch
from models.survival_head import SurvivalHead
from models.standalone_branch import StandaloneBranch
from models.hybinn_model import HyBINN

from training.trainer import train, test
from utils.logging import get_logger



# ---- Helper Functions ----

def instantiate_model(args, cfg, mapped, unmapped, clinical, mask, pathway_labels, embed_dim):
    """Instantiates and returns model object (hybrid or standalone branch) based on CLI args"""
    
    head = SurvivalHead(embed_dim)
    
    if args.model == 'hybrid':
        model = HyBINN(binn_in_nodes=len(mapped),
                       binn_pathway_nodes=len(pathway_labels),
                       binn_hidden_nodes=cfg['model']['binn']['hidden_nodes'],
                       pathway_mask=mask,
                       gene_in_nodes=len(unmapped),
                       gene_hidden_nodes_1=cfg['model']['gene']['hidden_nodes_1'],
                       gene_hidden_nodes_2=cfg['model']['gene']['hidden_nodes_2'],
                       clinical_in_nodes=len(clinical),
                       clinical_hidden_nodes_1=cfg['model']['clinical']['hidden_nodes_1'],
                       clinical_hidden_nodes_2=cfg['model']['clinical']['hidden_nodes_2'],
                       embedding_nodes=embed_dim)
        
    elif args.model == 'binn':
        branch = BINNBranch(in_nodes=len(mapped),
                            pathway_nodes=len(pathway_labels),
                            hidden_nodes=cfg['model']['binn']['hidden_nodes'],
                            embedding_nodes=embed_dim,
                            pathway_mask=mask)
        model = StandaloneBranch(branch, head)
    
    elif args.model == 'gene':
        branch = GeneBranch(in_nodes=len(unmapped),
                            hidden_nodes_1=cfg['model']['gene']['hidden_nodes_1'],
                            hidden_nodes_2=cfg['model']['gene']['hidden_nodes_2'],
                            embedding_nodes=embed_dim)
        model = StandaloneBranch(branch, head)
    
    elif args.model == 'clinical':
        branch = ClinicalBranch(in_nodes=len(clinical),
                                hidden_nodes_1=cfg['model']['clinical']['hidden_nodes_1'],
                                hidden_nodes_2=cfg['model']['clinical']['hidden_nodes_2'],
                                embedding_nodes=embed_dim)
        model = StandaloneBranch(branch, head)
    
    return model



# ---- Main Function ----

def main():
    
    # CLI Arg Handling
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',    choices=['hybrid', 'binn', 'gene', 'clinical'], required=True)
    parser.add_argument('--config',   type=str, default='../configs/model_config.yaml')
    parser.add_argument('--run_dir',  type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    # allow CLI overrides of config values:
    parser.add_argument('--epochs',   type=int,   default=None)
    parser.add_argument('--lr',       type=float, default=None)
    parser.add_argument('--wd',       type=float, default=None)
    parser.add_argument('--seed',     type=int,   default=None)

    args = parser.parse_args()


    # Load config file
    script_dir = os.path.abspath(__file__)
    cfg = load_config("../configs/model_config.yaml", script_dir)

    # CLI overrides
    if args.run_dir is not None:
        cfg['logging']['run_dir'] = os.path.abspath(args.run_dir)
    if args.run_name is not None:
        cfg['logging']['run_name'] = args.run_name
    if args.epochs is not None:
        cfg['training']['epochs'] = args.epochs
    if args.lr is not None:
        cfg['training']['lr'] = args.lr
    if args.wd is not None:
        cfg['training']['wd'] = args.wd
    if args.seed is not None:
        cfg['data']['random_seed'] = args.seed


    # Load dataset
    gene_data_path = cfg['data']['gene_data_path']
    clinical_data_path = cfg['data']['clinical_data_path']
    df = pd.read_csv(gene_data_path, index_col=0)
    clinical_df = pd.read_csv(clinical_data_path, index_col=0)
    clinical = clinical_df.columns.to_list()
    
    assert df.index.to_list() == clinical_df.index.to_list(), \
        f"Gene-Clinical index mismatch. Check preprocessing or re-run inner join."

    times = df['OS.time'].to_numpy(dtype=np.float32)
    events = df['OS'].to_numpy(dtype=np.float32)
    gene_df = df.drop(columns=['OS.time', 'OS'])

    # Map pathways and split genes
    reactome_path = cfg['data']['reactome_path']
    pathway_map = build_reactome_map(reactome_path)
    mapped, unmapped, valid_pathways = split_genes(gene_df, pathway_map)
    mask, gene_labels, pathway_labels = build_mask_matrix(mapped, pathway_map, valid_pathways)

    x_mapped = gene_df[gene_labels].to_numpy(dtype=np.float32)
    x_unmapped = gene_df[unmapped].to_numpy(dtype=np.float32)
    x_clinical = clinical_df.to_numpy(dtype=np.float32)

    # Dimensions check
    assert x_mapped.shape[1] == mask.shape[0], \
        f"Mapped gene mismatch: {x_mapped.shape[1]} vs {mask.shape[0]}"
        
    # Instantiate and split Dataset
    dataset = SurvivalDataset(x_mapped, x_unmapped, x_clinical, times, events)

    trainval_idx, test_idx = stratified_train_test_split(events, cfg)
    test_dataloader = get_dataloader(dataset, test_idx, cfg, shuffle=False)

    trainval_dataset = SurvivalDataset(x_mapped[trainval_idx, :],
                                       x_unmapped[trainval_idx, :],
                                       x_clinical[trainval_idx, :],
                                       times[trainval_idx],
                                       events[trainval_idx])


    # Logging path setup
    dir_path = cfg['logging']['run_dir']
    run_name = cfg['logging']['run_name']

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log_path = os.path.join(dir_path, f"{run_name}.log")
    best_model_path = os.path.join(dir_path, f"best_model_{run_name}.pt")

    if os.path.exists(log_path):
        os.remove(log_path)

    logger = get_logger(__name__, log_path)
    logger.info(f"train_hybinn.py log started {datetime.datetime.now()}")
    logger.info("=========================================")


    # TRAINING LOOP

    embed_dim = cfg['model']['embedding_dim']
    head = SurvivalHead(embed_dim)

    fold_cindices = []
    fold_best_epochs = []
    results = pd.DataFrame(columns=['fold', 'epoch', 'train_loss',
                                    'val_loss', 'cindexes'])

    for fold, (train_idx, val_idx) in enumerate(get_stratified_kfold_indices(trainval_dataset.y_event.numpy(), cfg)):
        
        # get dataloaders
        train_dataloader = get_dataloader(trainval_dataset, train_idx, cfg, shuffle=True)
        val_dataloader = get_dataloader(trainval_dataset, val_idx, cfg, shuffle=False)
        
        # instantiate fresh model
        model = instantiate_model(args, cfg, mapped, unmapped, clinical, mask, pathway_labels, embed_dim)
        
        # Start log
        logger.info("\n=========================================")
        logger.info(f"FOLD {fold+1}/{cfg['training']['folds']}")
        logger.info("=========================================\n")
        logger.info(f"Model: {model.__class__.__name__}")
        
        
        # train and log results
        train_loss, val_loss, cindexes = train(model, train_dataloader, val_dataloader, log_path, cfg)
        
        n_epochs = int(np.argmax(cindexes)) + 1     # the epoch # that yielded the max cindex
        fold_best_epochs.append(n_epochs)
        fold_cindices.append(max(cindexes))         # best cindex across epochs for this fold
        
        fold_results = pd.DataFrame({
            'fold': [fold+1] * len(cindexes),
            'epoch': range(1, len(cindexes)+1),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'cindexes': cindexes
        })
        results = pd.concat([results, fold_results], ignore_index=True)
        
    results.to_csv(os.path.join(dir_path, f"train_results_{run_name}.csv"))

    # Summarize training results
    cv_mean_cindex = np.mean(fold_cindices)    # mean best cindex across folds
    cv_stdev_cindex = np.std(fold_cindices)
    mean_best_epoch = int(np.round(np.mean(fold_best_epochs)))

    logger.info("\n=========================================")
    logger.info(f"CV training complete! Mean ± std C-index: {cv_mean_cindex:.4f} ± {cv_stdev_cindex:.4f}")
    logger.info(f"Best epochs per fold: {fold_best_epochs}")


    # RETRAINING on full train_val dataset
    # for the mean best epochs from training

    trainval_dataloader = get_dataloader(trainval_dataset, np.arange(len(trainval_dataset)),  # all indices
                                    cfg, shuffle=True)

    final_model = instantiate_model(args, cfg, mapped, unmapped, clinical, mask, pathway_labels, embed_dim)

    # Temporarily override epoch count
    retrain_cfg = cfg.copy()
    retrain_cfg['training'] = cfg['training'].copy()
    retrain_cfg['training']['epochs'] = mean_best_epoch

    logger.info("\n=========================================")
    logger.info(f"Retraining model on full train+val for {mean_best_epoch} epochs...")

    train_loss, val_loss, cindexes = train(final_model, trainval_dataloader, trainval_dataloader,  # val_loader not used since early_stopping=False
                                        log_path, retrain_cfg, early_stopping=False)
    retrain_mean_cindex = np.mean(cindexes)
    retrain_stdev_cindex = np.std(cindexes)

    # Save final model manually
    torch.save(final_model.state_dict(), best_model_path)


    # TESTING
    avg_test_loss, test_cindex = test(final_model, test_dataloader, best_model_path, log_path)

    # Summarize final results
    summary_results = pd.DataFrame({
        'instance': ['CV Training', 'Retraining', 'Test'],
        'mean_cindex': [cv_mean_cindex, retrain_mean_cindex, test_cindex],
        'stdev_cindex': [cv_stdev_cindex, retrain_stdev_cindex, 0]
    })
    summary_results.to_csv(os.path.join(dir_path, f"summary_results_{run_name}.csv"))

    # Save config used for this run
    os.makedirs(dir_path, exist_ok=True)
    shutil.copy("../configs/model_config.yaml",
                os.path.join(dir_path, f"config_{run_name}.yaml"))


    # INTERPRETABILITY
    # examine stored pathway layer activations on test set
    if args.model == 'hybrid':
        # TODO, different pathway extraction
        logger.info("Pathways TODO")
    elif args.model == 'binn':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger = get_logger("interpretability", log_path)

        all_activations = []
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                X_mapped = batch['X_mapped'].to(device)
                X_unmapped = batch['X_unmapped'].to(device)
                _ = model(X_mapped, X_unmapped)
                all_activations.append(model.branch.pathway_activations.detach().cpu())

        # Average activation per pathway across all test patients
        mean_activations = torch.cat(all_activations, dim=0).mean(dim=0).numpy()
        top_indices = np.argsort(mean_activations)[::-1][:10]
        top_pathways = [pathway_labels[i] for i in top_indices]

        logger.info("Top 10 activated pathways:")
        for rank, (idx, name) in enumerate(zip(top_indices, top_pathways)):
            logger.info(f"  {rank+1}. {name}  (mean activation: {mean_activations[idx]:.4f})")



if __name__ == "__main__":
    main()