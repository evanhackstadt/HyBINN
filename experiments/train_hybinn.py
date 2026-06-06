# experiments/train_hybinn.py

import os, argparse, yaml, json, datetime
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from utils.seed   import set_seed
from utils.bootstrap import bootstrap_cindex
from utils.config import load_config, write_config
from utils.json_utils import NumpyEncoder
from utils.splitting import stratified_train_test_split, get_stratified_kfold_indices

from processing.reactome import build_reactome_map, build_mask_matrix
from processing.split_genes import split_genes
from datasets.dataset import SurvivalDataset, get_dataloader

from models.binn_branch import BINNBranch
from models.gene_branch import GeneBranch
from models.clinical_branch import ClinicalBranch
from models.hybinn_model import HyBINN

from training.trainer import train, test
from utils.logging import get_logger



# ---- Helper Function ----

def instantiate_model(cfg, mapped, unmapped, clinical, mask, pathway_labels, embed_dim):
    """Instantiates and returns model object based on the active branches"""
    
    active_branches = cfg['model']['branches']
    
    branch_map = {}
    for branch_name in active_branches:
        if branch_name == 'binn':
            branch_map['binn'] = BINNBranch(in_nodes=len(mapped),
                                            pathway_nodes=len(pathway_labels),
                                            hidden_nodes=cfg['model']['binn']['hidden_nodes'],
                                            embedding_nodes=embed_dim,
                                            pathway_mask=mask)
        elif branch_name == 'gene':
            branch_map['gene'] = GeneBranch(in_nodes=len(unmapped),
                                            hidden_nodes_1=cfg['model']['gene']['hidden_nodes_1'],
                                            hidden_nodes_2=cfg['model']['gene']['hidden_nodes_2'],
                                            embedding_nodes=embed_dim)
        elif branch_name == 'clinical':
            branch_map['clinical'] = ClinicalBranch(in_nodes=len(clinical),
                                                    hidden_nodes_1=cfg['model']['clinical']['hidden_nodes_1'],
                                                    hidden_nodes_2=cfg['model']['clinical']['hidden_nodes_2'],
                                                    embedding_nodes=embed_dim)
        else:
            raise ValueError(f"Unknown branch name: {branch_name}")

    model = HyBINN(branch_map, 
                   embedding_nodes=cfg['model']['embedding_dim'])
    
    return model



# ---- Main Function ----

def main(branches, config, run_dir, run_name, 
         emb_dim, epochs, lr, wd, seed):

    # Load config file
    script_dir = os.path.abspath(__file__)
    cfg = load_config(config, script_dir)

    # CLI overrides
    if len(branches) > 0:
        cfg['model']['branches'] = branches
    if run_dir is not None:
        cfg['logging']['run_dir'] = os.path.abspath(run_dir)
    if run_name is not None:
        cfg['logging']['run_name'] = run_name
    if emb_dim is not None:
        cfg['model']['embedding_dim'] = emb_dim
    if epochs is not None:
        cfg['training']['epochs'] = epochs
    if lr is not None:
        cfg['training']['lr'] = lr
    if wd is not None:
        cfg['training']['wd'] = wd
    if seed is not None:
        cfg['data']['random_seed'] = seed

    # Set the random seed before anything dataloaders, splitting, etc.
    set_seed(seed)

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
    ids = df.index.values
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
    
    # Store indeces for reproducibility
    ids_df = pd.DataFrame(columns=['Partition', 'Fold', 'IDs'])
    ids_df.loc[len(ids_df)] = ['Testing', 'N/A', ids[test_idx]]

    # Logging path setup
    dir_path = cfg['logging']['run_dir']
    run_name = cfg['logging']['run_name']

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log_path = os.path.join(dir_path, f"{run_name}.log")
    best_model_path = os.path.join(dir_path, f"best_model.pt")

    if os.path.exists(log_path):
        os.remove(log_path)

    # Start log
    logger = get_logger(__name__, log_path)
    logger.info(f"train_hybinn.py log started {datetime.datetime.now()}")
    logger.info("=========================================")
    logger.info(f"Model Active Branches: {cfg['model']['branches']}")
    logger.info("=========================================")


    # TRAINING LOOP

    embed_dim = cfg['model']['embedding_dim']

    fold_cindices = []
    fold_best_epochs = []
    train_results_by_epoch = pd.DataFrame(columns=['fold', 'epoch', 'train_loss', 
                                                   'val_loss', 'cindex'])

    for fold, (train_idx, val_idx) in enumerate(get_stratified_kfold_indices(trainval_dataset.y_event.numpy(), cfg)):
        
        # log IDs
        ids_df.loc[len(ids_df)] = ['Training', fold+1, ids[train_idx]]
        ids_df.loc[len(ids_df)] = ['Validation', fold+1, ids[val_idx]]
        
        # get dataloaders
        train_dataloader = get_dataloader(trainval_dataset, train_idx, cfg, shuffle=True)
        val_dataloader = get_dataloader(trainval_dataset, val_idx, cfg, shuffle=False)
        
        # instantiate fresh model
        model = instantiate_model(cfg, mapped, unmapped, clinical, mask, pathway_labels, embed_dim)
        
        # Start log
        logger.info("\n=========================================")
        logger.info(f"FOLD {fold+1}/{cfg['training']['folds']}")
        logger.info("=========================================\n")
        
        
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
            'cindex': cindexes
        })
        train_results_by_epoch = pd.concat([train_results_by_epoch, fold_results], ignore_index=True)
    
    # Save training results
    # train_results_by_epoch.to_csv(os.path.join(dir_path, f"train_train_results_by_epoch.csv"))
    
    # Log indices for reproducibility
    ids_df.to_csv(os.path.join(dir_path, f"traintest_patient_ids.csv"))
    
    # Summarize training results across folds
    cv_mean_cindex = np.mean(fold_cindices)    # mean best cindex across folds
    cv_stdev_cindex = np.std(fold_cindices)
    mean_best_epoch = int(np.round(np.mean(fold_best_epochs)))
    
    # Build main results dict
    results = {
        "model":             cfg['model']['branches'],
        "seed":              cfg['data']['random_seed'],
        "n_trainval":        len(trainval_idx),
        "n_test":            len(test_idx),
        "cv_mean_cindex":    float(cv_mean_cindex),
        "cv_std_cindex":     float(cv_stdev_cindex),
        "mean_best_epoch":   int(mean_best_epoch),
        "fold_cindices":     [float(c) for c in fold_cindices],
        "fold_best_epochs":  fold_best_epochs,
    }

    logger.info("\n=========================================")
    logger.info(f"CV training complete! Mean ± std C-index: {cv_mean_cindex:.4f} ± {cv_stdev_cindex:.4f}")
    logger.info(f"Best epochs per fold: {fold_best_epochs}")


    # RETRAINING on full train_val dataset
    # for the mean best epochs from training

    trainval_dataloader = get_dataloader(trainval_dataset, np.arange(len(trainval_dataset)),  # all indices
                                    cfg, shuffle=True)

    final_model = instantiate_model(cfg, mapped, unmapped, clinical, mask, pathway_labels, embed_dim)

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
    
    # Add retraining to results
    results["retrain_mean_cindex"] = float(retrain_mean_cindex)
    results["retrain_std_cindex"]  = float(retrain_stdev_cindex)

    # Save final model manually
    torch.save(final_model.state_dict(), best_model_path)


    # TESTING
    
    avg_test_loss, test_cindex, test_outputs = test(
        final_model, test_dataloader, best_model_path, log_path
    )

    # Bootstrap CI on test predictions
    boot = bootstrap_cindex(
        times=test_outputs['times'],
        events=test_outputs['events'],
        risk_scores=test_outputs['risk_final'],
        n_bootstrap=1000,
        seed=cfg['data']['random_seed']
    )
    logger.info(f"Bootstrap 95% CI: [{boot['lower']:.4f}, {boot['upper']:.4f}]")

    results["test_cindex"]   = float(test_cindex)
    results["test_loss"]     = float(avg_test_loss)
    results["bootstrap_ci"]  = boot
    
    # Save branch weights (if using late fusion)
    if hasattr(final_model, 'fusion_logits'):
        import torch.nn.functional as F
        weights = F.softmax(final_model.fusion_logits.detach().cpu(), dim=0).numpy()
        branch_names = list(cfg['model']['branches'])
        results["branch_weights"] = {
            name: float(w) for name, w in zip(branch_names, weights)
        }
    
    # Save patient-level test predictions
    preds_df = pd.DataFrame({
        'patient_id':    ids[test_idx],
        'risk_final':    test_outputs['risk_final'],
        **{f'risk_{k}': v for k, v in test_outputs.items()
           if k.startswith('risk_') and k != 'risk_final'},
        'time':          test_outputs['times'],
        'event':         test_outputs['events'],
    })
    preds_df.to_csv(os.path.join(dir_path, "predictions_test.csv"), index=False)
    
    
    # INTERPRETABILITY
    
    # Learned Fusion Weights
    active_branches = cfg['model']['branches']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("interpretability", log_path)

    # Biological Pathways
    if 'binn' in active_branches:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger = get_logger("interpretability", log_path)

        all_activations = []
        final_model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                X_mapped = batch['X_mapped'].to(device)
                X_unmapped = batch['X_unmapped'].to(device)
                X_clinical = batch['X_clinical'].to(device)
                # run forward to populate branch internals
                _ = final_model(X_mapped, X_unmapped, X_clinical)

                if 'binn' not in final_model.branches:
                    continue
                binn_branch = final_model.branches['binn']

                # pathway_activations is (batch, n_pathways); collect along batch dim
                act = binn_branch.pathway_activations.detach().cpu()
                all_activations.append(act)

        if len(all_activations) == 0:
            logger.info("No pathway activations found for BINN branch.")
        else:
            activ_tensor = torch.cat(all_activations, dim=0).numpy()

            # Average activation per pathway across all test patients
            mean_activations = activ_tensor.mean(axis=0)
            top_indices = np.argsort(mean_activations)[::-1][:10]
            top_pathways = [pathway_labels[i] for i in top_indices]
            top_activations = [mean_activations[i] for i in top_indices]
            
            pathways_dict = {
                name: float(a) for name, a in zip(top_pathways, top_activations)
            }
            
            logger.info("Top 10 activated pathways:")
            logger.info(pathways_dict)
            
            results["top_pathways"] = pathways_dict
        
                
    # SAVE FILES
    
    # Write all scalar results to a single JSON
    with open(os.path.join(dir_path, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Results saved to {dir_path}/results.json")
    
    # Save config used for this run
    write_config(cfg, os.path.join(dir_path, f"config_frozen.yaml"))
    



# ---- CLI Arg Handling ----

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--branches', nargs='+', default=[])
    parser.add_argument('--config',   type=str,  default='../configs/model_config.yaml')
    parser.add_argument('--run_dir',  type=str,  default=None)
    parser.add_argument('--run_name', type=str,  default=None)

    # allow CLI overrides of config values:
    parser.add_argument('--emb_dim',  type=int,   default=None)
    parser.add_argument('--epochs',   type=int,   default=None)
    parser.add_argument('--lr',       type=float, default=None)
    parser.add_argument('--wd',       type=float, default=None)
    parser.add_argument('--seed',     type=int,   default=None)

    args = parser.parse_args()
    
    main(args.branches, args.config, args.run_dir, args.run_name,
         args.emb_dim, args.epochs, args.lr, args.wd, args.seed)

if __name__ == "__main__":
    parse_args()