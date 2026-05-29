# HyBINN: A Hybrid Biologically-Informed Neural Network for Cancer Survival Prediction

Personal research project. Manuscript forthcoming.

### Neural Network Architecture

The network contains 3 input branches
1. Sparse BINN <—— Genes mapped to pathways in Reactome
2. Gene MLP <—— All other genes (not mapped in Reactome)
3. Clinical MLP <—— clinical phenotype (T/N/M staging)

The model can be run using any combination of these branches. Each branch produces per-patient risk scores. If multiple branches are used, their risk scores are combined with learned weights into a final risk score per-patient.

### Repo Layout (outdated):
```
hybinn/
│
├── data/
│   ├── raw/ — raw breast cancer data downloaded from UCSC Xena
│   ├── processed/ — data saved during preprocessing, including the main data.csv used for training
│   └── reactome/ — raw data downloaded from Reactome
│
├── src/
│   ├── datasets/dataset.py — define Dataset class and get_dataloaders() function
│   │
│   ├── models/
│   │   └── binn.py — defines the PyTorch model class
│   │
│   ├── training/
│   │   ├── trainer.py — contains functions to train, validate, and test the model
│   │   └── loss.py — custom survival loss function that wraps CoxPH loss from pycox
│   │
│   ├── preprocessing/
│   │   ├── gene_split.py — filters and splits genes into those mapped to pathways by Reactome, and those not
│   │   └── reactome_processing.py — maps genes to pathways and builds mask matrix for the sparse layer
│   │
│   ├── utils/
│   │   └── logging.py — function to return basic logger object
│
├── experiments/
│   ├── runs/ — contains logs and results from different training runs
│   └── train_hybinn.py — primary script that creates, trains, tests, and logs a model
│
├── notebooks/
│   ├── preprocess_data.ipynb — one-time data pipeline that turns raw UCSC Xena data into data.csv
│   └── plot_losses.ipynb - notebook to manually create figures / graphs from the results
│
└── README.md
```

