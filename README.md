
# ECG Diploma Project (PTB-XL ECG Classification)

## Overview
Multi-label ECG classification on the PTB-XL dataset (21,799 ECGs, 12 leads, 10 seconds at 100 Hz).
Targets are 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP.
Primary metric: macro AUROC (PTB-XL benchmark).

## Data
Expected preprocessed files:
- data_preprocessed/ptbxl_sota_100hz_diagnostic_superclass.npz
- data_preprocessed/metadata_100hz.json (optional for analysis)

The NPZ already contains train/val/test splits. These splits are derived from PTB-XL strat_fold (1-8 train, 9 val, 10 test) in the preprocessing pipeline.

## Training entrypoints (notebooks)
All training is notebook-based in this repo.

- Model A baseline (CNN encoder + mean pooling): notebooks/notebook_baseline/model_a_training.ipynb
- InceptionTime baseline: notebooks/notebook_baseline/train_inception_pro (1).ipynb
- ResNet1d Wang baseline: notebooks/notebook_Dimash/01_baseline_resnet1d_wang_colab.ipynb
- Model B channel attention: notebooks/notebook_Damir/model_b_training.ipynb
- LeadWise GNN (Dimash): notebooks/notebook_Dimash/02_leadwise_gnn_model.ipynb
- RetNet v4 (Nurik): notebooks/notebook_Nurik/ecg_retnet_v4.ipynb

Each notebook controls its own output path (checkpoints, logs, metrics). The repository already contains example artifacts in results/.

## Evaluation
- Shared evaluator: src/eval.py (PTBXLEvaluator). For fair comparison, use the same evaluator and the same split.
- Some notebooks define local evaluation helpers; align them to PTBXLEvaluator if you need strict parity.

## Repository structure
```text
ecg-diploma/
├── data_preprocessed/     # Preprocessed NPZ arrays (ignored in git)
├── experiments/           # Config files for experiments (*.yaml)
├── notebooks/             # Training notebooks
│   ├── notebook_baseline/
│   ├── notebook_Damir/
│   ├── notebook_Dimash/
│   └── notebook_Nurik/
├── results/               # Checkpoints and logs
├── splits/                # Official PTB-XL split
├── src/
│   ├── augmentations/
│   ├── data.py            # Dataset and loaders
│   ├── eval.py            # Metrics and evaluation
│   ├── preprocess.py      # Preprocessing pipeline
│   └── models/
│       ├── model_baseline/
│       │   ├── encoder_cnn.py
│       │   └── model_a_baseline.py
│       ├── model_Damir/
│       │   └── model_b_attention.py
│       └── model_Dimash/
│           ├── basic_conv1d.py
│           └── resnet1d.py
├── step11_calibration.py
├── results_summary.csv
└── requirements.txt
```

## Setup (local)
1. Create a virtual environment and activate it.
2. Install dependencies: pip install -r requirements.txt
3. Place the NPZ file in data_preprocessed/.
4. Open the notebook for the model you want to train and run all cells.

## Results
results/ contains checkpoints and logs. When a notebook saves test metrics, it usually writes a test_results.json or test_metrics.json file.
results_summary.csv is the place to aggregate final test metrics across models.

## Data policy
Do not commit raw or preprocessed data to git. Provide data locally via Google Drive or local disk paths.

