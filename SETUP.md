
# 🚀 Setup Guide for ECG Diploma Project

## Prerequisites
- Python 3.10+
- CUDA 12.x (for local GPU training, optional)
- ~2 GB disk space for preprocessed data

## Step 1: Clone Repository
```bash
git clone https://github.com/dimash-dot/ecg-diploma.git
cd ecg-diploma
```

## Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Get Data
Place the preprocessed files directly into the `data_preprocessed/` directory:
```
data_preprocessed/
├── ptbxl_sota_100hz_diagnostic_superclass.npz
└── metadata_100hz.json
```

## Step 5: Start Development / Training
Training in this repository is notebook-based. Pick the notebook for your model and run all cells.

Recommended entrypoints:
- Model A baseline: notebooks/notebook_baseline/model_a_training.ipynb
- InceptionTime baseline: notebooks/notebook_baseline/train_inception_pro (1).ipynb
- ResNet1d Wang baseline: notebooks/notebook_Dimash/01_baseline_resnet1d_wang_colab.ipynb
- Model B attention: notebooks/notebook_Damir/model_b_training.ipynb
- LeadWise GNN: notebooks/notebook_Dimash/02_leadwise_gnn_model.ipynb
- RetNet: notebooks/notebook_Nurik/ecg_retnet_v4.ipynb

If you encounter path issues from Jupyter, run `%cd /path/to/ecg-diploma` to ensure `src` is in your working directory.

