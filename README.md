
# ECG Diploma Project

## 🎯 Project Overview
Multi-label ECG classification using the PTB-XL dataset (21,799 12-lead ECGs, 5 diagnostic superclasses: NORM, MI, STTC, CD, HYP).

**Primary Metric:** Macro AUROC (official PTB-XL benchmark)

## 📁 Project Structure
```text
ecg-diploma/
├── data_preprocessed/       # Preprocessed NPZ arrays (ignored in git)
├── experiments/             # Config files for models (`*.yaml`)
├── notebooks/               # EDA, Demo, and Colab Training Notebooks
├── splits/
│   └── folds.json           # 🔒 LOCKED - Official PTB-XL split
├── src/
│   ├── preprocess.py        # 🔒 LOCKED - SOTA preprocessing
│   ├── data.py              # 🔒 LOCKED - Dataset & augmentations
│   ├── eval.py              # 🔒 LOCKED - Evaluation metrics
│   ├── train.py             # 🔒 LOCKED - Universal Training loop (AMP, EarlyStopping)
│   ├── infer.py             # Inference script
│   ├── augmentations/       # 🔒 LOCKED - Time-series augmentations
│   └── models/
│       ├── encoder_cnn.py       # Base CNN encoder
│       ├── model_a_baseline.py  # Model A: Baseline ResNet
│       ├── model_b_attention.py # Model B: CNN + Attention (Transformer)
│       └── model_c_gnn.py       # Model C: CNN + GNN
└── requirements.txt         # 🔒 LOCKED - Project Dependencies
```

## 📈 Expected Results (Model A Baseline)
- **Test Macro AUROC:** ~0.9075
- **Features implemented:** Label Smoothing (0.1), BCEWithLogitsLoss, WeightedRandomSampler, Automatic Mixed Precision (AMP), SequentialLR (Warmup + CosineAnnealing).

## 🚀 Quick Start (Training via Google Colab)

### 📥 Dataset Setup
Since the data makes up several gigabytes, it isn't included in the GitHub repository. To run this project:
1. Download the preprocessed `ptbxl_sota_100hz_diagnostic_superclass.npz` and `metadata_100hz.json` dataset files. (Ask the project maintainer for the download link).
2. Create a folder named `data_preprocessed/` explicitly in the project root.
3. Place the downloaded `.npz` and `.json` files inside the `data_preprocessed/` folder.

### 🏋️ Colab Training
To efficiently train models using Google Colab GPUs without losing data:
1. Upload this repository to your Google Drive.
2. Open `notebooks/model_a_training.ipynb` via Google Colab.
3. Follow the steps inside the notebook. It will detect that it runs on Colab and mount Google Drive.
4. Run the training cell (it automatically saves checkpoints directly to your Drive workspace).

*If you have a local GPU, simply run:*
```bash
python -m src.train --model model_a_baseline --epochs 50 --batch_size 64 --accum_steps 2 --warmup_epochs 5 --label_smoothing 0.1 --use_weighted_sampler --output_dir results/model_a_baseline
```

## ⚠️ Data Policy
**NEVER commit raw or preprocessed data!**
Data directories are ignored in `.gitignore`. Provide data locally via Google Drive links.

## 🤝 Contributing
Review the locked files before making major structural changes. Do not modify `src/train.py` unless adding a new universally supported training parameter.

