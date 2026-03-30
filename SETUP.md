
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
The project uses a unified training pipeline inside `src/train.py`.

### To run the Baseline Model (Model A) locally:
```bash
python -m src.train --model model_a_baseline --epochs 50 --batch_size 64
```

### To develop new models:
1. Create your model definition in `src/models/` (e.g., `model_b_attention.py`).
2. Add it to the factory function inside `src/train.py` (in `get_model()` function).
3. Use the unified `train.py` script to seamlessly train and log your new model!

If you encounter path issues from Jupyter, run `%cd /path/to/ecg-diploma` to ensure `src` is in your working directory.

