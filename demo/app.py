#!/usr/bin/env python3
"""
ECG Classifier Demo — Flask backend
====================================
Run from the project root:

    python demo/app.py

Then open http://localhost:5000 in your browser.

Requirements (in addition to project dependencies):
    pip install flask
"""

import io
import sys
from pathlib import Path

# ── Add project root to sys.path so "src.*" imports work ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS   = ["I", "II", "III", "aVR", "aVL", "aVF",
           "V1", "V2", "V3", "V4",  "V5",  "V6"]

# Registry: name → (model_instance, metadata_dict)
MODELS: dict[str, torch.nn.Module] = {}
MODEL_INFO: dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _unwrap_state(raw) -> dict:
    """Pull the actual state-dict out of various checkpoint formats."""
    if not isinstance(raw, dict):
        return raw
    for key in ("model_state_dict", "ema_state_dict", "state_dict", "model_state", "ema", "model"):
        if key in raw:
            return raw[key]
    return raw


def _load_ckpt(path: Path):
    return _unwrap_state(
        torch.load(path, map_location=DEVICE, weights_only=False)
    )


def load_models():
    # ── 1. RetNet-ECG v4 ──────────────────────────────────────────────────
    try:
        from src.models.model_Nurik.ecg_retnet import RetNetECG
        m = RetNetECG(
            num_classes=5, d_model=512, n_blocks=10,
            num_heads=8, drop_path_rate=0.1
        )
        ckpt = PROJECT_ROOT / "results" / "model_Nurik_retnet" / "best_model.pt"
        m.load_state_dict(_load_ckpt(ckpt), strict=True)
        m.eval().to(DEVICE)
        MODELS["RetNet"] = m
        MODEL_INFO["RetNet"] = {
            "auroc":       0.9133,
            "params":      "33.5 M",
            "description": (
                "Retention-based sequence model with multi-scale patch embedding "
                "(k=5/11/25), xPos encoding, DropPath, and cross-lead Transformer."
            ),
            "color": "#818cf8",
        }
        print("\u2705  RetNet loaded")
    except Exception as exc:
        print(f"\u26a0\ufe0f   RetNet: {exc}")

    # ── 2. SE Attention ──────────────────────────────────────────────────
    try:
        from src.models.model_Damir.model_b_attention import ModelBAttention
        m = ModelBAttention(feature_dim=256, hidden_dim=128, num_classes=5)
        ckpt = PROJECT_ROOT / "results" / "model_b_attention" / "best_model_b_final.pth"
        m.load_state_dict(_load_ckpt(ckpt), strict=True)
        m.eval().to(DEVICE)
        MODELS["SE Attention"] = m
        MODEL_INFO["SE Attention"] = {
            "auroc":       0.9064,
            "params":      "~5 M",
            "description": (
                "ResNet1d CNN encoder with SE-blocks, cross-lead attention pooling, "
                "and generalized super-pooling (attention + max + mean)."
            ),
            "color": "#fbbf24",
        }
        print("\u2705  SE Attention loaded")
    except Exception as exc:
        print(f"\u26a0\ufe0f   SE Attention: {exc}")

    # ── 4. InceptionTime Pro ──────────────────────────────────────────────
    try:
        from src.models.model_baseline.ecg_inceptiontime import InceptionTimeBaseline
        m = InceptionTimeBaseline(num_classes=5)
        ckpt = PROJECT_ROOT / "results" / "model_inception_baseline" / "best_model.pt"
        m.load_state_dict(_load_ckpt(ckpt), strict=False)
        m.eval().to(DEVICE)
        MODELS["InceptionTime"] = m
        MODEL_INFO["InceptionTime"] = {
            "auroc":       0.9067,
            "params":      "~3 M",
            "description": (
                "InceptionTime с расширенным расписанием обучения, cosine LR decay, "
                "label smoothing и weighted sampling."
            ),
            "color": "#06b6d4",
        }
        print("✅  InceptionTime loaded")
    except Exception as exc:
        print(f"⚠️   InceptionTime: {exc}")

    # ── 6. LeadWise GNN ───────────────────────────────────────────────────
    try:
        from src.models.model_Dimash.leadwise_gnn import LeadWiseResNetGNN
        m = LeadWiseResNetGNN(embed_dim=128, hidden_dim=96, num_classes=5, dropout=0.15)
        ckpt = PROJECT_ROOT / "results" / "model_c_gnn" / "gnn_stage2_best.pth"
        m.load_state_dict(_load_ckpt(ckpt), strict=True)
        m.eval().to(DEVICE)
        MODELS["LeadWise GNN"] = m
        MODEL_INFO["LeadWise GNN"] = {
            "auroc":       0.912,
            "params":      "~2 M",
            "description": (
                "Lead-wise ResNet1d encoder + clinical adjacency graph (GAT). "
                "Each lead is a graph node; cross-lead edges encode ECG anatomy."
            ),
            "color": "#f43f5e",
        }
        print("✅  LeadWise GNN loaded")
    except Exception as exc:
        print(f"⚠️   LeadWise GNN: {exc}")

    # ── 6. ResNet1d Wang (baseline) ───────────────────────────────────────
    try:
        from src.models.model_Dimash.leadwise_gnn import ResNet1dWang
        m = ResNet1dWang(num_classes=5, input_channels=12)
        ckpt = PROJECT_ROOT / "results" / "model_c_gnn" / "resnet1d_wang_baseline.pth"
        m.load_state_dict(_load_ckpt(ckpt), strict=True)
        m.eval().to(DEVICE)
        MODELS["ResNet1d"] = m
        MODEL_INFO["ResNet1d"] = {
            "auroc":       0.905,
            "params":      "~1 M",
            "description": (
                "Classic ResNet1d-Wang baseline: 3 residual blocks, "
                "concat-pool head. Reference architecture from PTB-XL benchmarking."
            ),
            "color": "#a3a3a3",
        }
        print("✅  ResNet1d loaded")
    except Exception as exc:
        print(f"⚠️   ResNet1d: {exc}")


# ── Load at startup ────────────────────────────────────────────────────────
load_models()


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def normalize_ecg(signal: np.ndarray) -> np.ndarray:
    """Z-score normalise each lead independently (matches training pipeline)."""
    mean = signal.mean(axis=-1, keepdims=True)
    std  = signal.std(axis=-1, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (signal - mean) / std


def prepare_signal(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Load, validate, and reshape ECG data to float32 [12, 1000]."""
    fname = filename.lower()

    if fname.endswith(".npy"):
        data = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    elif fname.endswith(".csv"):
        import pandas as pd
        data = pd.read_csv(io.BytesIO(raw_bytes), header=None).values.astype(np.float32)
    else:
        raise ValueError(f'Unsupported format ".{fname.rsplit(".", 1)[-1]}". Use .npy or .csv')

    if data.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {list(data.shape)}")

    # Auto-transpose if shape is [1000, 12]
    if data.shape[0] != 12 and data.shape[1] == 12:
        data = data.T

    if data.shape[0] != 12:
        raise ValueError(
            f"Expected 12 leads. Got shape {list(data.shape)}. "
            "Provide an array of shape [12, 1000] or [1000, 12]."
        )

    # Enforce exactly 1000 samples
    T = 1000
    if data.shape[1] > T:
        data = data[:, :T]
    elif data.shape[1] < T:
        data = np.pad(data, ((0, 0), (0, T - data.shape[1])), mode="edge")

    return data.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(Path(__file__).parent), "index.html")


@app.route("/models")
def models_endpoint():
    """Return metadata for loaded models."""
    return jsonify(MODEL_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    fname = file.filename or "upload.npy"

    try:
        raw = file.read()
        data = prepare_signal(raw, fname)

        # Normalise
        signal_norm = normalize_ecg(data)

        # Inference
        x = torch.from_numpy(signal_norm).float().unsqueeze(0).to(DEVICE)
        predictions: dict[str, dict] = {}

        with torch.no_grad():
            for name, model in MODELS.items():
                try:
                    out = model(x)
                    # Some models return (logits, attn_weights)
                    if isinstance(out, tuple):
                        out = out[0]
                    probs = torch.sigmoid(out).squeeze(0).cpu().tolist()
                    predictions[name] = dict(zip(CLASSES, probs))
                except Exception as exc:
                    predictions[name] = {"error": str(exc)}

        # Downsample signal for JSON transfer (every 2nd sample → 50 Hz, 500 pts)
        # Replace NaN/Inf so they don't break JSON serialisation in the browser
        signal_safe = np.nan_to_num(signal_norm[:, ::2], nan=0.0, posinf=0.0, neginf=0.0)
        signal_down = signal_safe.tolist()

        return jsonify({
            "signal":      signal_down,
            "predictions": predictions,
            "fs":          50,        # after downsampling
            "duration":    10,
            "leads":       LEADS,
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Internal error: {exc}"}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("─" * 52)
    print("  ECG Classifier Demo")
    print(f"  URL    : http://localhost:5000")
    print(f"  Device : {DEVICE}")
    loaded = list(MODELS.keys())
    print(f"  Models : {loaded if loaded else '(none — check checkpoint paths)'}")
    print("─" * 52)
    print()
    app.run(debug=False, host="0.0.0.0", port=5000)
