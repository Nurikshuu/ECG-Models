"""
eval_final.py — Единый честный evaluation для всех моделей PTB-XL
=================================================================
Правила:
  1. Порог (threshold) ищется ТОЛЬКО на val set
  2. Test set используется ОДИН раз, только для финальных метрик
  3. Все модели оцениваются одним и тем же кодом
  4. Никакого TTA, никакого threshold search на тесте

Запуск:
    python eval_final.py --data_path data_preprocessed/ptbxl_sota_100hz_diagnostic_superclass.npz
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from typing import Dict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# КОНФИГУРАЦИЯ — каждый участник правит здесь
# ─────────────────────────────────────────────
MODELS_CONFIG = {
    "Model_A_Baseline": {
        "enabled": True,
        "checkpoint": "results/model_a_baseline/best_model.pt",
        "module": "src.models.model_baseline.model_a_baseline",
        "class":  "ModelABaseline",
        "kwargs": {"feature_dim": 256, "hidden_dim": 128, "num_classes": 5, "encoder_dropout": 0.1},
        "strict": True,
        "batch_size": 64,
    },
    "Channel_Attention": {
        "enabled": True,
        "checkpoint": "results/model_b_attention/best_model_b_final.pth",
        "module": "src.models.model_Damir.model_b_attention",
        "class":  "ModelBAttention",
        "kwargs": {"feature_dim": 256, "hidden_dim": 128, "num_classes": 5, "encoder_dropout": 0.1},
        "strict": True,
        "batch_size": 64,
    },
    "LeadWise_GNN": {
        "enabled": True,
        "checkpoint": "results/model_c_gnn/gnn_stage2_best.pth",
        "module": "src.models.model_Dimash.leadwise_gnn",
        "class":  "LeadWiseResNetGNN",
        "kwargs": {"embed_dim": 128, "hidden_dim": 96, "num_classes": 5, "dropout": 0.15},
        "strict": True,
        "batch_size": 64,
    },
    "InceptionTime": {
        "enabled": True,
        "checkpoint": "results/model_inception_baseline/best_model.pt",
        "module": "src.models.model_baseline.ecg_inceptiontime",
        "class":  "InceptionTimeBaseline",
        "kwargs": {
            "num_classes": 5,
            "n_filters": 32,
            "kernel_sizes": [10, 20, 40],
            "bottleneck_channels": 32,
            "n_blocks": 2,
            "depth_per_block": 3,
            "head_dropout": 0.5,
        },
        "strict": True,
        "batch_size": 128,
    },
    "RetNet": {
        "enabled": True,
        "checkpoint": "results/model_Nurik_retnet/best_model.pt",
        "module": "src.models.model_Nurik.ecg_retnet",
        "class":  "RetNetECG",
        "kwargs": {
            "num_classes": 5,
            "d_model": 512,
            "n_blocks": 10,
            "num_heads": 8,
            "ffn_mult": 2,
            "drop_path_rate": 0.1,
            "head_dropout": 0.2,
        },
        "state_key": "ema_state_dict",
        "strict": True,
        "batch_size": 16,
    },
}

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: str) -> Path:
    """Resolve repo-relative config paths even if eval_final.py is run from another cwd."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p

# ─────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ
# ─────────────────────────────────────────────
def load_data(data_path: str):
    """Загружает NPZ с ключами X_val/X_test или signals/labels/splits."""
    data = np.load(data_path, allow_pickle=True)

    # Поддержка разных схем именования
    def get(keys):
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"Не найден ни один из ключей: {keys}")

    if all(k in data for k in ["X_val", "y_val", "X_test", "y_test"]):
        X_val  = get(["X_val",  "x_val"])
        y_val  = get(["y_val",  "Y_val"])
        X_test = get(["X_test", "x_test"])
        y_test = get(["y_test", "Y_test"])
    elif all(k in data for k in ["signals", "labels", "splits"]):
        signals = data["signals"]
        labels  = data["labels"]
        splits  = data["splits"]

        if splits.dtype.kind in {"S", "O", "U"}:
            splits = splits.astype(str)
            val_mask = splits == "val"
            test_mask = splits == "test"
        else:
            # Numeric strat_fold: 1-8 train, 9 val, 10 test
            val_mask = splits == 9
            test_mask = splits == 10

        X_val  = signals[val_mask]
        y_val  = labels[val_mask]
        X_test = signals[test_mask]
        y_test = labels[test_mask]
    else:
        raise KeyError("NPZ должен содержать X_val/X_test или signals/labels/splits")

    X_val  = np.asarray(X_val, dtype=np.float32)
    y_val  = np.asarray(y_val, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    print(f"Val:  {X_val.shape}, Test: {X_test.shape}")
    return X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def get_probabilities(model: nn.Module, X: np.ndarray,
                      device: str, batch_size: int = 64) -> np.ndarray:
    """Прогоняет данные через модель, возвращает sigmoid-вероятности."""
    model.eval()
    all_probs = []

    X = np.asarray(X, dtype=np.float32)

    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i+batch_size]).to(device)
        logits = model(batch)
        # Если модель возвращает tuple (logits, attention) — берём только logits
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ─────────────────────────────────────────────
# ЧЕСТНЫЙ THRESHOLD SEARCH — только на val!
# ─────────────────────────────────────────────
def find_optimal_threshold_on_val(probs_val: np.ndarray,
                                   y_val: np.ndarray) -> float:
    """
    Ищет один глобальный порог, максимизирующий macro-F1 на val set.
    НЕ использует тест-данные.
    """
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.02):
        preds = (probs_val >= thr).astype(int)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return float(best_thr)


# ─────────────────────────────────────────────
# МЕТРИКИ
# ─────────────────────────────────────────────
def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_metrics(probs: np.ndarray, targets: np.ndarray,
                    threshold: float) -> Dict:
    """
    Считает все метрики с фиксированным порогом.
    Порог должен быть найден на val, не на тесте.
    """
    preds = (probs >= threshold).astype(int)

    try:
        macro_auc = float(roc_auc_score(targets, probs, average="macro"))
    except ValueError:
        macro_auc = float("nan")

    try:
        macro_auprc = float(average_precision_score(targets, probs, average="macro"))
    except ValueError:
        macro_auprc = float("nan")

    metrics = {
        "macro_AUROC":  round(macro_auc, 4),
        "macro_AUPRC":  round(macro_auprc, 4),
        "macro_F1":     round(f1_score(targets, preds, average="macro", zero_division=0), 4),
        "threshold":    round(threshold, 2),
    }

    # Per-class AUROC
    for i, cls in enumerate(CLASS_NAMES):
        metrics[f"AUROC_{cls}"] = round(_safe_roc_auc(targets[:, i], probs[:, i]), 4)

    return metrics


# ─────────────────────────────────────────────
# ЗАГРУЗКА МОДЕЛИ (динамический import)
# ─────────────────────────────────────────────
def load_model_from_config(cfg: Dict, device: str) -> nn.Module:
    import importlib
    import inspect

    ckpt_path = resolve_project_path(cfg["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    module = importlib.import_module(cfg["module"])
    factory = getattr(module, cfg["class"])
    if inspect.isclass(factory):
        model = factory(**cfg["kwargs"])
    else:
        model = factory(**cfg["kwargs"])

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict):
        preferred_key = cfg.get("state_key")
        if preferred_key and preferred_key in state:
            state = state[preferred_key]
        else:
            for key in ("model_state_dict", "state_dict", "model_state", "model", "ema_state_dict"):
                if key in state:
                    state = state[key]
                    break
    # DataParallel / torch.compile checkpoints often add wrappers to state names.
    if isinstance(state, dict):
        for prefix in ("module.", "_orig_mod."):
            if state and all(isinstance(k, str) and k.startswith(prefix) for k in state.keys()):
                state = {k[len(prefix):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=cfg.get("strict", True))
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="data_preprocessed/ptbxl_sota_100hz_diagnostic_superclass.npz",
        help="Path to NPZ with splits or X_val/X_test keys",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Загружаем данные ОДИН РАЗ
    data_path = resolve_project_path(args.data_path)
    X_val, y_val, X_test, y_test = load_data(str(data_path))

    results = {}

    for model_name, cfg in MODELS_CONFIG.items():
        if not cfg["enabled"]:
            print(f"\n[SKIP] {model_name} — disabled in config")
            continue

        print(f"\n{'='*55}")
        batch_size = int(cfg.get("batch_size", args.batch_size))
        print(f"  Evaluating: {model_name}  (batch_size={batch_size})")
        print(f"{'='*55}")

        try:
            model = load_model_from_config(cfg, device)
        except Exception as e:
            print(f"  [ERROR] Не удалось загрузить модель: {e}")
            continue

        # Шаг 1: Вероятности на val (для поиска порога)
        print("  → Inference on val set...")
        probs_val = get_probabilities(model, X_val, device, batch_size)

        # Шаг 2: Находим порог ТОЛЬКО на val
        threshold = find_optimal_threshold_on_val(probs_val, y_val)
        print(f"  → Optimal threshold (from val): {threshold:.2f}")

        # Шаг 3: Вероятности на test (ОДИН РАЗ)
        print("  → Inference on test set...")
        probs_test = get_probabilities(model, X_test, device, batch_size)

        # Шаг 4: Финальные метрики с val-порогом
        metrics = compute_metrics(probs_test, y_test, threshold)
        results[model_name] = metrics

        print(f"  macro-AUROC : {metrics['macro_AUROC']}")
        print(f"  macro-AUPRC : {metrics['macro_AUPRC']}")
        print(f"  macro-F1    : {metrics['macro_F1']}  (thr={metrics['threshold']})")
        print(f"  Per-class AUROC:")
        for cls in CLASS_NAMES:
            print(f"    {cls:6s}: {metrics[f'AUROC_{cls}']}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # ─── Итоговая таблица ───
    if results:
        print(f"\n{'='*55}")
        print("  FINAL COMPARISON TABLE")
        print(f"{'='*55}")
        header = f"  {'Model':<22} {'AUROC':>7} {'AUPRC':>7} {'F1':>7} {'Thr':>5}"
        print(header)
        print("  " + "-" * 50)
        for name, m in results.items():
            print(f"  {name:<22} {m['macro_AUROC']:>7.4f} "
                  f"{m['macro_AUPRC']:>7.4f} {m['macro_F1']:>7.4f} "
                  f"{m['threshold']:>5.2f}")

        # Сохраняем в CSV
        import csv, os
        csv_path = PROJECT_ROOT / "results" / "eval_final.csv"
        os.makedirs(csv_path.parent, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            cols = ["model", "macro_AUROC", "macro_AUPRC", "macro_F1", "threshold"] + \
                   [f"AUROC_{c}" for c in CLASS_NAMES]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for name, m in results.items():
                row = {"model": name, **m}
                w.writerow(row)
        print(f"\n  Saved → {csv_path}")


if __name__ == "__main__":
    main()
