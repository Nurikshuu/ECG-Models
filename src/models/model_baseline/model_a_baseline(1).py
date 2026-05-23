#!/usr/bin/env python3
"""
Model A Baseline — standalone training script (без src.train).
Сохраняет best checkpoint в заданную OUTPUT_DIR.

Запуск из корня проекта:
    python "src/models/model_baseline/model_a_baseline(1).py"
или с кастомными параметрами:
    python "src/models/model_baseline/model_a_baseline(1).py" --epochs 50 --batch_size 64 --output_dir "results/model_a_baseline"
"""

import os, sys, json, time, random, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_PATH = PROJECT_ROOT / "data_preprocessed" / "ptbxl_sota_100hz_diagnostic_superclass.npz"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "model_a_baseline"

# ──────────────────────────────────────────────
# Аргументы командной строки
# ──────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",       type=str,   default=str(DEFAULT_DATA_PATH),
                        help="Путь к актуальному .npz датасету проекта")
    parser.add_argument("--data_dir",        type=str,   default=str(DEFAULT_DATA_DIR),
                        help="Fallback-папка с .npy файлами (signals_train.npy, labels_train.npy, ...)")
    parser.add_argument("--output_dir",      type=str,
                        default=str(DEFAULT_OUTPUT_DIR),
                        help="Куда сохранять checkpoints и логи")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--accum_steps",     type=int,   default=2,
                        help="Gradient accumulation (effective batch = batch_size * accum_steps)")
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience",        type=int,   default=10)
    parser.add_argument("--num_workers",     type=int,   default=2)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--use_weighted_sampler", action="store_true", default=True)
    parser.add_argument("--feat_dim",        type=int,   default=256)
    parser.add_argument("--hidden_dim",      type=int,   default=128)
    parser.add_argument("--num_classes",     type=int,   default=5)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--allow_synthetic_debug", action="store_true",
                        help="Разрешить синтетические данные, если .npz/.npy не найдены")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Воспроизводимость
# ──────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")


# ──────────────────────────────────────────────
# Архитектура (из encoder_cnn.py + model_a_baseline.py)
# ──────────────────────────────────────────────
class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = self.se(x).unsqueeze(-1)
        return x * gate


class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1    = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                                  stride=stride, padding=3, bias=False)
        self.bn1      = nn.BatchNorm1d(out_channels)
        self.drop     = nn.Dropout(dropout)
        self.conv2    = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                                  stride=1, padding=3, bias=False)
        self.bn2      = nn.BatchNorm1d(out_channels)
        self.se       = SEBlock1d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual, inplace=True)


class CNNEncoder(nn.Module):
    STAGES = [
        (1,   32,  1),   # Stem
        (32,  64,  2),   # Block 1: 1000->500
        (64,  128, 2),   # Block 2: 500->250
        (128, 256, 2),   # Block 3: 250->125
        (256, 256, 2),   # Block 4: 125->63
    ]

    def __init__(self, feat_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        assert feat_dim == 256
        layers = [ResidualBlock1d(ic, oc, s, dropout) for ic, oc, s in self.STAGES]
        self.backbone  = nn.Sequential(*layers)
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.feat_dim  = feat_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)
        x = self.backbone(x)
        x = self.gap(x).squeeze(-1)
        x = x.reshape(B, L, self.feat_dim)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=256, hidden=128, num_cls=5,
                 drop1=0.3, drop2=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop1),
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(drop2),
            nn.Linear(hidden, num_cls, bias=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ModelABaseline(nn.Module):
    def __init__(self, feat_dim=256, hidden_dim=128,
                 num_classes=5, encoder_dropout=0.1):
        super().__init__()
        self.encoder = CNNEncoder(feat_dim=feat_dim, dropout=encoder_dropout)
        self.head    = ClassifierHead(in_dim=feat_dim, hidden=hidden_dim,
                                      num_cls=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        embeddings = self.encoder(x)
        pooled     = embeddings.mean(dim=1)
        logits     = self.head(pooled)
        return logits


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class ECGDataset(Dataset):
    """
    Ожидает файлы:
        {data_dir}/signals_{split}.npy  -- float32, shape (N, 12, 1000)
        {data_dir}/labels_{split}.npy   -- float32, shape (N, 5)  multi-hot
    Если файлов нет -- генерирует синтетику для отладки.
    """
    CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

    def __init__(
        self,
        data_dir: str,
        split: str,
        augment: bool = False,
        allow_synthetic: bool = False,
    ):
        sig_path = Path(data_dir) / f"signals_{split}.npy"
        lbl_path = Path(data_dir) / f"labels_{split}.npy"

        if sig_path.exists() and lbl_path.exists():
            self.signals = np.load(sig_path, mmap_mode="r")
            self.labels  = np.load(lbl_path)
            print(f"  Loaded {split} split: {len(self.signals)} samples")
        else:
            if not allow_synthetic:
                raise FileNotFoundError(
                    f"Dataset files not found: {sig_path} and {lbl_path}. "
                    f"Use --data_path for the project .npz dataset or pass "
                    f"--allow_synthetic_debug for a dry debug run."
                )
            print(f"  [WARN] {sig_path} not found -- using SYNTHETIC data for debug!")
            n = {"train": 200, "val": 40, "test": 40}.get(split, 40)
            self.signals = np.random.randn(n, 12, 1000).astype(np.float32)
            self.labels  = (np.random.rand(n, 5) > 0.7).astype(np.float32)

        self.augment = augment
        print(f"    Signal shape: {self.signals.shape[1:]}")
        print(f"    Classes: {', '.join(self.CLASS_NAMES)}")
        self._print_dist()

    def _print_dist(self):
        print("    Label distribution:")
        n = len(self.labels)
        for i, c in enumerate(self.CLASS_NAMES):
            cnt = int(self.labels[:, i].sum())
            print(f"      {c}: {cnt}  ({100*cnt/n:.1f}%)")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = torch.tensor(self.signals[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.float32)
        if self.augment:
            x = x * (0.9 + 0.2 * torch.rand(1))
            x = x + 0.01 * torch.randn_like(x)
        return x, y


# ──────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────
def compute_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probs, average="macro"))
    except Exception:
        return 0.0


def compute_fmax(labels: np.ndarray, probs: np.ndarray, steps: int = 100) -> tuple:
    best_f, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, steps):
        preds = (probs >= thr).astype(float)
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f  = 2 * p * r / (p + r + 1e-8)
        if f > best_f:
            best_f, best_thr = f, thr
    return float(best_f), float(best_thr)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def print_test_report(results: dict, class_names, title: str = "TEST - MODEL_A_BASELINE"):
    per_class = results.get("per_class", {})
    thresholds = results.get("per_class_thresholds", {})
    n_classes = len(class_names)
    valid_classes = int(results.get("macro_auc_valid_classes", n_classes))

    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)

    print("\n Core Metrics (Official PTB-XL)")
    print(f"   Macro AUROC:  {results.get('macro_auc', 0.0):.4f} ({valid_classes}/{n_classes} classes)")
    print(f"   Micro AUROC:  {results.get('micro_auc', 0.0):.4f}")
    print(f"   Macro AUPRC:  {results.get('macro_auprc', 0.0):.4f}")
    print(
        f"   Fmax:         {results.get('fmax', 0.0):.4f} "
        f"(optimal threshold: {results.get('optimal_threshold', 0.5):.3f})"
    )
    print(
        f"   F1 Macro:     {results.get('f1_macro', 0.0):.4f} "
        f"(@ threshold={results.get('threshold_used', 0.50):.2f})"
    )
    print(f"   F1 Micro:     {results.get('f1_micro', 0.0):.4f}")

    print("\n Medical Metrics")
    print(f"   Sensitivity (Recall): {results.get('sensitivity', 0.0):.4f}")
    print(f"   Specificity:          {results.get('specificity', 0.0):.4f}")

    print("\n Per-Class Metrics")
    print(f"{'Class':<10} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'Sens.':<8} {'Spec.':<8} {'Opt.Thr':<8} {'Support':<8}")
    print("-" * 74)
    for cls_name in class_names:
        m = per_class.get(cls_name, {})
        thr = thresholds.get(cls_name, None)
        opt_thr = float(thr) if thr is not None else 0.5
        print(
            f"{cls_name:<10} "
            f"{m.get('auroc', 0.0):.4f}   "
            f"{m.get('auprc', 0.0):.4f}   "
            f"{m.get('f1', 0.0):.4f}   "
            f"{m.get('sensitivity', 0.0):.4f}   "
            f"{m.get('specificity', 0.0):.4f}   "
            f"{opt_thr:.3f}    "
            f"{int(m.get('support', 0)):<8}"
        )
    print("=" * 70)


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────
class LabelSmoothBCE(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing  = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


# ──────────────────────────────────────────────
# Scheduler: Linear Warmup + CosineAnnealing
# ──────────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
    warmup_epochs = max(1, warmup_epochs)
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0,
    )
    if total_epochs <= warmup_epochs:
        return warmup
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ──────────────────────────────────────────────
# Одна эпоха обучения
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
            loss = criterion(model(x), y) / accum_steps
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
    return total_loss / len(loader)


# ──────────────────────────────────────────────
# Валидация / тест
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auroc  = compute_auroc(labels, probs)
    fmax, opt_thr = compute_fmax(labels, probs)
    preds = (probs >= 0.5).astype(float)
    tp = (preds * labels).sum()
    fp = (preds * (1 - labels)).sum()
    fn = ((1 - preds) * labels).sum()
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return {
        "loss":    total_loss / len(loader),
        "auroc":   auroc,
        "fmax":    fmax,
        "opt_thr": opt_thr,
        "f1":      float(f1),
        "probs":   probs,
        "labels":  labels,
    }


# ──────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────
def main():
    args = get_args()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  PTB-XL SOTA Training -- MODEL_A_BASELINE")
    print("="*60)
    print(f"  Device:       {device}  |  AMP: {'ON' if use_amp else 'OFF'}")
    print(f"  Batch size:   {args.batch_size} x {args.accum_steps} accum = {args.batch_size*args.accum_steps} effective")
    print(f"  LR schedule:  Warmup {args.warmup_epochs}ep -> CosineAnnealing")
    print(f"  Label smooth: {args.label_smoothing}")
    print(f"  Epochs:       {args.epochs}  |  Patience: {args.patience}")
    print(f"  Output:       {out_dir}\n")

    data_path = Path(args.data_path)
    if data_path.exists():
        from src.data import get_class_weights_from_dataloader, get_dataloaders

        print(f"  Data path:    {data_path}")
        train_loader, val_loader, test_loader = get_dataloaders(
            data_path=str(data_path),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_weighted_sampler=args.use_weighted_sampler,
            augment_train=True,
            seed=args.seed,
        )
        pos_weight = get_class_weights_from_dataloader(train_loader).to(device)
        class_names = list(getattr(train_loader.dataset, "class_names", ECGDataset.CLASS_NAMES))
    else:
        print(f"  [WARN] .npz data_path not found: {data_path}")
        print(f"  Falling back to .npy data_dir: {args.data_dir}")

        train_ds = ECGDataset(
            args.data_dir, "train", augment=True,
            allow_synthetic=args.allow_synthetic_debug,
        )
        val_ds = ECGDataset(
            args.data_dir, "val", augment=False,
            allow_synthetic=args.allow_synthetic_debug,
        )
        test_ds = ECGDataset(
            args.data_dir, "test", augment=False,
            allow_synthetic=args.allow_synthetic_debug,
        )

        pos_freq = np.clip(train_ds.labels.mean(axis=0).astype(np.float32), 0.01, 0.99)
        pos_weight = torch.tensor((1 - pos_freq) / pos_freq, dtype=torch.float32).to(device)
        class_names = ECGDataset.CLASS_NAMES

        if args.use_weighted_sampler:
            label_counts = np.maximum(train_ds.labels.sum(axis=0), 1.0)
            sample_weights = np.array([
                1.0 / label_counts[np.where(lbl > 0)[0]].mean()
                if len(np.where(lbl > 0)[0]) else 1.0
                for lbl in train_ds.labels
            ])
            sampler = WeightedRandomSampler(
                torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(train_ds), replacement=True
            )
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=sampler, num_workers=args.num_workers,
                                      pin_memory=(device.type == "cuda"), drop_last=True)
            print("  Using WeightedRandomSampler for balanced training")
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=(device.type == "cuda"), drop_last=True)

        val_loader  = DataLoader(val_ds,  batch_size=args.batch_size*2, shuffle=False,
                                 num_workers=args.num_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size*2, shuffle=False,
                                 num_workers=args.num_workers)

    print(f"\n  pos_weight: {pos_weight.cpu().numpy().round(2)}")
    print(f"  Classes:    {', '.join(class_names)}\n")

    print(f"  Train batches: {len(train_loader)}  |  Val: {len(val_loader)}  |  Test: {len(test_loader)}\n")

    # Модель
    model = ModelABaseline(
        feat_dim=args.feat_dim, hidden_dim=args.hidden_dim,
        num_classes=args.num_classes, encoder_dropout=args.encoder_dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: ModelABaseline  |  Parameters: {total_params:,}\n")

    # Loss / Optimizer / Scheduler
    criterion = LabelSmoothBCE(pos_weight=pos_weight, smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = GradScaler(enabled=use_amp)

    log_path   = out_dir / "train_log.csv"
    best_ckpt  = out_dir / "best_model.pt"
    last_ckpt  = out_dir / "last_checkpoint.pt"

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_auroc,val_fmax,val_f1,lr,time_s\n")

    best_auroc = -float("inf")
    no_improve = 0

    print(f"{'Ep':>4}  {'TrainLoss':>9}  {'ValLoss':>7}  {'AUROC':>6}  {'Fmax':>6}  {'F1':>6}  {'LR':>9}  Time")
    print("-"*72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss  = train_one_epoch(model, train_loader, criterion,
                                      optimizer, scaler, device, args.accum_steps)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        flag    = " *" if val_metrics["auroc"] > best_auroc else ""

        print(f"{epoch:>4}  {train_loss:>9.4f}  {val_metrics['loss']:>7.4f}  "
              f"{val_metrics['auroc']:>6.4f}  {val_metrics['fmax']:>6.4f}  "
              f"{val_metrics['f1']:>6.4f}  {cur_lr:>9.2e}  {elapsed:.1f}s{flag}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_metrics['loss']:.6f},"
                    f"{val_metrics['auroc']:.6f},{val_metrics['fmax']:.6f},"
                    f"{val_metrics['f1']:.6f},{cur_lr:.2e},{elapsed:.1f}\n")

        # Last checkpoint (для resume)
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler":    scaler.state_dict(),
            "val_auroc": val_metrics["auroc"],
            "args":      vars(args),
        }, last_ckpt)

        # Best checkpoint
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            no_improve = 0
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "val_auroc": best_auroc,
                "args":      vars(args),
            }, best_ckpt)
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"\n  Early stopping after {args.patience} epochs without improvement.")
            break

    print(f"\n  Best val AUROC: {best_auroc:.4f}  ->  {best_ckpt}")

    print("\nEvaluating on test set (Test Set)...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = evaluate(model, test_loader, criterion, device)

    from src.eval import PTBXLEvaluator
    evaluator = PTBXLEvaluator(class_names=class_names, seed=args.seed, verbose=False)
    test_results = evaluator.evaluate(test_m["labels"], test_m["probs"], threshold=0.5)
    print_test_report(test_results, class_names, title="TEST - MODEL_A_BASELINE")

    results_path = out_dir / "test_results.json"
    with open(results_path, "w") as f:
        serializable = _json_safe(test_results)
        serializable["macro_auroc"] = serializable.get("macro_auc", 0.0)
        serializable["opt_thr"] = serializable.get("optimal_threshold", 0.5)
        json.dump(serializable, f, indent=2)

    print(f"\n  Best model   : {best_ckpt}")
    print(f"  Last ckpt    : {last_ckpt}")
    print(f"  Train log    : {log_path}")
    print(f"  Test results : {results_path}\n")


if __name__ == "__main__":
    main()
