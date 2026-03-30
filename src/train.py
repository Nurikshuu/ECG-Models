#!/usr/bin/env python3
"""
Universal SOTA Training Script - PTB-XL ECG Classification v2.0
Supports: model_a_baseline | model_b_attention | model_c_gnn

Улучшения v2.0 vs v1.0:
    ✅ Linear Warmup + CosineAnnealing (WarmupCosineScheduler)
    ✅ Gradient Accumulation (effective_batch = batch × accum_steps)
    ✅ PyTorch 2.3+ AMP API (torch.amp.GradScaler)
    ✅ cudnn.benchmark = True (+15-20% скорости)
    ✅ Label Smoothing для шумных меток PTB-XL
    ✅ Resume Training (--resume флаг)
    ✅ Полный checkpoint: epoch, scheduler, scaler state

Usage:
    # Новое обучение
    python train.py --model model_a_baseline

    # Продолжить с чекпоинта (Colab прервался)
    python train.py --model model_a_baseline --resume

    # Все параметры
    python train.py --model model_c_gnn \
        --epochs 60 --batch_size 64 --accum_steps 2 \
        --warmup_epochs 5 --label_smoothing 0.1 \
        --use_weighted_sampler
"""

import argparse
import csv
import importlib
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from src.data import get_dataloaders, get_class_weights_from_dataloader, set_seed
from src.eval import PTBXLEvaluator


def _ensure_utf8_console() -> None:
    if os.name != "nt":
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        return
    except Exception:
        pass

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        try:
            setattr(sys, stream_name, io.TextIOWrapper(buffer, encoding="utf-8", errors="replace"))
        except Exception:
            continue


def _make_grad_scaler(use_amp: bool) -> Optional[Any]:
    if not use_amp:
        return None

    amp_module = getattr(torch, "amp", None)
    amp_scaler_cls = getattr(amp_module, "GradScaler", None) if amp_module is not None else None
    if amp_scaler_cls is not None:
        return amp_scaler_cls("cuda")

    cuda_amp_module = getattr(torch.cuda, "amp", None)
    cuda_amp_scaler_cls = getattr(cuda_amp_module, "GradScaler", None) if cuda_amp_module is not None else None
    if cuda_amp_scaler_cls is not None:
        return cuda_amp_scaler_cls()

    return None


# ─────────────────────────────────────────────
#  Argument Parser
# ─────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTB-XL SOTA Training v2.0")

    # Модель
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["model_a_baseline", "model_b_attention", "model_c_gnn", "model_c_plus"],
        help="Какую модель обучать"
    )

    # Данные
    parser.add_argument(
        "--data", type=str,
        default="data_preprocessed/ptbxl_sota_100hz_diagnostic_superclass.npz",
        help="Путь к .npz файлу"
    )

    # Гиперпараметры обучения
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument(
        "--accum_steps", type=int, default=2,
        help="Gradient accumulation steps. effective_batch = batch_size × accum_steps"
    )

    # Scheduler
    parser.add_argument(
        "--warmup_epochs", type=int, default=5,
        help="Эпох линейного прогрева LR (0 → 1e-3)"
    )

    # Loss
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help="Label smoothing для шумных меток PTB-XL (0 = выключено)"
    )

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Эпох без улучшения val AUROC до остановки"
    )

    # Окружение
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="DataLoader workers (0 = Colab/Windows)"
    )
    parser.add_argument(
        "--use_weighted_sampler", action="store_true",
        help="WeightedRandomSampler для балансировки"
    )

    # Resume
    parser.add_argument(
        "--resume", action="store_true",
        help="Продолжить обучение с последнего чекпоинта"
    )

    # Сохранение
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed",       type=int, default=42)

    parser.add_argument(
        "--fast_dev_run", action="store_true",
        help="Быстрый smoke-test: ограничивает число батчей train/val/test"
    )
    parser.add_argument(
        "--log_every_n_steps", type=int, default=0,
        help="Печатать прогресс каждые N train-steps (0 = выключено)"
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
#  SOTA: Warmup + Cosine Scheduler
# ─────────────────────────────────────────────

def build_scheduler(
    optimizer:     torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs:  int,
):
    """
    Linear Warmup (0 → lr за warmup_epochs эпох)
    → CosineAnnealingLR (lr → eta_min за оставшиеся эпохи)

    Зачем warmup:
        В начале обучения BatchNorm не откалиброван, веса случайны.
        Большой LR сразу = нестабильные градиенты = плохой старт.
        Warmup даёт модели "разогреться" перед полным шагом.
    """
    if warmup_epochs <= 0:
        # Только Cosine без warmup
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # Линейный прогрев: LR = base_lr * (epoch / warmup_epochs)
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs
    )

    # Косинусный анил на оставшихся эпохах
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )

    # SequentialLR: сначала warmup, потом cosine
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )


# ─────────────────────────────────────────────
#  SOTA: Label Smoothing BCE Loss
# ─────────────────────────────────────────────

class LabelSmoothingBCE(nn.Module):
    """
    BCEWithLogitsLoss + Label Smoothing для multi-label задачи.

    Зачем:
        PTB-XL содержит шумные метки — разные кардиологи могут
        по-разному разметить один и тот же ЭКГ сигнал.
        Label smoothing (ε=0.1) заменяет жёсткие 0/1 на 0.05/0.95,
        не давая модели быть чрезмерно уверенной.
        Доказано: +0.3–0.5% macro AUROC на медицинских датасетах.

    Формула:
        y_smooth = y × (1 - ε) + ε / 2
    """

    def __init__(
        self,
        pos_weight:       Optional[torch.Tensor] = None,
        label_smoothing:  float = 0.1,
    ):
        super().__init__()
        self.smoothing   = label_smoothing
        self.bce         = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            # y=1 → 1 - ε/2 | y=0 → ε/2
            targets = targets * (1.0 - self.smoothing) + self.smoothing / 2.0
        return self.bce(logits, targets)


# ─────────────────────────────────────────────
#  Model Factory
# ─────────────────────────────────────────────

_MODEL_CLASS_MAP = {
    "model_a_baseline": "ModelABaseline",
    "model_b_attention": "ModelBAttention",
    "model_c_gnn":       "ModelCGNN",
    "model_c_plus":      "ModelCPlus",
}


def load_model(model_name: str) -> nn.Module:
    """Динамический импорт модели по имени — train.py не меняется при добавлении новых моделей."""
    module     = importlib.import_module(f"src.models.{model_name}")
    ModelClass = getattr(module, _MODEL_CLASS_MAP[model_name])
    return ModelClass()


# ─────────────────────────────────────────────
#  Train One Epoch (с Gradient Accumulation)
# ─────────────────────────────────────────────

def train_one_epoch(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
    accum_steps: int = 2,
    scaler:      Optional[Any] = None,
    max_batches: Optional[int] = None,
    log_every_n_steps: int = 0,
) -> float:
    """
    Обучение одной эпохи с Gradient Accumulation.

    Gradient Accumulation:
        Обновление весов происходит раз в accum_steps батчей.
        effective_batch = batch_size × accum_steps.
        Важно: loss делится на accum_steps перед backward()
        чтобы градиенты были в правильном масштабе.

    Gradient Clipping:
        max_norm=1.0 применяется ПОСЛЕ unscale — это важно
        при AMP чтобы не клипить уже масштабированные градиенты.
    """
    model.train()
    total_loss   = 0.0
    optimizer.zero_grad()

    steps_processed = 0
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        x, y        = x.to(device), y.to(device)
        is_last     = (step == len(loader) - 1)
        should_step = ((step + 1) % accum_steps == 0) or is_last

        if scaler is not None:                               # CUDA + AMP
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                # Делим loss на accum_steps для правильного масштаба градиентов
                loss   = criterion(logits, y) / accum_steps
            scaler.scale(loss).backward()

            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        else:                                                # CPU / MPS
            logits = model(x)
            loss   = criterion(logits, y) / accum_steps
            loss.backward()

            if should_step:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accum_steps   # восстанавливаем реальный loss для лога
        steps_processed += 1

        if log_every_n_steps > 0 and (step + 1) % log_every_n_steps == 0:
            print(f"      step {step + 1}/{len(loader)} | loss={total_loss / max(steps_processed, 1):.4f}")

    if steps_processed == 0:
        return 0.0
    return total_loss / steps_processed


# ─────────────────────────────────────────────
#  Validate / Test
# ─────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    evaluator: PTBXLEvaluator,
    device:    torch.device,
    max_batches: Optional[int] = None,
) -> tuple[float, dict]:
    """
    Собирает логиты, применяет sigmoid → вероятности,
    передаёт в PTBXLEvaluator.
    """
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    steps_processed = 0
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item()

        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
        steps_processed += 1

    if steps_processed == 0:
        return 0.0, {"macro_auc": 0.0, "fmax": 0.0, "f1_macro": 0.0, "micro_auc": 0.0, "macro_auprc": 0.0}

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics    = evaluator.evaluate(all_labels, all_probs)

    return total_loss / steps_processed, metrics


# ─────────────────────────────────────────────
#  CSV Logger
# ─────────────────────────────────────────────

class CSVLogger:
    FIELDS = [
        "epoch", "train_loss", "val_loss",
        "val_macro_auc", "val_micro_auc", "val_macro_auprc",
        "val_fmax", "val_f1_macro", "lr", "time_sec",
    ]

    def __init__(self, filepath: str, resume: bool = False):
        self.filepath = filepath
        mode = "a" if resume and Path(filepath).exists() else "w"
        if mode == "w":
            with open(filepath, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, row: dict):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow({k: row.get(k, "") for k in self.FIELDS})


# ─────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """Останавливает обучение если val macro AUROC не растёт `patience` эпох."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_score  = -float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """True → новый лучший результат."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    _ensure_utf8_console()
    args = get_args()
    set_seed(args.seed)

    # ── SOTA: cudnn.benchmark для фиксированного input size ───────────
    # B×12×1000 всегда одинаковый → cuDNN подбирает
    # оптимальный алгоритм свёртки один раз = +15-20% скорости
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False   # benchmark несовместим с deterministic

    # ── Пути ──────────────────────────────────────────────────────────
    output_dir      = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    last_ckpt_path  = output_dir / "last_checkpoint.pt"   # для resume
    csv_path        = output_dir / "train_log.csv"

    # ── Device + AMP ──────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # SOTA API: torch.amp.GradScaler("cuda") с fallback на старый torch.cuda.amp.GradScaler
    scaler = _make_grad_scaler(use_amp)

    effective_batch = args.batch_size * args.accum_steps

    print(f"\n{'='*65}")
    print(f"  PTB-XL SOTA Training v2.0 — {args.model.upper()}")
    print(f"{'='*65}")
    print(f"  Device          : {device}  {'[AMP ON]' if use_amp else '[AMP OFF]'}")
    print(f"  Batch size      : {args.batch_size} × {args.accum_steps} accum = {effective_batch} effective")
    print(f"  LR schedule     : Warmup({args.warmup_epochs}ep) → CosineAnnealing")
    print(f"  Label smoothing : {args.label_smoothing}")
    print(f"  Epochs          : {args.epochs}  |  Patience: {args.patience}")
    print(f"  Output          : {output_dir}")
    print(f"  Resume          : {'YES ← продолжаем' if args.resume else 'NO ← новое обучение'}")

    if args.fast_dev_run:
        print("  Fast dev run    : ON (train=5 batches, val/test=2 batches)")

    # ── DataLoaders ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
        augment_train=not args.fast_dev_run,
        seed=args.seed,
    )
    class_names = train_loader.dataset.class_names

    # ── Loss ──────────────────────────────────────────────────────────
    pos_weight = get_class_weights_from_dataloader(train_loader).to(device)
    criterion  = LabelSmoothingBCE(
        pos_weight=pos_weight,
        label_smoothing=args.label_smoothing,
    )
    print(f"\n  pos_weight      : {np.round(pos_weight.cpu().numpy(), 2)}")
    print(f"  Classes         : {class_names}")

    # ── Модель ────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    model = load_model(args.model).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model           : {model.__class__.__name__}")
    print(f"  Parameters      : {total:,} total  /  {trainable:,} trainable")

    # ── Optimizer + Scheduler ─────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 1
    best_auroc  = -float("inf")

    if args.resume and last_ckpt_path.exists():
        print(f"\n  ⏩ Resuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler and "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_auroc  = ckpt["best_auroc"]
        print(f"     Resumed from epoch {ckpt['epoch']}  |  best AUROC so far: {best_auroc:.4f}")
    elif args.resume:
        print(f"  ⚠️  Checkpoint не найден — начинаем с нуля.")

    # ── Вспомогательные объекты ───────────────────────────────────────
    evaluator  = PTBXLEvaluator(class_names, seed=args.seed, verbose=False)
    early_stop = EarlyStopping(patience=args.patience)
    early_stop.best_score = best_auroc   # синхронизируем с resume
    logger     = CSVLogger(str(csv_path), resume=args.resume)

    # ── Шапка таблицы ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Ep':>4}  {'TrainLoss':>10}  {'ValLoss':>9}  "
          f"{'AUROC':>7}  {'Fmax':>7}  {'F1':>7}  {'LR':>9}  {'Time':>5}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*9}  "
          f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*5}")

    # ── Цикл обучения ─────────────────────────────────────────────────
    max_train_batches = 5 if args.fast_dev_run else None
    max_val_batches = 2 if args.fast_dev_run else None
    max_test_batches = 2 if args.fast_dev_run else None

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, args.accum_steps, scaler,
            max_batches=max_train_batches,
            log_every_n_steps=args.log_every_n_steps,
        )
        val_loss, metrics = validate(
            model, val_loader, criterion, evaluator, device,
            max_batches=max_val_batches,
        )
        scheduler.step()

        macro_auc  = metrics["macro_auc"]
        fmax       = metrics["fmax"]
        f1_macro   = metrics["f1_macro"]
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0
        is_best    = early_stop.step(macro_auc)

        # Строка лога
        marker = " ★" if is_best else ""
        print(f"  {epoch:>4}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
              f"{macro_auc:>7.4f}  {fmax:>7.4f}  {f1_macro:>7.4f}  "
              f"{current_lr:>9.2e}  {elapsed:>4.1f}s{marker}")

        # Сохраняем ЛУЧШИЙ чекпоинт (только веса модели)
        if macro_auc > best_auroc:
            best_auroc = macro_auc
            torch.save({
                "epoch":       epoch,
                "model_name":  args.model,
                "model_state": model.state_dict(),
                "best_auroc":  best_auroc,
                "class_names": class_names,
                "args":        vars(args),
            }, checkpoint_path)

        # Сохраняем ПОСЛЕДНИЙ чекпоинт (для resume — полное состояние)
        torch.save({
            "epoch":           epoch,
            "model_name":      args.model,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict() if scaler else None,
            "best_auroc":      best_auroc,
            "class_names":     class_names,
            "args":            vars(args),
        }, last_ckpt_path)

        # CSV log
        logger.log({
            "epoch":           epoch,
            "train_loss":      round(train_loss, 6),
            "val_loss":        round(val_loss, 6),
            "val_macro_auc":   round(macro_auc, 6),
            "val_micro_auc":   round(metrics.get("micro_auc", 0), 6),
            "val_macro_auprc": round(metrics.get("macro_auprc", 0), 6),
            "val_fmax":        round(fmax, 6),
            "val_f1_macro":    round(f1_macro, 6),
            "lr":              round(current_lr, 8),
            "time_sec":        round(elapsed, 2),
        })

        # Early stopping
        if early_stop.should_stop:
            print(f"\n  ⏹  Early stopping: нет улучшения {args.patience} эпох подряд.")
            break

    # ── Финальный тест ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Загружаю лучший чекпоинт (val AUROC = {best_auroc:.4f})...")

    best_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    _, test_metrics = validate(
        model, test_loader, criterion, evaluator, device,
        max_batches=max_test_batches,
    )
    evaluator.print_results(test_metrics, title=f"TEST — {args.model.upper()}")

    results_path = output_dir / "test_results.json"
    evaluator.save_results(test_metrics, str(results_path))

    print(f"  ✅ Best model   → {checkpoint_path}")
    print(f"  ✅ Last ckpt    → {last_ckpt_path}  (для resume)")
    print(f"  ✅ Train log    → {csv_path}")
    print(f"  ✅ Test results → {results_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
