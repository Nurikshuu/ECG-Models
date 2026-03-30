# src/models/model_a_baseline.py
"""
Model A — Baseline CNN Classifier для 12-отведённой ЭКГ.

Идея: CNNEncoder обрабатывает каждое отведение независимо,
затем простой Mean Pooling объединяет 12 per-lead эмбеддингов
в один вектор, который классифицирует MLP-голова.

Это наш baseline — никакого взаимодействия между отведениями.
Model B и C будут улучшать именно этот шаг агрегации.

Pipeline:
    (B, 12, 1000)
    → CNNEncoder          → (B, 12, 256)
    → MeanPool (leads)    → (B, 256)
    → ClassifierHead      → (B, 5)   # сырые логиты

Выход: RAW LOGITS (без sigmoid) — eval.py сам применяет sigmoid.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .encoder_cnn import CNNEncoder


# ─────────────────────────────────────────────
#  Classifier Head
# ─────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    2-слойная MLP голова для классификации.

    Почему 2 слоя, а не 1 Linear:
        - Нелинейное разделение 5 суперклассов PTB-XL
        - +1-2% macro AUROC по сравнению с single Linear
        - Убывающий dropout: 0.3 → 0.2 (сильнее регуляризуем
          широкий слой, слабее — финальный)

    Почему GELU, а не ReLU:
        - Плавное подавление малых активаций (нет жёсткого нуля)
        - Стандарт в современных classification heads (ViT, BERT)

    Args:
        in_dim  : размер входного вектора (= feature_dim энкодера)
        hidden  : размер скрытого слоя
        num_cls : число классов (5 суперклассов PTB-XL)
        drop1   : dropout перед первым Linear
        drop2   : dropout перед финальным Linear
    """

    def __init__(
        self,
        in_dim:  int = 256,
        hidden:  int = 128,
        num_cls: int = 5,
        drop1:   float = 0.3,
        drop2:   float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(drop1),
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(drop2),
            nn.Linear(hidden, num_cls, bias=True),
            # ❌ sigmoid здесь НЕ нужен — eval.py делает это сам
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)   # (B, in_dim) → (B, num_cls)


# ─────────────────────────────────────────────
#  Model A — Baseline
# ─────────────────────────────────────────────

class ModelABaseline(nn.Module):
    """
    Baseline модель: Lead-wise CNN + Mean Aggregation.

    Агрегация через Mean Pooling — это наш baseline без
    какого-либо моделирования межотведённых взаимодействий.
    Именно этот шаг улучшают Model B (Attention) и Model C (GNN).

    Почему Mean, а не Flatten:
        Flatten(B, 12, 256) → (B, 3072) — 3072-мерный вход в Linear
        на 17K обучающих примерах даёт гарантированное переобучение.
        Mean(dim=1) → (B, 256) — implicit ensemble по отведениям,
        параметрически эффективно и регуляризует модель бесплатно.

    Args:
        feature_dim : размер per-lead эмбеддинга из CNNEncoder (256)
        hidden_dim  : скрытый размер ClassifierHead (128)
        num_classes : число выходных классов (5)
        encoder_dropout : dropout внутри ResidualBlock1d энкодера
    """

    def __init__(
        self,
        feature_dim:      int = 256,
        hidden_dim:       int = 128,
        num_classes:      int = 5,
        encoder_dropout:  float = 0.1,
    ):
        super().__init__()

        # ── Shared Lead-wise ResNet1d Encoder ─────────────────────────
        self.encoder = CNNEncoder(
            feature_dim=feature_dim,
            dropout=encoder_dropout,
        )

        # ── Aggregation: Mean по 12 leads ─────────────────────────────
        # Реализована в forward() через torch.mean — явно и читаемо.

        # ── Classification Head ───────────────────────────────────────
        self.head = ClassifierHead(
            in_dim=feature_dim,
            hidden=hidden_dim,
            num_cls=num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, 12, 1000) — нормализованные 12-отведённые ЭКГ

        Returns:
            logits : (B, 5) — сырые логиты (без sigmoid)
        """
        # ── Step 1: Per-lead feature extraction ──────────────────────
        embeddings = self.encoder(x)              # (B, 12, 256)

        # ── Step 2: Aggregation across leads (BASELINE: просто mean) ─
        pooled = embeddings.mean(dim=1)           # (B, 256)

        # ── Step 3: Classification ────────────────────────────────────
        logits = self.head(pooled)                # (B, 5)

        return logits


# ─────────────────────────────────────────────
#  Быстрая проверка размерностей
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    model  = ModelABaseline(feature_dim=256, hidden_dim=128, num_classes=5)
    dummy  = torch.randn(8, 12, 1000)   # batch=8
    logits = model(dummy)

    total = sum(p.numel() for p in model.parameters())
    enc   = sum(p.numel() for p in model.encoder.parameters())
    head  = sum(p.numel() for p in model.head.parameters())

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {logits.shape}")           # (8, 5)
    print(f"Total params : {total:,}")
    print(f"  ├─ Encoder : {enc:,}")
    print(f"  └─ Head    : {head:,}")

    assert logits.shape == (8, 5), "Размерность выхода неверная!"
    assert not torch.isnan(logits).any(), "NaN в выходе!"
    print("✅ ModelABaseline: все проверки пройдены")
