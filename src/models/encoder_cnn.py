# src/models/encoder_cnn.py
"""
Lead-wise ResNet1d Encoder for 12-lead ECG classification.

Design principle: каждое отведение обрабатывается НЕЗАВИСИМО через
shared weights, сохраняя топологию 12 leads для downstream моделей.

Input:  (B, 12, 1000)  — 12 leads, 10 sec @ 100 Hz
Output: (B, 12, feature_dim)  — per-lead embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────
#  SE-Block (Squeeze-and-Excitation по времени)
# ─────────────────────────────────────────────

class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation block по временной оси.
    Учит модель фокусироваться на важных фичах внутри одного отведения.
    Доказано: +0.5–1% macro AUROC на PTB-XL при минимальном overhead.

    r=8 — reduction ratio (баланс качество/параметры).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)  # не меньше 8 нейронов
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # (B*12, C, 1)
            nn.Flatten(),                      # (B*12, C)
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),                      # gate: значения в [0, 1]
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*12, C, T)
        gate = self.se(x).unsqueeze(-1)       # (B*12, C, 1)
        return x * gate                        # channel-wise rescaling


# ─────────────────────────────────────────────
#  Residual Block
# ─────────────────────────────────────────────

class ResidualBlock1d(nn.Module):
    """
    Pre-activation ResNet блок с SE-gate.

    Структура:
        Conv(k=7) → BN → ReLU → Dropout
        Conv(k=7) → BN
        + skip (1×1 conv если размерность меняется)
        → SE-Block → ReLU

    kernel_size=7 → рецептивное поле ~70ms на 100Hz,
    достаточно для захвата QRS-комплекса.
    stride применяется только в первом conv (temporal downsampling).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop  = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.se    = SEBlock1d(out_channels)

        # Skip connection: выравниваем размерность если нужно
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*12, C_in, T)
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))          # без ReLU перед skip
        out = self.se(out)                        # SE recalibration

        return F.relu(out + residual, inplace=True)


# ─────────────────────────────────────────────
#  CNN Encoder — главный класс
# ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Lead-wise ResNet1d Encoder.

    Ключевой приём — reshape trick:
        (B, 12, 1000)
        → (B*12, 1, 1000)   # 12 отведений как независимые сэмплы
        → 4× ResidualBlock   # shared weights, независимая обработка
        → GlobalAvgPool      # (B*12, 256)
        → (B, 12, 256)       # топология leads сохранена!

    Это гарантирует, что Model B (Attention) и Model C (GNN)
    получат чистые per-lead представления без преждевременного
    смешивания информации между отведениями.

    Прогрессия каналов:
        Stem:    1  →  32  | T: 1000  (stride=1)
        Block1:  32 →  64  | T:  500  (stride=2)
        Block2:  64 → 128  | T:  250  (stride=2)
        Block3: 128 → 256  | T:  125  (stride=2)
        Block4: 256 → 256  | T:   63  (stride=2)
        GAP:    256 →  1×  | T:    1

    Параметры:
        feature_dim (int): размер итогового эмбеддинга на отведение.
                           Default=256 — оптимально для PTB-XL.
        dropout (float):   dropout внутри residual блоков. Default=0.1.
    """

    # Архитектура: (in_ch, out_ch, stride)
    _STAGES = [
        (1,   32,  1),   # Stem — без downsampling, сохраняем детали
        (32,  64,  2),   # Block 1 — 1000 → 500
        (64,  128, 2),   # Block 2 —  500 → 250
        (128, 256, 2),   # Block 3 —  250 → 125
        (256, 256, 2),   # Block 4 —  125 →  63
    ]

    def __init__(self, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        assert feature_dim == 256, (
            "Текущая архитектура выдаёт 256 каналов. "
            "Для других значений измените _STAGES."
        )

        # Строим слои из таблицы _STAGES
        layers = []
        for in_ch, out_ch, stride in self._STAGES:
            layers.append(ResidualBlock1d(in_ch, out_ch, stride, dropout))
        self.backbone = nn.Sequential(*layers)

        # Global Average Pooling по временной оси
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.feature_dim = feature_dim
        self._init_weights()

    def _init_weights(self):
        """Kaiming init для Conv, стандартный для BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 12, 1000) — нормализованные 12-отведённые ЭКГ

        Returns:
            embeddings: (B, 12, feature_dim) — per-lead representations
        """
        B, L, T = x.shape          # B=batch, L=12 leads, T=1000

        # ── Reshape: leads → независимые сэмплы ──────────────────────
        x = x.reshape(B * L, 1, T)   # (B*12, 1, 1000)

        # ── Lead-wise ResNet backbone ─────────────────────────────────
        x = self.backbone(x)          # (B*12, 256, ~63)

        # ── Global Average Pooling по временной оси ───────────────────
        x = self.gap(x)               # (B*12, 256, 1)
        x = x.squeeze(-1)             # (B*12, 256)

        # ── Восстанавливаем топологию leads ───────────────────────────
        x = x.reshape(B, L, self.feature_dim)  # (B, 12, 256)

        return x


# ─────────────────────────────────────────────
#  Быстрая проверка размерностей
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    encoder = CNNEncoder(feature_dim=256)
    dummy   = torch.randn(8, 12, 1000)         # batch=8
    out     = encoder(dummy)

    total_params = sum(p.numel() for p in encoder.parameters())

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")        # ожидаем (8, 12, 256)
    print(f"Parameters   : {total_params:,}")   # ~500K — ок для Colab
    assert out.shape == (8, 12, 256), "Размерность выхода неверная!"
    print("✅ CNNEncoder: все проверки пройдены")
