"""
src/models/ecg_retnet.py

Полная архитектура RetNet-ECG v4 — извлечена из ecg_retnet_v4.ipynb без изменений.

Содержит:
  - DropPath               — Stochastic Depth регуляризация
  - MultiScaleRetention    — Retention с xPos позиционным кодированием
  - RetNetFFN              — SwiGLU Feed-Forward Network
  - RetNetBlock            — один блок RetNet (Retention + FFN + DropPath)
  - MultiScalePatchEmbed   — Inception-style патч-эмбеддинг (k=5/11/25)
  - LeadFusionTransformer  — 2-слойный кросс-лидовый Transformer
  - RetNetECG              — полная модель v4
  - ModelEMA               — Exponential Moving Average wrapper

Использование:
    from src.models.ecg_retnet import RetNetECG, ModelEMA

    model = RetNetECG(num_classes=5, d_model=512, n_blocks=10, num_heads=8, drop_path_rate=0.1)
    logits = model(x)   # x: [B, 12, 1000]  → logits: [B, 5]

    # Inference с EMA:
    ema = ModelEMA(model, decay=0.9999)
    # ...после обучения...
    ema.ema.eval()
    with torch.no_grad():
        logits = ema.ema(x)

Параметры модели: ~33.5M (d_model=512, n_blocks=10, num_heads=8)

Метрики на PTB-XL (EMA + TTA × 15):
    Macro AUROC : 0.9133
    Macro AUPRC : 0.7845
    Fmax        : 0.7264  (threshold=0.575)
"""

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# DropPath (Stochastic Depth)
# ─────────────────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """[NEW v4] Stochastic depth: randomly drops entire residual path.
    Доказано улучшает обобщение у глубоких сетей (+0.2-0.5% AUROC).
    Во время инференса автоматически отключается.
    Источник: https://arxiv.org/abs/1603.09382
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (batch, 1, 1, ...) для broadcast
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep_prob, device=x.device, dtype=x.dtype)
        ) / keep_prob
        return x * mask


# ─────────────────────────────────────────────────────────────────────────────
# MultiScaleRetention
# ─────────────────────────────────────────────────────────────────────────────

class MultiScaleRetention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        gammas = 1.0 - 2.0 ** (
            -(5 + torch.arange(num_heads, dtype=torch.float32) * 3.0 / num_heads)
        )
        self.register_buffer('gammas', gammas)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_G = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, d_model)

        freq = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        freq = 1.0 / (10000 ** (freq / self.head_dim))
        self.register_buffer('freq', freq)

    def _build_decay_mask(self, T, device):
        idx  = torch.arange(T, device=device, dtype=torch.float32)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)
        mask = (diff >= 0).float()
        D = self.gammas.view(-1, 1, 1) ** diff.unsqueeze(0) * mask.unsqueeze(0)
        return D

    def _apply_xpos(self, x, T):
        pos   = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)
        freq  = self.freq.to(x.dtype)
        angles = pos * freq
        cos_a = angles.cos().unsqueeze(0).unsqueeze(2)
        sin_a = angles.sin().unsqueeze(0).unsqueeze(2)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack([x1 * cos_a - x2 * sin_a,
                              x1 * sin_a + x2 * cos_a], dim=-1)
        return x_rot.flatten(-2)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_Q(x).view(B, T, H, D)
        K = self.W_K(x).view(B, T, H, D)
        V = self.W_V(x).view(B, T, H, D)
        G = F.silu(self.W_G(x))

        Q = self._apply_xpos(Q, T)
        K = self._apply_xpos(K, T)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        scale  = D ** -0.5
        attn   = torch.matmul(Q, K.transpose(-2, -1)) * scale
        D_mask = self._build_decay_mask(T, x.device)
        attn   = attn * D_mask.unsqueeze(0)
        denom  = D_mask.sum(dim=-1, keepdim=True).unsqueeze(0).clamp(min=1e-6)
        attn   = attn / denom

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        out = out * G
        return self.W_O(out)


# ─────────────────────────────────────────────────────────────────────────────
# RetNetFFN — SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class RetNetFFN(nn.Module):
    def __init__(self, d_model, ffn_mult=2):
        super().__init__()
        d_ff = int(d_model * ffn_mult)
        self.W1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        ab = self.W1(x)
        a, b = ab.chunk(2, dim=-1)
        return self.W2(F.silu(a) * b)


# ─────────────────────────────────────────────────────────────────────────────
# RetNetBlock
# ─────────────────────────────────────────────────────────────────────────────

class RetNetBlock(nn.Module):
    """[UPDATED v4] Добавлен DropPath для stochastic depth."""
    def __init__(self, d_model, num_heads, ffn_mult=2, drop_path=0.0):
        super().__init__()
        self.ln1       = nn.LayerNorm(d_model)
        self.retention = MultiScaleRetention(d_model, num_heads)
        self.ln2       = nn.LayerNorm(d_model)
        self.ffn       = RetNetFFN(d_model, ffn_mult)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.retention(self.ln1(x)))
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MultiScalePatchEmbed
# ─────────────────────────────────────────────────────────────────────────────

class MultiScalePatchEmbed(nn.Module):
    """Inception-style патч-эмбеддинг с тремя параллельными ядрами.

    k=5  (stride=5) → 200 токенов — детали морфологии (QRS спайки, зазубрины)
    k=11 (stride=5) → 200 токенов — огибающая QRS комплекса
    k=25 (stride=5) → 200 токенов — волны P/T, медленные осцилляции

    Три ветки конкатенируются по каналам, затем 1×1 Conv → d_model.
    Одна и та же архитектура для каждого лида (применяется к B*12 последовательностям).

    Мотивация: разные клинически значимые паттерны ЭКГ существуют
    на разных временных масштабах. Один патч-размер не покрывает всё.
    """
    def __init__(self, d_model=512, stride=5):
        super().__init__()
        d  = d_model // 3
        d3 = d_model - 2 * d   # остаток (для точного d_model)
        # kernel=5,  pad=0  → output = (1000-5)/5+1 = 200
        # kernel=11, pad=3  → output = (1000+6-11)/5+1 = 200
        # kernel=25, pad=10 → output = (1000+20-25)/5+1 = 200
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, d,  kernel_size=5,  stride=stride, padding=0,  bias=False),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, d,  kernel_size=11, stride=stride, padding=3,  bias=False),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, d3, kernel_size=25, stride=stride, padding=10, bias=False),
            nn.GELU(),
        )
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B*n_leads, 1, L)
        b1 = self.branch1(x)   # (B*n_leads, d,  T)
        b2 = self.branch2(x)   # (B*n_leads, d,  T)
        b3 = self.branch3(x)   # (B*n_leads, d3, T)
        T  = min(b1.shape[-1], b2.shape[-1], b3.shape[-1])  # выравниваем
        out = torch.cat([b1[..., :T], b2[..., :T], b3[..., :T]], dim=1)  # (B*n_leads, d_model, T)
        out = self.proj(out)                                               # (B*n_leads, d_model, T)
        return self.norm(out.transpose(1, 2))                              # (B*n_leads, T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# LeadFusionTransformer
# ─────────────────────────────────────────────────────────────────────────────

class LeadFusionTransformer(nn.Module):
    """Кросс-лидовое внимание: обрабатывает 12 лидов как 12 токенов.

    Клиническая мотивация: диагностика ЭКГ использует пространственные
    паттерны между лидами:
      - II/III/aVF (нижние лиды) → нижний ИМ
      - I/aVL + V5/V6 (боковые) → боковой ИМ
      - V1-V4 → БПНПГ, передний ИМ

    Простое усреднение лидов (v3) теряет эту информацию.
    2 слоя трансформера над 12 лидами добавляют ~2.5M параметров,
    но принципиально улучшают учёт 12-лидовой структуры.

    Learnable lead-position embeddings: модель сама учится, что лид I
    ≠ лид V1 ≠ лид aVF и т.д.
    """
    def __init__(self, d_model=512, num_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # Pre-norm: более стабильное обучение
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lead_embed = nn.Embedding(12, d_model)   # позиционные embeddings для лидов
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, 12, d_model)
        B, n_leads, D = x.shape
        leads = torch.arange(n_leads, device=x.device)
        x = x + self.lead_embed(leads).unsqueeze(0)   # добавляем лидовые позиции
        x = self.encoder(x)                            # (B, 12, d_model)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# RetNetECG v4 — полная модель
# ─────────────────────────────────────────────────────────────────────────────

class RetNetECG(nn.Module):
    """RetNet-ECG v4 — полная архитектура.

    Ключевые изменения по сравнению с v3:
      1. MultiScalePatchEmbed (k=5/11/25) — multi-resolution temporal features
      2. DropPath (stochastic depth rate=0.1) — лучшая регуляризация
      3. LeadFusionTransformer — кросс-лидовое внимание (2 слоя)
      4. d_model=512, num_heads=8 — увеличенная ёмкость

    Forward pipeline:
      (B, 12, 1000) → MultiScalePatchEmbed → (B*12, T=200, 512)
                    → RetNet blocks × 10   → (B*12, T, 512)
                    → temporal mean pool   → (B, 12, 512)
                    → LeadFusion (2L attn) → (B, 12, 512)
                    → lead mean pool       → (B, 512)
                    → classifier           → (B, num_classes)
    """
    def __init__(self, num_classes=5, d_model=512, n_blocks=10,
                 num_heads=8, ffn_mult=2, drop_path_rate=0.1, head_dropout=0.2):
        super().__init__()
        self.patch_embed = MultiScalePatchEmbed(d_model, stride=5)

        # Линейный рост drop_path от 0 до drop_path_rate по блокам
        dp_rates = [
            drop_path_rate * i / max(n_blocks - 1, 1)
            for i in range(n_blocks)
        ]
        self.blocks = nn.ModuleList([
            RetNetBlock(d_model, num_heads, ffn_mult, drop_path=dp_rates[i])
            for i in range(n_blocks)
        ])
        self.norm        = nn.LayerNorm(d_model)
        self.lead_fusion = LeadFusionTransformer(d_model, num_heads=8, n_layers=2)
        self.classifier  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(head_dropout / 2),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        B, n_leads, L = x.shape
        x = x.reshape(B * n_leads, 1, L)   # (B*12, 1, L)
        x = self.patch_embed(x)             # (B*12, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)                   # (B*12, d_model) — temporal pool
        x = x.reshape(B, n_leads, -1)       # (B, 12, d_model)
        x = self.lead_fusion(x)             # (B, 12, d_model) — lead attention
        x = x.mean(dim=1)                   # (B, d_model)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# ModelEMA — Exponential Moving Average
# ─────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """EMA сглаживает параметры модели по обучению.

    Практически бесплатное улучшение инференса (+0.3-0.8% AUROC):
    EMA-веса менее подвержены шуму последних батчей.
    Используется только для валидации/теста — не для backward pass.

    decay=0.9999 → ~7000 шагов для достижения 50% веса текущего состояния.
    """
    def __init__(self, model, decay=0.9999):
        self.ema   = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, m_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.lerp_(m_p.detach(), 1.0 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Быстрая проверка при прямом запуске: python ecg_retnet.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = RetNetECG(
        num_classes=5,
        d_model=512,
        n_blocks=10,
        num_heads=8,
        drop_path_rate=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model     : RetNet-ECG v4')
    print(f'Params    : {n_params:,}')
    print(f'd_model   : 512  |  num_heads: 8  |  head_dim: {512//8}')
    print(f'n_blocks  : 10   |  drop_path_rate: 0.1 (linear schedule)')
    print(f'patch_embed: Multi-Scale k=5/11/25, stride=5 → 200 tokens/lead')
    print(f'lead_fusion: 2-layer cross-lead Transformer')

    x = torch.randn(2, 12, 1000)
    with torch.no_grad():
        out = model(x)

    print(f'\nForward check: {tuple(x.shape)} → {tuple(out.shape)}')
    assert out.shape == (2, 5), f'Ожидали (2, 5), получили {out.shape}'
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    # EMA проверка
    ema = ModelEMA(model, decay=0.9999)
    ema.update(model)
    ema.ema.eval()
    with torch.no_grad():
        out_ema = ema.ema(x)
    assert out_ema.shape == (2, 5)

    print('\n✅ Все проверки пройдены — RetNetECG v4 OK')
