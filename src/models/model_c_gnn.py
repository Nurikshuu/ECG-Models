# src/models/model_c_gnn.py
"""
Model C — CNN + Graph Attention Network (GAT) для 12-отведённой ЭКГ.

Научная новизна: явное моделирование клинических взаимосвязей
между 12 отведениями через граф + Multi-head Attention.

Pipeline:
    (B, 12, 1000)
    → CNNEncoder                → (B, 12, 256)   per-lead embeddings
    → 2-layer GAT (клин. граф) → (B, 12, 256)   обогащённые features
    → Global Attention Pooling  → (B, 256)        граф-уровень
    → ClassifierHead            → (B, 5)          сырые логиты

Граф: 12 узлов = 12 отведений
      Рёбра = клинические связи (Эйнтховен, inferior, lateral,
              anterior, precordial chain) — фиксированы, не обучаются

Pure PyTorch — PyTorch Geometric НЕ требуется.
Выход: RAW LOGITS (без sigmoid) — eval.py сам применяет sigmoid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder_cnn import CNNEncoder


# ─────────────────────────────────────────────
#  Клинический граф отведений
# ─────────────────────────────────────────────

def build_clinical_adjacency() -> Tensor:
    """
    Строит бинарную матрицу смежности 12×12 на основе клинических
    и электрических связей между отведениями ЭКГ.

    Индексы отведений:
        I=0, II=1, III=2, aVR=3, aVL=4, aVF=5,
        V1=6, V2=7, V3=8, V4=9, V5=10, V6=11

    Группы (важны для диплома):
        Нижние  (inferior):   II(1), III(2), aVF(5)
        Боковые (lateral):    I(0), aVL(4), V5(10), V6(11)
        Передние (anterior):  V1(6)–V6(11) прекордиальная цепь
        Треугольник Эйнтховена: I(0), II(1), III(2)
        Производные:          aVR = −(I+II)/2, aVL = (I−III)/2,
                              aVF = (II+III)/2
    """
    N = 12
    A = torch.zeros(N, N)

    edges = [
        # ── Треугольник Эйнтховена ─────────────────────────
        (0, 1),  # I  — II
        (0, 2),  # I  — III
        (1, 2),  # II — III

        # ── Производные отведения (математические связи) ───
        (3, 0), (3, 1),          # aVR ← I, II
        (4, 0), (4, 2),          # aVL ← I, III
        (5, 1), (5, 2),          # aVF ← II, III

        # ── Нижняя группа (inferior territory) ─────────────
        (1, 5), (2, 5),          # II-aVF, III-aVF

        # ── Боковая группа (lateral territory) ─────────────
        (0, 4),                  # I — aVL
        (0, 10), (0, 11),        # I — V5, V6
        (4, 10), (4, 11),        # aVL — V5, V6
        (10, 11),                # V5 — V6

        # ── Прекордиальная цепь (anterior territory) ────────
        (6, 7), (7, 8),          # V1-V2, V2-V3
        (8, 9), (9, 10),         # V3-V4, V4-V5
        (10, 11),                # V5-V6  (уже есть, не страшно)

        # ── Прекордиальные соседи через один ────────────────
        (6, 8), (7, 9),          # V1-V3, V2-V4
        (8, 10), (9, 11),        # V3-V5, V4-V6

        # ── Межгрупповые (септальные и правые) ──────────────
        (5, 6),                  # aVF — V1  (нижне-септальный)
        (3, 6),                  # aVR — V1  (правосторонний)
        (4, 11),                 # aVL — V6  (лево-боковой)
    ]

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0   # граф ненаправленный

    # Петли (self-loops): каждый узел агрегирует и себя
    A = A + torch.eye(N)
    A = A.clamp(max=1.0)   # убираем дубликаты

    return A  # (12, 12), binary


# ─────────────────────────────────────────────
#  Multi-head Graph Attention Layer
# ─────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer (Veličković et al., ICLR 2018).
    Pure PyTorch — zero external dependencies.

    Декомпозированная формула внимания (эффективнее оригинала):
        Attention источника: f_src_i = h_i · a_src   ∈ R^H
        Attention цели:      f_dst_j = h_j · a_dst   ∈ R^H
        e_ij = LeakyReLU(f_src_i + f_dst_j)           per head
        α_ij = softmax_{j ∈ N(i)}(e_ij)               masked softmax
        h'_i = Σ_j α_ij · (W h_j)                     aggregation

    Эквивалентно оригинальному a^T [Wh_i || Wh_j], но без явного
    построения (B, N, N, H, 2d) тензора — экономит память.

    Args:
        in_dim     : входная размерность фич узла
        out_dim    : размерность на одну голову
        num_heads  : число голов внимания
        dropout    : dropout на веса внимания
        concat     : True = concat голов, False = среднее (для последнего слоя)
    """

    def __init__(
        self,
        in_dim:    int,
        out_dim:   int,
        num_heads: int  = 4,
        dropout:   float = 0.1,
        concat:    bool  = True,
    ):
        super().__init__()
        self.H      = num_heads
        self.d      = out_dim
        self.concat = concat

        # Линейная проекция для всех голов сразу
        self.W = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        # Декомпозированные векторы внимания: a_src, a_dst ∈ R^{H×d}
        self.a_src = nn.Parameter(torch.empty(1, 1, num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, 1, num_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_drop  = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x:   (B, N, in_dim)  — фичи узлов
            adj: (N, N)          — бинарная матрица смежности (device=x.device)
        Returns:
            (B, N, H*d) если concat=True, else (B, N, d)
        """
        B, N, _ = x.shape

        # ── Проекция: (B, N, H, d) ────────────────────────────────────
        h = self.W(x).view(B, N, self.H, self.d)

        # ── Декомпозированное внимание ────────────────────────────────
        # f_src[b, i, h] = Σ_d h[b,i,h,d] * a_src[h,d]  → (B, N, H)
        f_src = (h * self.a_src).sum(dim=-1)  # (B, N, H)
        f_dst = (h * self.a_dst).sum(dim=-1)  # (B, N, H)

        # e_ij = LeakyReLU(f_src_i + f_dst_j) → (B, N, N, H)
        e = self.leaky_relu(
            f_src.unsqueeze(2) + f_dst.unsqueeze(1)  # broadcast: i × j
        )

        # ── Маска несуществующих рёбер ────────────────────────────────
        # adj == 0 → -inf перед softmax
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        e = e.masked_fill(mask, float("-inf"))

        # ── Softmax по соседям (dim=2 = по j) ────────────────────────
        alpha = F.softmax(e, dim=2)            # (B, N, N, H)
        alpha = torch.nan_to_num(alpha, nan=0.0)   # изолированные узлы → 0
        alpha = self.attn_drop(alpha)

        # ── Агрегация: h'_i = Σ_j α_ij · h_j ────────────────────────
        # alpha: (B, N, N, H) → (B, H, N, N)
        # h:     (B, N, H, d) → (B, H, N, d)
        alpha = alpha.permute(0, 3, 1, 2)     # (B, H, N, N)
        h     = h.permute(0, 2, 1, 3)         # (B, H, N, d)
        h_out = torch.matmul(alpha, h)         # (B, H, N, d)
        h_out = h_out.permute(0, 2, 1, 3)     # (B, N, H, d)

        if self.concat:
            return h_out.reshape(B, N, self.H * self.d)  # (B, N, H*d)
        else:
            return h_out.mean(dim=2)                      # (B, N, d)


# ─────────────────────────────────────────────
#  2-layer GAT Module с Residual + LayerNorm
# ─────────────────────────────────────────────

class GNNModule(nn.Module):
    """
    2-слойный GAT с остаточными связями и LayerNorm.

    Архитектура:
        Input  (B, 12, 256)
          │
          ├─ GAT Layer 1: 256 → 4×64=256, concat ─ ELU ─ LN ─ Drop
          │    + residual ─────────────────────────────────────────────
          │
          ├─ GAT Layer 2: 256 → 4×64=256, concat ─ ELU ─ LN ─ Drop
          │    + residual ─────────────────────────────────────────────
          │
        Output (B, 12, 256)

    Почему LayerNorm, а не BatchNorm:
        N=12 узлов — слишком мало для стабильного BN по батч-размерности.
        LayerNorm нормирует по feature-размерности, не зависит от N.

    Почему ELU, а не ReLU:
        ELU — стандартная активация в GAT (оригинальная статья).
        Отрицательные значения → лучший градиентный поток.
    """

    def __init__(
        self,
        in_dim:    int   = 256,
        num_heads: int   = 4,
        head_dim:  int   = 64,
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert in_dim == num_heads * head_dim, (
            f"in_dim={in_dim} должен равняться num_heads×head_dim={num_heads*head_dim}"
        )

        # Layer 1
        self.gat1 = GATLayer(in_dim, head_dim, num_heads, dropout, concat=True)
        self.ln1  = nn.LayerNorm(in_dim)

        # Layer 2
        self.gat2 = GATLayer(in_dim, head_dim, num_heads, dropout, concat=True)
        self.ln2  = nn.LayerNorm(in_dim)

        self.elu  = nn.ELU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        x:   (B, 12, 256)
        adj: (12, 12)  — бинарная матрица смежности
        →    (B, 12, 256)
        """
        # ── GAT Layer 1 + Pre-LN + Residual ──────────────────────────
        h = self.ln1(x)
        h = self.gat1(h, adj)
        h = self.elu(h)
        h = self.drop(h)
        h = h + x               # residual

        # ── GAT Layer 2 + Pre-LN + Residual ──────────────────────────
        h2 = self.ln2(h)
        h2 = self.gat2(h2, adj)
        h2 = self.elu(h2)
        h2 = self.drop(h2)
        h2 = h2 + h             # residual

        return h2               # (B, 12, 256)


# ─────────────────────────────────────────────
#  Global Attention Pooling
# ─────────────────────────────────────────────

class GlobalAttentionPooling(nn.Module):
    """
    Обученный взвешенный пулинг по узлам графа.

    Лучше Mean Pooling: разные отведения несут разный вклад
    в конкретный диагноз.
    Например: для inferior MI важны II, III, aVF → их вес выше.

    Формула (softmax-версия — proper probability distribution):
        score_i = MLP(h_i) ∈ R           (gating network)
        α_i     = softmax_i(score_i)     (нормировка по узлам)
        pooled  = Σ_i α_i · h_i          (взвешенное суммирование)

    Архитектура MLP: Linear → Tanh → Linear(→1)
    """

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, N, d) → pooled: (B, d)
        """
        scores = self.gate(x)                    # (B, N, 1)
        alpha  = F.softmax(scores, dim=1)        # (B, N, 1) — softmax по узлам
        pooled = (alpha * x).sum(dim=1)          # (B, d)
        return pooled


# ─────────────────────────────────────────────
#  Classifier Head (идентичен Model A/B)
# ─────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    2-слойная MLP голова.
    Идентична Model A/B для честного сравнения в дипломе.
    """

    def __init__(
        self,
        in_dim:  int   = 256,
        hidden:  int   = 128,
        num_cls: int   = 5,
        drop1:   float = 0.3,
        drop2:   float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop1),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop2),
            nn.Linear(hidden, num_cls),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  Model C — главный класс
# ─────────────────────────────────────────────

class ModelCGNN(nn.Module):
    """
    Model C: Lead-wise CNN Encoder + Graph Attention Network.

    Ключевое отличие от Model A (Mean) и Model B (SE-Attention):
        GAT явно моделирует СТРУКТУРНЫЕ связи между отведениями
        на основе клинической топологии ЭКГ.

        Пример: для диагностики inferior MI нужны сигналы
        II, III и aVF одновременно. GAT учится агрегировать
        эту информацию через клинические рёбра графа.

    Граф фиксирован (A_clinical) — не обучается.
    Это осознанное решение: клинические связи известны заранее,
    и их фиксация является inductive bias, а не ограничением.

    Args:
        feature_dim       : размер per-lead эмбеддинга (256)
        gat_heads         : число голов GAT (4)
        gat_head_dim      : размерность на голову (64) → 4×64=256
        gnn_dropout       : dropout в GAT слоях
        pool_hidden       : скрытый слой Global Attention Pooling
        hidden_dim        : скрытый слой ClassifierHead
        num_classes       : число выходных классов (5)
        encoder_dropout   : dropout в ResidualBlock1d энкодера
    """

    def __init__(
        self,
        feature_dim:     int   = 256,
        gat_heads:       int   = 4,
        gat_head_dim:    int   = 64,
        gnn_dropout:     float = 0.1,
        pool_hidden:     int   = 128,
        hidden_dim:      int   = 128,
        num_classes:     int   = 5,
        encoder_dropout: float = 0.1,
    ):
        super().__init__()

        assert feature_dim == gat_heads * gat_head_dim, (
            f"feature_dim={feature_dim} должен равняться "
            f"gat_heads×gat_head_dim={gat_heads*gat_head_dim}"
        )

        # ── Lead-wise ResNet1d Encoder ─────────────────────────────
        self.encoder = CNNEncoder(
            feature_dim=feature_dim,
            dropout=encoder_dropout,
        )

        # ── Клинический граф (фиксированный, не обучается) ─────────
        # register_buffer: граф переезжает на GPU вместе с моделью
        adj = build_clinical_adjacency()   # (12, 12)
        self.register_buffer("adj", adj)

        # ── 2-layer GAT ────────────────────────────────────────────
        self.gnn = GNNModule(
            in_dim=feature_dim,
            num_heads=gat_heads,
            head_dim=gat_head_dim,
            dropout=gnn_dropout,
        )

        # ── Global Attention Pooling ────────────────────────────────
        self.pool = GlobalAttentionPooling(
            in_dim=feature_dim,
            hidden=pool_hidden,
        )

        # ── Classification Head ─────────────────────────────────────
        self.head = ClassifierHead(
            in_dim=feature_dim,
            hidden=hidden_dim,
            num_cls=num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 12, 1000) — нормализованные 12-отведённые ЭКГ

        Returns:
            logits: (B, 5) — сырые логиты (без sigmoid)
        """
        # ── Step 1: Per-lead feature extraction ──────────────────
        node_feats = self.encoder(x)              # (B, 12, 256)

        # ── Step 2: Graph message passing (GAT) ──────────────────
        node_feats = self.gnn(node_feats, self.adj)   # (B, 12, 256)

        # ── Step 3: Graph-level readout ───────────────────────────
        graph_repr = self.pool(node_feats)        # (B, 256)

        # ── Step 4: Classification ────────────────────────────────
        logits = self.head(graph_repr)            # (B, 5)

        return logits


# ─────────────────────────────────────────────
#  Быстрая проверка размерностей
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    model  = ModelCGNN(
        feature_dim=256, gat_heads=4, gat_head_dim=64,
        gnn_dropout=0.1, num_classes=5,
    )
    dummy  = torch.randn(8, 12, 1000)
    logits = model(dummy)

    total   = sum(p.numel() for p in model.parameters())
    enc     = sum(p.numel() for p in model.encoder.parameters())
    gnn     = sum(p.numel() for p in model.gnn.parameters())
    pool    = sum(p.numel() for p in model.pool.parameters())
    head    = sum(p.numel() for p in model.head.parameters())

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {logits.shape}")          # (8, 5)
    print(f"\nTotal params : {total:,}")
    print(f"  ├─ Encoder : {enc:,}")
    print(f"  ├─ GNN     : {gnn:,}")
    print(f"  ├─ Pool    : {pool:,}")
    print(f"  └─ Head    : {head:,}")

    # Проверяем граф
    adj = model.adj
    print(f"\nГраф отведений:")
    print(f"  Узлов  : {adj.shape[0]}")
    print(f"  Рёбер  : {int((adj - torch.eye(12)).sum().item() / 2)}")
    print(f"  Степени узлов: {adj.sum(dim=1).int().tolist()}")

    assert logits.shape == (8, 5), "Размерность выхода неверная!"
    assert not torch.isnan(logits).any(), "NaN в выходе!"
    print("\n✅ ModelCGNN: все проверки пройдены")
