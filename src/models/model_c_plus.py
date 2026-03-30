"""
Model C+ — Multi-Scale CNN + Dual-Graph Adaptive GAT
для PTB-XL (5 суперклассов).

SOTA улучшения над Model C:
    1. Multi-scale Inception-style stem (k=3,5,7,11 + 1×1 mix)
    2. Lead Positional Encoding (12 learnable embeddings)
    3. Dual-Graph: A_clin (learnable edges) + A_feat (per-sample
       cosine similarity) → λ*A_clin + (1-λ)*A_feat, λ learnable
    4. Transformer-style blocks: GAT + FFN + Pre-LN + Residual
    5. Multi-Readout Pooling: mean + attention → proj

Input:  (B, 12, 1000)
Output: (B, 5) — raw logits (без sigmoid)

Pure PyTorch — никаких внешних зависимостей.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────
#  Клинический граф (как в Model C)
# ─────────────────────────────────────────────

def build_clinical_adjacency() -> Tensor:
    """
    Бинарная матрица смежности 12×12.

    I=0, II=1, III=2, aVR=3, aVL=4, aVF=5,
    V1=6, V2=7, V3=8, V4=9, V5=10, V6=11
    """
    N = 12
    A = torch.zeros(N, N)

    edges = [
        # Треугольник Эйнтховена
        (0, 1), (0, 2), (1, 2),
        # Производные
        (3, 0), (3, 1), (4, 0), (4, 2), (5, 1), (5, 2),
        # Нижние
        (1, 5), (2, 5),
        # Боковые
        (0, 4), (0, 10), (0, 11), (4, 10), (4, 11), (10, 11),
        # Прекордиальная цепь
        (6, 7), (7, 8), (8, 9), (9, 10),
        # Соседи через один
        (6, 8), (7, 9), (8, 10), (9, 11),
        # Межгрупповые
        (5, 6), (3, 6), (4, 11),
    ]
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0

    return (A + torch.eye(N)).clamp(max=1.0)


# ─────────────────────────────────────────────
#  SE-block и ResidualBlock1d
# ─────────────────────────────────────────────

class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x).unsqueeze(-1)


class ResidualBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = SEBlock1d(out_ch)
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        r   = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(self.drop(out)))
        return F.relu(self.se(out) + r, inplace=True)


# ─────────────────────────────────────────────
#  Multi-Scale CNN Encoder
# ─────────────────────────────────────────────

class MultiScaleCNNEncoder(nn.Module):
    """
    Inception-style lead-wise ResNet1d.

    Стем: 4 параллельные ветки Conv1d(k=3/5/7/11, 8ch)
          каждая: Conv → BN → ReLU
          concat(32ch) → 1×1 mixing conv → BN → ReLU

    Backbone: те же 4 ResidualBlock1d что в encoder_cnn.py
    Output:   (B, 12, 256)

    Преимущество: k=3 ловит QRS (~30ms), k=7 — ST (~70ms),
    k=11 — T-волну (~110ms), k=5 — переходные паттерны.
    """

    _BRANCH_CH = 8
    _KERNELS    = [3, 5, 7, 11]
    _STAGES     = [(32, 64, 2), (64, 128, 2), (128, 256, 2), (256, 256, 2)]

    def __init__(self, feature_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        stem_ch = self._BRANCH_CH * len(self._KERNELS)  # 32

        # Inception ветки (каждая с BN+ReLU)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self._BRANCH_CH, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(self._BRANCH_CH),
                nn.ReLU(inplace=True),
            )
            for k in self._KERNELS
        ])

        # 1×1 mixing conv
        self.stem_mix = nn.Sequential(
            nn.Conv1d(stem_ch, stem_ch, 1, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.ReLU(inplace=True),
        )

        # ResNet backbone
        layers, in_ch = [], stem_ch
        for (cin, cout, s) in self._STAGES:
            assert cin == in_ch
            layers.append(ResidualBlock1d(cin, cout, s, dropout))
            in_ch = cout
        self.backbone = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)

        x = self.stem_mix(torch.cat([b(x) for b in self.branches], dim=1))
        x = self.gap(self.backbone(x)).squeeze(-1)

        return x.reshape(B, L, self.feature_dim)


# ─────────────────────────────────────────────
#  Dual-Graph Builder
# ─────────────────────────────────────────────

class DualGraphBuilder(nn.Module):
    """
    Строит адаптивную матрицу смежности (B, 12, 12):

        A_clin: клинический граф (фиксированный скелет + learnable
                edge weights → row-normalize)
        A_feat: per-sample feature similarity (cosine, temperature-scaled
                softmax) — разная для каждого ЭКГ!
        A_fused = λ * A_clin + (1-λ) * A_feat,  λ = sigmoid(lambda_logit)

    При инициализации λ ≈ 0.5 — равный вклад обоих графов.
    Модель сама учит, сколько доверять клинике vs данным.

    Аналог "topology graph + feature graph" fusion из работы MSAGFN
    (Chen et al., 2025 — adaptive multi-channel GNN для 12-lead ECG).
    """

    def __init__(self, num_leads: int = 12, init_temp: float | None = None):
        super().__init__()
        if init_temp is None:
            init_temp = math.sqrt(256)  # √feature_dim — как в scaled dot-product

        self.edge_logits  = nn.Parameter(torch.zeros(num_leads, num_leads))
        self.log_temp     = nn.Parameter(torch.tensor(math.log(init_temp)))
        self.lambda_logit = nn.Parameter(torch.tensor(0.0))  # λ_init = 0.5

    def _clinical_adj(self, adj_base: Tensor) -> Tensor:
        """(12,12) learnable-weighted, row-normalized clinical graph."""
        sym      = 0.5 * (self.edge_logits + self.edge_logits.t())
        weights  = torch.sigmoid(sym) * adj_base
        row_sum  = weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return weights / row_sum

    def _feature_adj(self, h: Tensor) -> Tensor:
        """(B,12,12) cosine similarity, temperature-scaled softmax."""
        h_n  = F.normalize(h, dim=-1)
        sim  = torch.bmm(h_n, h_n.transpose(1, 2))
        temp = self.log_temp.exp().clamp(min=0.1)
        return F.softmax(sim / temp, dim=-1)

    def forward(self, node_feats: Tensor, adj_base: Tensor) -> Tensor:
        B   = node_feats.size(0)
        lam = torch.sigmoid(self.lambda_logit)

        A_clin = self._clinical_adj(adj_base).unsqueeze(0).expand(B, -1, -1)
        A_feat = self._feature_adj(node_feats)

        return lam * A_clin + (1.0 - lam) * A_feat  # (B, 12, 12)


# ─────────────────────────────────────────────
#  GAT Layer (soft adjacency gate)
# ─────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Multi-head GAT с soft adjacency gating.

    Формула (structure-aware attention):
        e_ij = LeakyReLU(a_src_i + a_dst_j)           content score
        α_raw = softmax_j(e_ij)                        content attention
        α_gated = α_raw * A_fused[b,i,j]              structural gate
        α = normalize(α_gated)                         renormalize
        h'_i = Σ_j α_ij * W h_j                       aggregate

    Гейт A_fused: клинически сильные связи (высокий вес) + feature-похожие
    leads (высокое cosine) → большой attention. Остальные → маленький.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.H, self.d, self.concat = num_heads, out_dim, concat

        self.W     = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, 1, num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(1, 1, num_heads, out_dim))
        self.leaky = nn.LeakyReLU(0.2)
        self.drop  = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        x:   (B, N, in_dim)
        adj: (B, N, N) — soft adjacency [0, 1]
        """
        B, N, _ = x.shape

        h     = self.W(x).view(B, N, self.H, self.d)
        f_src = (h * self.a_src).sum(-1)        # (B, N, H)
        f_dst = (h * self.a_dst).sum(-1)        # (B, N, H)

        e = self.leaky(f_src.unsqueeze(2) + f_dst.unsqueeze(1))  # (B, N, N, H)
        alpha_raw   = F.softmax(e, dim=2)

        gate        = adj.unsqueeze(-1)                           # (B, N, N, 1)
        alpha_gated = alpha_raw * gate
        alpha       = alpha_gated / (alpha_gated.sum(2, keepdim=True) + 1e-9)
        alpha       = torch.nan_to_num(alpha, nan=0.0)
        alpha       = self.drop(alpha)

        alpha = alpha.permute(0, 3, 1, 2)   # (B, H, N, N)
        h     = h.permute(0, 2, 1, 3)       # (B, H, N, d)
        out   = torch.matmul(alpha, h).permute(0, 2, 1, 3)  # (B, N, H, d)

        return out.reshape(B, N, self.H * self.d) if self.concat else out.mean(2)


# ─────────────────────────────────────────────
#  Transformer-style Dual Graph Block
# ─────────────────────────────────────────────

class DualGraphBlock(nn.Module):
    """
    Transformer-style GNN block с Pre-LN:

        x → LayerNorm → GAT(A_fused) → x + Residual
          → LayerNorm → FFN(4×)       → x + Residual

    Pre-LN (LayerNorm ДО слоя) → стабильнее, чем Post-LN.
    FFN = Linear → GELU → Dropout → Linear → Dropout.
    FFN ratio=4 — стандарт трансформеров, даёт per-node нелинейность.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4,
                 head_dim: int = 64, dropout: float = 0.1, ffn_ratio: int = 4):
        super().__init__()
        assert dim == num_heads * head_dim

        self.ln1 = nn.LayerNorm(dim)
        self.gat = GATLayer(dim, head_dim, num_heads, dropout, concat=True)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout),
        )

        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        x = x + self.gat(self.ln1(x), adj)
        x = x + self.ffn(self.ln2(x))
        return x


# ─────────────────────────────────────────────
#  Multi-Readout Pooling
# ─────────────────────────────────────────────

class MultiReadoutPooling(nn.Module):
    """
    Объединяет mean pooling и attention pooling:
        mean:  среднее по 12 leads — глобальный контекст
        att:   softmax-weighted sum — фокус на важных leads

    output = proj(cat[mean, att]) → (B, d)

    Mean + Attention лучше одного attention:
    mean не теряет общий контекст когда attention фокусируется.
    """

    def __init__(self, in_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )
        self.proj = nn.Linear(in_dim * 2, in_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        mean_p = x.mean(dim=1)
        alpha  = F.softmax(self.gate(x), dim=1)
        att_p  = (alpha * x).sum(dim=1)
        return self.proj(torch.cat([mean_p, att_p], dim=-1))


# ─────────────────────────────────────────────
#  Classifier Head
# ─────────────────────────────────────────────

class ClassifierHead(nn.Module):
    def __init__(self, in_dim=256, hidden=128, num_cls=5,
                 drop1=0.3, drop2=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop1), nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(drop2), nn.Linear(hidden, num_cls),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  ModelCPlus — главный класс
# ─────────────────────────────────────────────

class ModelCPlus(nn.Module):
    """
    Model C+ — Multi-Scale CNN + Dual-Graph Adaptive GAT.

    Каждое из 5 ключевых улучшений над Model C имеет
    научное обоснование:

    1. MultiScaleCNNEncoder:
       - Inception-style stem ловит QRS/ST/T разными ядрами
       - 1×1 mixing учит оптимальную комбинацию масштабов

    2. Lead Positional Encoding:
       - GAT без PE не знает, I это lead или V5
       - 12 learnable vectors → тождественность leads

    3. DualGraphBuilder (A_clin + A_feat):
       - A_clin: клинические связи + learnable силы рёбер
       - A_feat: cosine similarity ТЕКУЩЕГО батча → адаптивно
       - λ: баланс структурного vs data-driven priора

    4. DualGraphBlock (GAT + FFN + Pre-LN):
       - Transformer-style → лучше чем GAT+ELU+Dropout из C
       - FFN даёт per-node нелинейную трансформацию

    5. MultiReadoutPooling (mean + attention):
       - Лучше одиночного GlobalAttentionPooling из C

    Выход: RAW LOGITS (B, 5) — без sigmoid.
    """

    def __init__(
        self,
        feature_dim:     int   = 256,
        gat_heads:       int   = 4,
        gat_head_dim:    int   = 64,
        gnn_dropout:     float = 0.15,
        num_gnn_layers:  int   = 2,
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

        # 1. Multi-scale encoder
        self.encoder = MultiScaleCNNEncoder(feature_dim, encoder_dropout)

        # 2. Lead positional encoding (каждое из 12 отведений — уникальный вектор)
        self.lead_embed = nn.Parameter(torch.zeros(12, feature_dim))
        nn.init.normal_(self.lead_embed, std=0.02)

        # 3. Clinical adjacency skeleton
        self.register_buffer("adj_base", build_clinical_adjacency())

        # 4. Dual-graph builder
        self.graph_builder = DualGraphBuilder(
            num_leads=12,
            init_temp=math.sqrt(feature_dim),
        )

        # 5. Transformer-style GNN blocks
        self.gnn_blocks = nn.ModuleList([
            DualGraphBlock(
                dim=feature_dim, num_heads=gat_heads, head_dim=gat_head_dim,
                dropout=gnn_dropout, ffn_ratio=4,
            )
            for _ in range(num_gnn_layers)
        ])
        self.post_gnn_ln = nn.LayerNorm(feature_dim)

        # 6. Multi-readout pooling
        self.pool = MultiReadoutPooling(feature_dim, pool_hidden)

        # 7. Classifier head
        self.head = ClassifierHead(feature_dim, hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, 12, 1000)  →  logits: (B, 5)
        """
        # Step 1: Lead-wise multi-scale feature extraction
        h = self.encoder(x)                                  # (B, 12, 256)

        # Step 2: Lead identity injection
        h = h + self.lead_embed.unsqueeze(0)                 # (B, 12, 256)

        # Step 3: Build adaptive dual graph (per-sample!)
        A = self.graph_builder(h, self.adj_base)             # (B, 12, 12)

        # Step 4: Transformer-style graph message passing
        for block in self.gnn_blocks:
            h = block(h, A)                                  # (B, 12, 256)
        h = self.post_gnn_ln(h)

        # Step 5: Multi-readout pooling
        g = self.pool(h)                                     # (B, 256)

        # Step 6: Classification
        return self.head(g)                                  # (B, 5)


# ─────────────────────────────────────────────
#  Быстрая проверка
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = ModelCPlus()
    dummy = torch.randn(8, 12, 1000)
    out   = model(dummy)

    def count(m): return sum(p.numel() for p in m.parameters())

    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")
    print(f"\\nParameters:")
    print(f"  ├─ Encoder        : {count(model.encoder):>9,}")
    print(f"  ├─ Lead embed     : {model.lead_embed.numel():>9,}")
    print(f"  ├─ Graph builder  : {count(model.graph_builder):>9,}")
    print(f"  ├─ GNN blocks     : {count(model.gnn_blocks):>9,}")
    print(f"  ├─ Pooling        : {count(model.pool):>9,}")
    print(f"  └─ Head           : {count(model.head):>9,}")
    print(f"  {'─'*26}")
    print(f"  TOTAL             : {count(model):>9,}")

    lam = torch.sigmoid(model.graph_builder.lambda_logit).item()
    print(f"\\nGraph λ init : {lam:.3f}  (0.5 = равный вклад A_clin и A_feat)")

    assert out.shape == (8, 5)
    assert not torch.isnan(out).any()
    print("\\n✅ ModelCPlus: все проверки пройдены")
