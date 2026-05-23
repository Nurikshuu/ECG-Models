"""
src/models/leadwise_gnn.py

Полная архитектура LeadWiseResNetGNN — извлечена из 02_leadwise_gnn_model.ipynb без изменений.

Содержит:
  - Примитивы: AdaptiveConcatPool1d, create_head1d, conv1d_pad, BasicBlock1d, _make_layer
  - ResNet1dWang          — оригинальный baseline (нужен для load_leadwise_from_baseline)
  - LeadWiseResNet1d      — lead-wise encoder, вход [B,12,1000] → [B,12,embed_dim]
  - build_clinical_adj    — клинико-анатомический граф 12 отведений
  - normalize_adj         — D^{-1/2}(A+I)D^{-1/2}
  - GATLayer              — один GAT-слой (Veličković et al., 2018)
  - GNNHead               — 2×GAT + per-class attention readout → logits + XAI
  - LeadWiseResNetGNN     — полная модель (encoder + gnn_head)
  - load_leadwise_from_baseline — перенос весов ResNet1dWang → LeadWiseResNet1d

Использование:
    from src.models.leadwise_gnn import LeadWiseResNetGNN, load_leadwise_from_baseline

    model = LeadWiseResNetGNN(embed_dim=128, hidden_dim=96, num_classes=5, dropout=0.15)
    logits, attn_info = model(x)   # x: [B, 12, 1000]
    # attn_info['attn_leads'] : [B, 5, 12]  — XAI: важность отведений per class
    # attn_info['gat1_alpha'] : [B, 4, 12, 12]
    # attn_info['gat2_alpha'] : [B, 1, 12, 12]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК A: Общие примитивы
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


def create_head1d(nf, nc, ps=0.5):
    return nn.Sequential(
        AdaptiveConcatPool1d(),
        nn.Flatten(),
        nn.BatchNorm1d(2 * nf),
        nn.Dropout(ps / 2),
        nn.Linear(2 * nf, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(ps),
        nn.Linear(512, nc),
    )


def conv1d_pad(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv1d(
        in_planes, out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False,
    )


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 kernel_size=[5, 3], downsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size // 2 + 1]
        self.conv1 = conv1d_pad(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d_pad(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


def _make_layer(in_planes, planes, blocks, stride, kernel_size):
    downsample = None
    if stride != 1 or in_planes != planes * BasicBlock1d.expansion:
        downsample = nn.Sequential(
            nn.Conv1d(in_planes, planes * BasicBlock1d.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes * BasicBlock1d.expansion),
        )
    layers = [BasicBlock1d(in_planes, planes, stride, kernel_size, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock1d(planes, planes, kernel_size=kernel_size))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК B: ResNet1dWang — оригинал (нужен для загрузки весов baseline)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet1dWang(nn.Module):
    def __init__(self, num_classes=5, input_channels=12, ps_head=0.5):
        super().__init__()
        inplanes = 128
        kernel_size = [5, 3]
        self.stem = nn.Sequential(
            conv1d_pad(input_channels, inplanes, stride=1, kernel_size=7),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_layer(inplanes, inplanes, blocks=1, stride=1, kernel_size=kernel_size)
        self.layer2 = _make_layer(inplanes, inplanes, blocks=1, stride=2, kernel_size=kernel_size)
        self.layer3 = _make_layer(inplanes, inplanes, blocks=1, stride=2, kernel_size=kernel_size)
        self.head = create_head1d(inplanes, nc=num_classes, ps=ps_head)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК C: LeadWiseResNet1d — энкодер для GNN
# ─────────────────────────────────────────────────────────────────────────────

class LeadWiseResNet1d(nn.Module):
    """
    Lead-wise версия ResNet1dWang.
    Каждый из 12 leads обрабатывается ОДНОЙ сетью (weight sharing).
    Вход:  [B, 12, 1000]
    Выход: [B, 12, embed_dim]  ← узлы графа GNN
    """
    def __init__(self, embed_dim=128, dropout=0.1):
        super().__init__()
        inplanes = 128
        kernel_size = [5, 3]

        self.stem = nn.Sequential(
            conv1d_pad(1, inplanes, stride=1, kernel_size=7),  # 1 канал вместо 12
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_layer(inplanes, inplanes, blocks=1, stride=1, kernel_size=kernel_size)
        self.layer2 = _make_layer(inplanes, inplanes, blocks=1, stride=2, kernel_size=kernel_size)
        self.layer3 = _make_layer(inplanes, inplanes, blocks=1, stride=2, kernel_size=kernel_size)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(inplanes),
            nn.Dropout(dropout),
            nn.Linear(inplanes, embed_dim, bias=False),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)   # [B*12, 1, 1000]
        h = self.stem(x)              # [B*12, 128, 1000]
        h = self.layer1(h)            # [B*12, 128, 1000]
        h = self.layer2(h)            # [B*12, 128, 500]
        h = self.layer3(h)            # [B*12, 128, 250]
        h = self.pool(h)              # [B*12, 128, 1]
        h = self.proj(h)              # [B*12, embed_dim]
        return h.reshape(B, L, -1)    # [B, 12, embed_dim]


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК D: Перенос весов baseline → LeadWiseResNet1d
# ─────────────────────────────────────────────────────────────────────────────

def load_leadwise_from_baseline(leadwise_model, ckpt_path, device='cpu'):
    """
    Переносит веса из обученного ResNet1dWang в LeadWiseResNet1d.

    Переносится : layer1/2/3 (все Conv+BN), stem BN
    Адаптируется: stem.0.weight: 12-канальный → 1-канальный через mean по каналам
                  (эвристика, лучше чем random init, но не идеал)
    Пропускается : head (убрана)
    """
    baseline_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Поддержка разных форматов чекпоинта
    if isinstance(baseline_state, dict):
        baseline_state = (baseline_state.get('model_state_dict')
                          or baseline_state.get('model')
                          or baseline_state.get('state_dict')
                          or baseline_state)

    leadwise_state = leadwise_model.state_dict()
    transferred, skipped_shape, skipped_absent, adapted = 0, 0, 0, 0

    for k, v in baseline_state.items():
        # head — не нужен
        if k.startswith('head'):
            skipped_absent += 1
            continue

        # Специальный случай: stem.0.weight [128,12,7] → [128,1,7]
        if k == 'stem.0.weight':
            if k in leadwise_state and leadwise_state[k].shape == (128, 1, 7):
                leadwise_state[k] = v.mean(dim=1, keepdim=True)
                adapted += 1
                print(f'  ⚡ stem.0.weight: усреднён [128,12,7]→[128,1,7] (warm init)')
            else:
                skipped_shape += 1
            continue

        # Обычный перенос
        if k in leadwise_state:
            if leadwise_state[k].shape == v.shape:
                leadwise_state[k] = v
                transferred += 1
            else:
                skipped_shape += 1
        else:
            skipped_absent += 1

    leadwise_model.load_state_dict(leadwise_state, strict=False)
    print(f'✅ Перенесено : {transferred} тензоров (layer1/2/3 + stem BN)')
    print(f'   Адаптировано: {adapted} (stem.0.weight warm init)')
    print(f'   Пропущено   : {skipped_absent} (head — ожидаемо)')
    if skipped_shape > 0:
        print(f'   ⚠️ Несовпадение формы: {skipped_shape} тензоров')
    return leadwise_model


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК E: Клинический граф
# Порядок: [I=0, II=1, III=2, aVR=3, aVL=4, aVF=5, V1=6, V2=7, V3=8, V4=9, V5=10, V6=11]
# ─────────────────────────────────────────────────────────────────────────────

def build_clinical_adj():
    """
    Клинико-анатомический граф 12-отведённой ЭКГ.
    Self-loops НЕ добавляются здесь — они добавляются в normalize_adj.

    Группы и медицинская логика связей:
    ─────────────────────────────────────────────────────
    ФРОНТАЛЬНЫЕ (I, II, III, aVR, aVL, aVF):
        Все 6 отведений смотрят на фронтальную плоскость сердца.
        Связи по закону Кирхгофа: I = II − III, aVR = −(I+II)/2.
        → Полный граф (30 направленных рёбер).

    ПРЕКАРДИАЛЬНЫЕ (V1–V6):
        Физически расположены по окружности грудной клетки.
        V1 (правый желудочек) → V6 (левый желудочек) — непрерывная цепочка.
        → Цепочка соседей (10 рёбер) + скип-связи V1↔V3, V2↔V4, V3↔V5, V4↔V6 (8 рёбер).

    МЕЖГРУППОВЫЕ (клинически значимые пары):
        I   ↔ V5 — боковая стенка ЛЖ (единый миокардиальный регион)
        I   ↔ V6 — боковая стенка ЛЖ
        II  ↔ V4 — нижняя + передне-боковая зоны
        aVF ↔ V4 — нижняя диафрагмальная + передняя
        aVL ↔ V5 — высокая боковая + боковая (HYP: гипертрофия ЛЖ)
        aVR ↔ V1 — правые отделы (зеркальные паттерны при БЛНПГ)
    ─────────────────────────────────────────────────────
    """
    A = torch.zeros(12, 12)

    # 1. Фронтальные — полный граф
    frontal = [0, 1, 2, 3, 4, 5]
    for i in frontal:
        for j in frontal:
            if i != j:
                A[i, j] = 1.0

    # 2. Прекардиальные — цепочка соседей
    for i in range(6, 11):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    # 3. Прекардиальные — скип-связи через одно (для захвата зональных паттернов)
    for i in range(6, 10):
        A[i, i + 2] = 1.0
        A[i + 2, i] = 1.0

    # 4. Межгрупповые клинические связи
    cross = [
        (0, 10),  # I   ↔ V5 — боковая стенка
        (0, 11),  # I   ↔ V6 — боковая стенка
        (1, 9),   # II  ↔ V4 — нижняя + передне-боковая
        (5, 9),   # aVF ↔ V4 — нижняя + передняя
        (4, 10),  # aVL ↔ V5 — высокая боковая (важно для HYP)
        (3, 6),   # aVR ↔ V1 — правые отделы
    ]
    for i, j in cross:
        A[i, j] = 1.0
        A[j, i] = 1.0

    return A  # [12, 12], без self-loops


def normalize_adj(A):
    """
    Симметричная нормализация: D^{-1/2} (A + I) D^{-1/2}
    Стандарт из GCN (Kipf & Welling, 2017).
    Self-loops добавляются ТОЛЬКО здесь.
    """
    A = A + torch.eye(A.shape[0])   # self-loops добавляем один раз
    deg = A.sum(dim=1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    D = torch.diag(d_inv_sqrt)
    return D @ A @ D  # [12, 12]


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК F: Graph Attention Layer (чистый PyTorch, без PyG)
# Veličković et al., 2018
# ─────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Один GAT-слой.
    Вход : [B, N, in_dim]
    Выход: [B, N, out_dim * heads]  если concat=True
           [B, N, out_dim]          если concat=False
           + attention weights [B, heads, N, N] для XAI
    """
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x, adj):
        """
        x   : [B, N, in_dim]
        adj : [N, N] нормализованная матрица смежности (с self-loops)
        """
        B, N, _ = x.shape
        H, D = self.heads, self.out_dim

        Wh = self.W(x).view(B, N, H, D)  # [B, N, H, D]

        # e_ij = LeakyReLU(a_src·Wh_i + a_dst·Wh_j)
        src = (Wh * self.a_src).sum(-1)   # [B, N, H]
        dst = (Wh * self.a_dst).sum(-1)   # [B, N, H]

        # [B, H, N, 1] + [B, H, 1, N] → [B, H, N, N]
        e = self.leaky(
            src.permute(0, 2, 1).unsqueeze(-1) +
            dst.permute(0, 2, 1).unsqueeze(-2)
        )

        # Маскируем несуществующие рёбра
        mask = (adj == 0).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        e = e.masked_fill(mask, float('-inf'))

        alpha = F.softmax(e, dim=-1)  # [B, H, N, N]

        # Защита от NaN (возникает если ВСЕ соседи узла замаскированы)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        alpha = self.drop(alpha)

        # Агрегация соседей
        Wh_t = Wh.permute(0, 2, 1, 3)          # [B, H, N, D]
        out = torch.matmul(alpha, Wh_t)          # [B, H, N, D]

        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(B, N, H * D)
        else:
            out = out.mean(dim=1)                # [B, N, D]

        return out, alpha  # alpha → XAI


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК G: GNNHead — 2 слоя GAT + per-class attention readout
# ─────────────────────────────────────────────────────────────────────────────

class GNNHead(nn.Module):
    """
    GNN-head поверх per-lead embeddings.
    Вдохновлён xGNN4MI (npj Digital Medicine, 2026).

    Вход : z [B, 12, node_dim]
    Выход: logits     [B, num_classes]
           attn_leads [B, num_classes, 12]  ← главный XAI-выход
           gat1_alpha [B, heads, 12, 12]    ← внутренние GAT-веса слоя 1
           gat2_alpha [B, 1, 12, 12]        ← внутренние GAT-веса слоя 2

    v1.1: hidden_dim=96, 2×GAT с residual после обоих слоёв.
    """
    def __init__(self, node_dim=128, hidden_dim=96,
                 num_classes=5, heads=4, dropout=0.15):
        super().__init__()

        A_raw = build_clinical_adj()
        A_norm = normalize_adj(A_raw)
        self.register_buffer('adj', A_norm)

        # GAT слой 1: node_dim → hidden_dim * heads
        self.gat1 = GATLayer(node_dim, hidden_dim,
                             heads=heads, dropout=dropout, concat=True)

        # GAT слой 2: hidden_dim * heads → hidden_dim
        self.gat2 = GATLayer(hidden_dim * heads, hidden_dim,
                             heads=1, dropout=dropout, concat=False)

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Residual projections (выравниваем размерности для skip-connection)
        self.res_proj1 = nn.Linear(node_dim, hidden_dim * heads, bias=False)
        self.res_proj2 = nn.Linear(hidden_dim * heads, hidden_dim, bias=False)

        # Per-class attention readout: каждый класс «смотрит» на нужные leads
        self.class_queries = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.class_queries.unsqueeze(0))

        # Classifier: [B, C, hidden] → [B, C]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z):
        """
        z : [B, 12, node_dim]

        Поток данных:
            z → GAT1 → residual+norm → GAT2 → residual+norm
              → per-class attention readout → classifier → logits
        """
        # GAT слой 1 + residual connection
        h1, alpha1 = self.gat1(z, self.adj)        # [B, 12, hidden*heads]
        h1 = self.norm1(h1 + self.res_proj1(z))

        # GAT слой 2 + residual connection
        h2, alpha2 = self.gat2(h1, self.adj)        # [B, 12, hidden]
        h2 = self.norm2(h2 + self.res_proj2(h1))

        # Per-class attention readout с масштабированием (scaled dot-product)
        scale = h2.shape[-1] ** 0.5
        scores = torch.einsum('cd,bnd->bcn', self.class_queries, h2) / scale
        attn_leads = F.softmax(scores, dim=-1)       # [B, C, 12] — сумма по leads = 1
        pooled = torch.einsum('bcn,bnd->bcd', attn_leads, h2)  # [B, C, hidden]

        logits = self.classifier(pooled).squeeze(-1)  # [B, C]

        return logits, {
            'attn_leads': attn_leads,   # [B, 5, 12] — XAI: важность отведений per class
            'gat1_alpha': alpha1,       # [B, 4, 12, 12] — GAT attention слой 1
            'gat2_alpha': alpha2,       # [B, 1, 12, 12] — GAT attention слой 2
        }


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК H: Полная модель
# ─────────────────────────────────────────────────────────────────────────────

class LeadWiseResNetGNN(nn.Module):
    """
    LeadWiseResNet1d (encoder) → GNNHead

    Encoder : [B, 12, 1000] → [B, 12, embed_dim]
    GNN-head: [B, 12, embed_dim] → logits [B, 5] + attn_info dict
    """
    def __init__(self, embed_dim=128, hidden_dim=96,
                 num_classes=5, dropout=0.15):
        super().__init__()
        self.encoder = LeadWiseResNet1d(embed_dim=embed_dim,
                                        dropout=dropout)
        self.gnn_head = GNNHead(node_dim=embed_dim,
                                hidden_dim=hidden_dim,
                                num_classes=num_classes,
                                heads=4,
                                dropout=dropout)

    def forward(self, x):
        z = self.encoder(x)               # [B, 12, embed_dim]
        logits, attn_info = self.gnn_head(z)  # [B, 5], dict
        return logits, attn_info


# ─────────────────────────────────────────────────────────────────────────────
# Быстрая проверка при прямом запуске: python leadwise_gnn.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    EMBED_DIM = 128

    model = LeadWiseResNetGNN(
        embed_dim=EMBED_DIM,
        hidden_dim=96,
        num_classes=5,
        dropout=0.15,
    )

    x = torch.randn(4, 12, 1000)
    logits, attn = model(x)

    p_enc = sum(p.numel() for p in model.encoder.parameters())
    p_gnn = sum(p.numel() for p in model.gnn_head.parameters())

    print(f'Encoder params : {p_enc:,}')
    print(f'GNN-head params: {p_gnn:,}')
    print(f'Total params   : {p_enc + p_gnn:,}')
    print()
    print(f'logits      : {logits.shape}')
    print(f'attn_leads  : {attn["attn_leads"].shape}')
    print(f'gat1_alpha  : {attn["gat1_alpha"].shape}')
    print(f'gat2_alpha  : {attn["gat2_alpha"].shape}')

    assert logits.shape == (4, 5)
    assert attn['attn_leads'].shape == (4, 5, 12)
    assert attn['gat1_alpha'].shape == (4, 4, 12, 12)
    assert attn['gat2_alpha'].shape == (4, 1, 12, 12)

    attn_sum = attn['attn_leads'].sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)

    adj = model.gnn_head.adj
    assert torch.allclose(adj, adj.T, atol=1e-5)
    assert (adj.diag() > 0).all()

    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    print()
    print('✅ Все проверки пройдены — LeadWiseResNetGNN OK')
