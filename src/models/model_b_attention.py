import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from src.models.encoder_cnn import CNNEncoder


class ClassifierHead(nn.Module):
    """
    Линейная голова (2-слойный MLP) для классификации, 
    идентичная той, что используется в Baseline для честного сравнения.
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
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LeadAttention(nn.Module):
    """
    Attention-механизм по оси отведений (leads).
    Вычисляет важность каждого из 12 отведений на основе его признаков
    и взвешивает их для получения итогового эмбеддинга.
    """
    def __init__(self, feature_dim: int = 256, reduction: int = 4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.Tanh(),
            nn.Linear(feature_dim // reduction, 1)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: (B, 12, feature_dim)
        
        # Вычисляем скоры для каждого отведения
        attn_scores = self.attention(x)  # (B, 12, 1)
        
        # Нормализуем по оси отведений (leads)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, 12, 1)
        
        # Взвешенная сумма эмбеддингов отведений
        pooled = torch.sum(x * attn_weights, dim=1)  # (B, feature_dim)
        
        return pooled, attn_weights.squeeze(-1)


class ModelBAttention(nn.Module):
    """
    Model B - CNN + Channel Attention по leads
    Общий CNN-энкодер -> Attention по отведениям -> Классификатор.
    """
    def __init__(
        self, 
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 5,
        encoder_dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Общий базовый энкодер
        self.encoder = CNNEncoder(
            feature_dim=feature_dim,
            dropout=encoder_dropout
        )
        
        # 2. Модуль внимания по отведениями (наша новизна)
        self.attention = LeadAttention(
            feature_dim=feature_dim, 
            reduction=8
        )
        
        # 3. Голова классификации (как в Model A)
        self.head = ClassifierHead(
            in_dim=feature_dim,
            hidden=hidden_dim,
            num_cls=num_classes
        )
        
    def forward(self, x: Tensor, return_attention: bool = False):
        """
        Args:
            x : (B, 12, 1000) — нормализованные 12-отведённые ЭКГ
            return_attention : флаг для возврата весов внимания (для интерпретируемости)

        Returns:
            logits : (B, 5) — сырые логиты
            (опционально) attn_weights : (B, 12) — веса важности отведений
        """
        # Извлекаем признаки (каждое отведение независимо)
        features = self.encoder(x)  # (B, 12, 256)
        
        # Взвешиваем отведения через Attention
        pooled_features, attn_weights = self.attention(features)  # (B, 256), (B, 12)
        
        # Классификация
        logits = self.head(pooled_features)  # (B, 5)
        
        if return_attention:
            return logits, attn_weights
            
        return logits

# Для быстрой проверки
if __name__ == "__main__":
    model = ModelBAttention()
    dummy_x = torch.randn(8, 12, 1000)  # batch=8
    logits, weights = model(dummy_x, return_attention=True)
    
    print(f"Input shape: {dummy_x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention sum per sample: {weights.sum(dim=1).detach().numpy()}")

