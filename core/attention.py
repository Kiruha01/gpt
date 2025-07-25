from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, head_size: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("tril", torch.tril(torch.ones(256, 256)))  # Соответствует block_size=125

        # Инициализация весов
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Проверка на nan/inf в q, k, v
        if torch.isnan(q).any() or torch.isinf(q).any():
            print("NaN or Inf in queries:", q)
        if torch.isnan(k).any() or torch.isinf(k).any():
            print("NaN or Inf in keys:", k)

        scores = q @ k.transpose(-2, -1)  # (B, T, T)
        scores = scores / (k.shape[-1] ** 0.5)  # Делим отдельно для отладки
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("Attention scores contain NaN or Inf:", scores)

        # Применяем маски
        mask = self.tril[:T, :T]
        if pad_mask is not None:
            pad_mask = pad_mask[:, :T]  # Обрезаем до T
            scores = scores.masked_fill(pad_mask.unsqueeze(-1) == 0, float("-inf"))

        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Защита от nan в softmax
        weights = F.softmax(scores, dim=-1)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print("Attention weights contain NaN or Inf:", weights)

        weights = self.dropout(weights)
        out = weights @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttention(embed_dim, self.head_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

        # Инициализация весов
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None):
        out = torch.cat([h(x, pad_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


if __name__ == "__main__":
    attn = SelfAttention(embed_dim=32, head_size=32)
    x = torch.randn(2, 5, 32)
    pad_mask = torch.ones(2, 5, dtype=torch.bool)
    out = attn(x, pad_mask)
    print(out.shape)