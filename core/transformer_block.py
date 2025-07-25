import torch.nn as nn

from core.attention import MultiHeadAttention
from core.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)  # нормализуем → attention → residual
        x = x + self.ff(self.ln2(x))   # нормализуем → FF → residual
        return x
