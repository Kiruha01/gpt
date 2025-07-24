import torch
import torch.nn as nn
import torch.nn.functional as F

from core.transformer_block import TransformerBlock

from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 256
    block_size: int = 16
    n_embed: int= 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1


class MiniGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embed)

        self.blocks = nn.Sequential(*[
            TransformerBlock(config.n_embed, config.n_heads) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size)

        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embed(idx)                    # (B, T, C)
        position_ids = torch.arange(T, device=idx.device)          # (T,)
        pos_embeddings = self.pos_embed(position_ids)[None, :, :]  # (1, T, C)
        x = token_embeddings + pos_embeddings                      # (B, T, C)

        x = self.blocks(x)           # (B, T, C)
        x = self.ln_f(x)             # (B, T, C)
        logits = self.head(x)        # (B, T, vocab_size)

        if targets is None:
            return logits

        # reshape for cross_entropy: (B*T, vocab_size) vs (B*T)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
