import torch
import torch.nn as nn
import torch.nn.functional as F
from core.transformer_block import TransformerBlock
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 256
    block_size: int = 125
    n_embed: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    pad_token_id: int = 0

class MiniGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.n_embed, config.n_heads) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size)
        self.block_size = config.block_size
        self.pad_token_id = config.pad_token_id

        # Инициализация весов
        self.apply(self._init_weights)
        # Специальная инициализация для [PAD]
        with torch.no_grad():
            self.token_embed.weight[config.pad_token_id].fill_(0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, pad_mask=None):
        B, T = idx.shape
        token_embeddings = self.token_embed(idx)  # (B, T, C)
        position_ids = torch.arange(T, device=idx.device)
        pos_embeddings = self.pos_embed(position_ids)[None, :]  # (1, T, C)
        x = token_embeddings + pos_embeddings

        if torch.isnan(x).any():
            print("NaN in embeddings:", x)

        for block in self.blocks:
            x = block(x, pad_mask)
            if torch.isnan(x).any():
                print("NaN after block:", x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        if torch.isnan(logits).any():
            print("NaN in logits:", logits)

        if targets is None:
            return logits

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T), ignore_index=self.pad_token_id)
        return logits, loss