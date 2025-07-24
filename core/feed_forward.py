import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # расширяем
            nn.ReLU(),                            # нелинейность
            nn.Linear(4 * embed_dim, embed_dim),  # сжимаем обратно
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)
