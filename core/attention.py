import torch
import torch.nn as nn
import torch.nn.functional as F


# Создаём собственный модуль внимания, унаследованный от базового nn.Module
class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, head_size: int):
        super().__init__()  # Обязательный вызов конструктора родителя (nn.Module)

        # Линейные слои, которые создают Query, Key и Value из входных векторов.
        # Эти слои обучаются. Они принимают вектор размером embed_dim и выдают head_size.
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        # Dropout — способ "отключать" часть нейронов во время обучения, чтобы избежать переобучения.
        self.dropout = nn.Dropout(0.1)

        # Регистрируем маску в виде нижнетреугольной матрицы (размером 1024x1024).
        # Она нужна, чтобы при обучении GPT токен не "видел будущее".
        self.register_buffer("tril", torch.tril(torch.ones(1024, 1024)))

    def forward(self, x):
        # x — входной тензор размера (batch_size, seq_len, embed_dim)
        B, T, C = x.shape  # Распаковываем размеры батча, длины последовательности и каналов

        # Применяем обучаемые линейные слои для получения Q, K и V
        q = self.query(x)  # (B, T, head_size) — запросы
        k = self.key(x)    # (B, T, head_size) — ключи
        v = self.value(x)  # (B, T, head_size) — значения

        # Считаем attention scores:
        # Матрица произведения Q и транспонированной K по последним осям
        # Делим на sqrt(d_k) для стабилизации значений
        scores = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # (B, T, T)

        # Создаём маску, чтобы запрещать токенам "смотреть вперёд"
        # tril[:T, :T] — обрезаем маску до нужной длины (если, например, T = 5 → 5x5 маска)
        mask = self.tril[:T, :T]

        # Заменяем все "будущие" значения в attention на -бесконечность
        # После softmax они станут нулями
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Применяем softmax — превращаем очки внимания в вероятности
        weights = F.softmax(scores, dim=-1)  # (B, T, T)
        print(weights.shape)

        # Dropout для стабилизации обучения
        weights = self.dropout(weights)

        # Перемножаем веса на значения — получаем взвешенную сумму "смыслов"
        out = weights @ v  # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0  # чтобы голова влезла

        self.head_size = embed_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttention(embed_dim, self.head_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(embed_dim, embed_dim)  # финальное объединение
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # применяем каждую голову внимания
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # объединяем головы
        out = self.dropout(self.proj(out))  # проецируем обратно в embed_dim
        return out


if __name__ == "__main__":
    attn = SelfAttention(embed_dim=32, head_size=32)
    x = torch.randn(2, 5, 32)  # batch=2, seq_len=5, embedding_dim=32
    out = attn(x)
    print(out.shape)  # ожидаем: (2, 5, 32)

    print()
    print(x)
    print(out)
