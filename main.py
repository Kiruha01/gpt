import torch
from loguru import logger
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from core.gpt import MiniGPT, Config
from tokenizator import BPEDataset
from core.utils import load_model, save_model


batch_size = 32

tokenizer = Tokenizer.from_file("tokenizer/putin.json")
# tokenizer = Tokenizer.from_file("tokenizer/first.json")
# tokenizer = BPETokenizer("tokenizer/first.json")
file_dataset = "tiny.txt"


device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

temperature = 1

def probe(cfg: Config, model: MiniGPT):
    print(sum(sum(p.size()) for p in model.parameters()))

    if prompt := input("Enter prompt: "):
        context = tokenizer.encode(prompt).ids
    else:
        context = [0]  # Пустой токен (если есть) или просто ноль

    context = torch.tensor(context, dtype=torch.long, device=device)[None, :]  # (1, T)

    # context = torch.zeros((1, 1), dtype=torch.long, device=device)  # токен начала

    model.eval()
    for _ in range(128):
        # Обрезаем до допустимой длины
        idx_cond = context[:, -cfg.block_size:]

        print(idx_cond)

        # Получаем логиты
        logits = model(idx_cond)  # (1, T, vocab_size)
        logits = logits[:, -1, :] / temperature  # последние логиты, shape (1, vocab_size)

        # Преобразуем в вероятности
        probs = F.softmax(logits, dim=-1)  # (1, vocab_size)

        # Выбираем следующий токен
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Добавляем к текущей последовательности
        context = torch.cat([context, next_token], dim=1)

    output_ids = context[0].tolist()
    print(tokenizer.decode(output_ids)
)


@logger.catch
def train(cfg, model):
    model.to(device)
    with open(file_dataset, "r", encoding="utf-8") as f:
        text = f.read()

    dataset = BPEDataset(text, tokenizer, cfg.block_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    try:
        for epoch in range(10):
            model.train()
            total_loss = 0
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                B, T, V = logits.shape
                loss = criterion(logits.view(B * T, V), y.view(B * T))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 100 == 0:
                    print(f"Epoch {epoch + 1}, part {i+1}, loss: {total_loss / len(loader):.4f}")

            print(f"Epoch {epoch + 1}, loss: {total_loss / len(loader):.4f}")
    except KeyboardInterrupt:
        print("interrupted")
    else:
        print("finishing training")
    finally:
        name = input("name: ")
        print(f"saving as {name}")
        save_model(model, cfg, path=f"models/{name}")

if name := input("load name: "):
    model_loaded, cfg = load_model(f"models/{name}")
    print("Loaded model with vocab size:", cfg.vocab_size)

    if training := input("continue train? "):
        train(cfg, model_loaded)

    probe(cfg, model_loaded)


else:
    cfg = Config(
        vocab_size=1000,
        block_size=256,
        n_embed=64,
        n_heads=2,
        n_layers=2,
        dropout=0.2,
    )
    cfg.vocab_size = tokenizer.get_vocab_size()
    model = MiniGPT(cfg).to(device)

    train(cfg, model)
    probe(cfg, model)




