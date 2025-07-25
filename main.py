import torch
from loguru import logger
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from core.gpt import MiniGPT, Config
from core.loaders import create_loader, BaseGPTLoader
from core.utils import load_model, save_model


batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
temperature = 1


@logger.catch
def probe(cfg: Config, model: MiniGPT, loader: BaseGPTLoader):
    prompt = input("Enter prompt: ") or "[BOS]"
    generated = loader.generate(model, prompt, max_tokens=128, temperature=temperature, device=device)
    print("Generated:", generated)


@logger.catch
def train(cfg, model, loader: BaseGPTLoader, dataset_path: str):
    model.to(device)
    dataset = loader.load_dataset(dataset_path)  # или load_chat_dataset для чата
    dataloader = loader.get_dataloader(dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss(ignore_index=loader.tokenizer.encode("[PAD]")[0])  # Игнорируем [PAD] в лоссе

    try:
        for epoch in range(10):
            model.train()
            total_loss = 0
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                pad_mask = (x != loader.pad_token_id).to(device) if loader.pad_token_id is not None else None
                logits, loss = model(x, y, pad_mask=pad_mask)  # Извлекаем logits и loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 100 == 0:
                    print(f"Epoch {epoch + 1}, part {i + 1}, loss: {total_loss / (i + 1):.4f}")

            print(f"Epoch {epoch + 1}, loss: {total_loss / len(dataloader):.4f}")
    except KeyboardInterrupt:
        print("interrupted")
    else:
        print("finishing training")
    finally:
        name = input("name: ")
        print(f"saving as {name}")
        save_model(model, cfg, path=f"models/{name}")


if __name__ == "__main__":
    loader_type = input("Loader type (byte/bpe/chat1/chat2): ") or "chat1"
    tokenizer_path = input("Tokenizer path (if applicable): ") or "tokenizer/bpe_tokenizer.json"
    dataset_path = "datasets/dialog.json"

    loader = create_loader(loader_type, tokenizer_path)
    print(loader)

    if name := input("load name: "):
        model_loaded, cfg = load_model(f"models/{name}")
        print("Loaded model with vocab size:", cfg.vocab_size)

        if input("continue train? ").lower() == "y":
            train(cfg, model_loaded, loader, dataset_path)

        probe(cfg, model_loaded, loader)
    else:
        cfg = Config(
            vocab_size=loader.tokenizer.get_vocab_size(),
            block_size=loader.block_size,
            n_embed=64,
            n_heads=2,
            n_layers=2,
            dropout=0.2,
        )
        print(cfg)
        model = MiniGPT(cfg).to(device)
        train(cfg, model, loader, dataset_path)
        probe(cfg, model, loader)