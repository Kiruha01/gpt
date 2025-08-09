import os.path
import time
from multiprocessing.managers import Value

import torch
from loguru import logger
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime

from core.gpt import MiniGPT, Config
from core.loaders import create_loader, BaseGPTLoader
from core.utils import load_model, save_model

import sys

logger.remove()
logger.add(sys.stdout, level="DEBUG")
logger.add("app.log", rotation="10 MB", retention="10 days", level="DEBUG")

batch_size = 64  # Увеличен для стабильности
device = "cuda" if torch.cuda.is_available() else "cpu"
temperature = 1.0

@logger.catch
def probe(cfg: Config, model: MiniGPT, loader: BaseGPTLoader):
    context = ""
    while prompt := input("Enter prompt: "):
        context += f" [EOS] [USER] {prompt} [BOT]"
        # print(context)
        # print("Encoded prompt:", loader.tokenizer.encode(prompt))
        generated = loader.generate(model, context, max_tokens=128, temperature=temperature, device=device)

        messages = generated.split("[BOT]")

        for m in messages:
            print(">>", m)
            # time.sleep(0.5)

        context += f" {generated}"

@logger.catch
def train(cfg, model, loader: BaseGPTLoader, dataset_path: str):
    model.to(device)
    dataset = loader.load_dataset(dataset_path)
    # for i in range (100):
    #     x, y = dataset[i]
    #     print(i, "x=", loader.tokenizer.decode(x.tolist()))
    #     print("y=", loader.tokenizer.decode(y.tolist()), "\n")

    # input()

    print(len(dataset))
    dataloader = loader.get_dataloader(dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    try:
        for epoch in range(1000):
            model.train()
            total_loss = 0
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                pad_mask = (x != loader.pad_token_id).to(device) if loader.pad_token_id is not None else None
                # print("x shape:", x.shape, "x[0]:", x[0].cpu().tolist())
                # print("y shape:", y.shape, "y[0]:", y[0].cpu().tolist())
                # print("pad_mask[0]:", pad_mask[0].cpu().tolist() if pad_mask is not None else None)
                logits, loss = model(x, y, pad_mask=pad_mask)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                if i % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}, part {i + 1}, loss: {loss.item():.4f}")

            logger.info(f"Epoch {epoch + 1}, loss: {total_loss / len(dataloader):.4f}")
    except KeyboardInterrupt:
        print("interrupted")
    else:
        print("finishing training")
    finally:
        name = input("name: ")
        print(f"Saving as {name}")
        save_model(model, cfg, path=f"models/{name}")

if __name__ == "__main__":
    loader_type = input("Loader type (byte/bpe/chat1/chat2): ") or "chat1"
    tokenizer_path = input("Tokenizer path (if applicable): ") or "models/val/bpe_tokenizer.json"
    dataset_path = input("Dataset path: ") or "datasets/val.json"


    if name := input("load name: "):
        if os.path.exists(f"{name}/bpe_tokenizer.json"):
            tokenizer_path = f"{name}/bpe_tokenizer.json"

        logger.debug(f"load bpe from {tokenizer_path}")
        loader = create_loader(loader_type, tokenizer_path)
        logger.debug(f"Loader: {loader}")

        model_loaded, cfg = load_model(f"models/{name}")
        logger.debug(f"Loaded model with vocab size: {cfg.vocab_size}")
        cfg.pad_token_id = loader.pad_token_id or 0
        if input("continue train? ").lower() == "y":
            train(cfg, model_loaded, loader, dataset_path)
        probe(cfg, model_loaded, loader)
    else:
        if not tokenizer_path:
            raise ValueError("Tokenizer path")

        loader = create_loader(loader_type, tokenizer_path)
        logger.debug(f"Loader: {loader}")

        cfg = Config(
            vocab_size=loader.tokenizer.get_vocab_size(),
            block_size=loader.block_size,
            n_embed=128,
            n_heads=4,
            n_layers=4,
            dropout=0.1,
            pad_token_id=loader.pad_token_id or 0
        )
        logger.debug(f"Config: {cfg}")
        model = MiniGPT(cfg).to(device)
        train(cfg, model, loader, dataset_path)
        probe(cfg, model, loader)