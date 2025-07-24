import os
import json
import torch

from core.gpt import MiniGPT


def save_model(model, config, path="models/mymodel"):
    os.makedirs(path, exist_ok=True)

    # Сохраняем веса
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))

    # Сохраняем конфиг
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("__")}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"✅ Model saved to {path}")


def load_model(path="models/mymodel"):
    # Загружаем конфиг
    with open(os.path.join(path, "config.json"), "r") as f:
        config_dict = json.load(f)

    # Создаём объект конфигурации
    class LoadedConfig:
        pass
    config = LoadedConfig()
    for k, v in config_dict.items():
        setattr(config, k, v)

    # Создаём модель и загружаем веса
    model = MiniGPT(config)
    model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=torch.device('cpu')))
    model.eval()
    return model, config
