import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode('utf-8'))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode('utf-8', errors='replace')


class BPETokenizer:
    def __init__(self, file):
        self.tokenizer = Tokenizer.from_file(file)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class GPTDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


class BPEDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        # Кодируем весь текст целиком
        self.ids = tokenizer.encode(text).ids

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y
