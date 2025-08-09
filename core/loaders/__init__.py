import json
from glob import iglob
from loguru import logger

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel, CharDelimiterSplit, Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder, WordPiece
from tokenizers.trainers import BpeTrainer
from torch.onnx.symbolic_opset9 import tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Union
from abc import ABC, abstractmethod
from torch.functional import F

from core.gpt import MiniGPT
from tokenizator import ByteTokenizer, BPETokenizer, BaseTokenizer

class BaseGPTDataset(Dataset, ABC):
    def __init__(self, block_size: int, pad_token_id: Optional[int] = None):
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.sequence_lengths = self._calculate_samples_len()

    @abstractmethod
    def _calculate_samples_len(self) -> List[int]:
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    def __len__(self):
        return sum(self.sequence_lengths)
        return sum(max(1, length - self.block_size + 1) for length in self.sequence_lengths)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def _pad_sequence(self, sequence: List[int], target_length: Optional[int] = None) -> List[int]:
        target_length = target_length or self.block_size
        if len(sequence) < target_length and self.pad_token_id is not None:
            return [self.pad_token_id] * (target_length - len(sequence)) + sequence
        return sequence[-target_length:]

class BaseGPTLoader(ABC):
    def __init__(self):
        self.tokenizer: BaseTokenizer = self._initialize_tokenizer()
        self.block_size = self._set_block_size()
        self.special_tokens = self._set_special_tokens()
        self.stop_token = self._set_stop_token()
        self.pad_token_id = self.tokenizer.encode("[PAD]")[0] if "[PAD]" in self.special_tokens else None
        self.stop_token_id = self.tokenizer.encode(self.stop_token)[0] if self.stop_token else None

    @abstractmethod
    def _initialize_tokenizer(self):
        pass

    @abstractmethod
    def _set_block_size(self) -> int:
        pass

    @abstractmethod
    def _set_special_tokens(self) -> List[str]:
        pass

    @abstractmethod
    def _set_stop_token(self) -> Optional[str]:
        pass

    def load_dataset(self, file_path: str) -> BaseGPTDataset:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f) if file_path.endswith('.json') else f.read()
        return self._create_dataset(data)

    @abstractmethod
    def _create_dataset(self, data: Union[str, List[dict]]) -> BaseGPTDataset:
        pass

    def get_dataloader(self, dataset: BaseGPTDataset, batch_size: int = 4, shuffle: bool = True) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _pad_sequence(self, sequence: List[int], target_length: Optional[int] = None) -> List[int]:
        target_length = target_length or self.block_size
        if len(sequence) < target_length and self.pad_token_id is not None:
            return [self.pad_token_id] * (target_length - len(sequence)) + sequence
        return sequence[-target_length:]

    def generate(self, model: MiniGPT, prompt: str, max_tokens: int = 128, temperature: float = 1.0, device: str = "cpu") -> str:
        model.eval()
        context = self.tokenizer.encode(prompt)
        context = self._pad_sequence(context)
        context = torch.tensor([context], dtype=torch.long, device=device)
        new_tokens = torch.tensor([], dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(self.block_size):
                idx_cond = context[:, -self.block_size:]
                pad_mask = (idx_cond != self.pad_token_id).to(device) if self.pad_token_id is not None else None
                logits = model(idx_cond, pad_mask=pad_mask)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("Logits contain NaN or Inf:", logits)
                    return "Generation failed due to invalid logits"
                temperature = max(temperature, 1e-6)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print("Probs contain NaN or Inf:", probs)
                    return "Generation failed due to invalid probs"
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat([context, next_token], dim=1)
                if self.stop_token_id and next_token.item() == self.stop_token_id:
                    break

                new_tokens = torch.cat([new_tokens, next_token], dim=1)
        if new_tokens.tolist():
            logger.debug(new_tokens[0].tolist())
            return self.tokenizer.decode(new_tokens[0].tolist())
        else:
            ""

class ByteLoader(BaseGPTLoader):
    def _initialize_tokenizer(self):
        return ByteTokenizer()

    def _set_block_size(self) -> int:
        return 125

    def _set_special_tokens(self) -> List[str]:
        return []

    def _set_stop_token(self) -> Optional[str]:
        return None

    def _create_dataset(self, data: Union[str, List[dict]]) -> BaseGPTDataset:
        class ByteDataset(BaseGPTDataset):
            def __init__(self, tokenizer: ByteTokenizer, data: str, block_size: int):
                self.data = data
                self.tokenizer = tokenizer
                super().__init__(block_size)

            def _calculate_samples_len(self) -> List[int]:
                if not isinstance(self.data, str):
                    raise ValueError("ByteDataset requires a string input")
                return [len(self.tokenizer.encode(self.data))]

            def encode(self, text: str) -> List[int]:
                return self.tokenizer.encode(text)

            def decode(self, tokens: List[int]) -> str:
                return self.tokenizer.decode(tokens)

            def __getitem__(self, idx):
                ids = self.tokenizer.encode(self.data)
                start = idx
                sequence = ids[start:start + self.block_size]
                x = self._pad_sequence(sequence)
                y_sequence = ids[start + 1:start + self.block_size + 1]
                y = self._pad_sequence(y_sequence)
                if len(y) < self.block_size:
                    y.extend([-1] * (self.block_size - len(y)))
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

        return ByteDataset(ByteTokenizer(), data, self.block_size)

class BPELoader(BaseGPTLoader):
    def __init__(self, tokenizer_path: str = "tokenizer/bpe_tokenizer.json"):
        self.tokenizer_path = tokenizer_path
        super().__init__()

    @classmethod
    def train_tokenizer(cls, files: list[str], save_path="bpe_tokenizer.json", vocab_size=5000):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = WordPiece()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=cls._set_special_tokens(),
        )

        tokenizer.train(files=files, trainer=trainer)
        tokenizer.save(f"{save_path}")
        print(f"✅ Tokenizer saved to {save_path}")

    def _initialize_tokenizer(self):
        return BPETokenizer(self.tokenizer_path)

    def _set_block_size(self) -> int:
        return 125

    @classmethod
    def _set_special_tokens(cls) -> List[str]:
        return ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[USER]", "[BOT]", "[attachment]"]

    def _set_stop_token(self) -> Optional[str]:
        return "[EOS]"

    def _create_dataset(self, data: Union[str, List[dict]]) -> BaseGPTDataset:
        class BPEDataset(BaseGPTDataset):
            def __init__(self, data: str, block_size: int, tokenizer, pad_token_id):
                self.tokenizer = tokenizer
                self.data = data
                super().__init__(block_size, pad_token_id)

            def _calculate_samples_len(self) -> List[int]:
                if not isinstance(self.data, str):
                    raise ValueError("BPEDataset requires a string input")
                return [len(self.tokenizer.encode(self.data))]

            def encode(self, text: str) -> List[int]:
                return self.tokenizer.encode(text)

            def decode(self, tokens: List[int]) -> str:
                return self.tokenizer.decode(tokens)

            def __getitem__(self, idx):
                ids = self.tokenizer.encode(self.data)
                start = idx
                sequence = ids[start:start + self.block_size]
                x = self._pad_sequence(sequence)
                y_sequence = ids[start + 1:start + self.block_size + 1]
                y = self._pad_sequence(y_sequence)
                if len(y) < self.block_size:
                    y.extend([-1] * (self.block_size - len(y)))
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

        return BPEDataset(data, self.block_size, self.tokenizer, self.pad_token_id)

class ChatLoader1(BaseGPTLoader):
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        super().__init__()

    def _initialize_tokenizer(self):
        return BPETokenizer(self.tokenizer_path)

    def _set_block_size(self) -> int:
        return 128

    def _set_special_tokens(self) -> List[str]:
        return ["[PAD]", "[USER]", "[BOT]", "[EOS]", "[UNK]", "[attachment]"]

    def _set_stop_token(self) -> Optional[str]:
        return "[EOS]"

    def load_dataset(self, file_path: str) -> BaseGPTDataset:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._create_dataset(data)

    @classmethod
    def create_text_data(cls, data: List[dict]) -> str:
        seq = []
        for msg in data:
            role = msg["role"].lower()
            content = msg["content"]
            token = "[USER]" if role == "user" else "[BOT]"
            seq.append(f"{token} {content}")
        return "\n".join(seq)

    def _create_dataset(self, data: Union[str, List[dict]]) -> BaseGPTDataset:
        class ChatDataset(BaseGPTDataset):
            def __init__(self, data: List[dict], block_size: int, tokenizer, special_tokens, pad_token_id):
                self.data = data
                self.tokenizer = tokenizer
                self.user_token = special_tokens[1]
                self.bot_token = special_tokens[2]
                self.eos_token = special_tokens[3]
                super().__init__(block_size, pad_token_id)

            def _calculate_samples_len(self) -> List[int]:
                if not isinstance(self.data, list):
                    raise ValueError("ChatDataset requires a list of messages")

                # Формируем последовательности, заканчивающиеся [EOS]
                lens = 0
                current_sequence = []
                for msg in self.data:
                    role = msg["role"].lower()
                    content = msg["content"]
                    token = self.user_token if role == "user" else self.bot_token

                    if role == "user":
                        encoded = self.encode(f"{token}{content}\n{self.bot_token}")
                        current_sequence.extend(encoded)
                        continue

                    if role == "bot":
                        encoded = self.encode(content + self.eos_token)
                        lens += len(encoded)
                        current_sequence.extend(encoded)

                # Сохраняем длины последовательностей
                return [lens]

            def encode(self, text: str) -> List[int]:
                return self.tokenizer.encode(text)

            def decode(self, tokens: List[int]) -> str:
                return self.tokenizer.decode(tokens)

            def _make_xy(self, sequence: list[int], last_token: int):
                x = torch.tensor(self._pad_sequence(sequence[-self.block_size:]), dtype=torch.long)
                y = torch.tensor(self._pad_sequence(sequence[-self.block_size + 1:] + [last_token]), dtype=torch.long)
                return x, y

            def __getitem__(self, idx):
                # Находим последовательность и позицию внутри неё
                current_sequence = []

                for msg in self.data:
                    role = msg["role"].lower()
                    content = msg["content"]
                    token = self.user_token if role == "user" else self.bot_token

                    if role == "user":
                        encoded = self.encode(f"{token} {content} {self.bot_token}")
                        current_sequence.extend(encoded)
                        continue

                    if role == "bot":
                        encoded = self.encode(content + self.eos_token)
                        for t in encoded:
                            if idx == 0:
                                return self._make_xy(current_sequence, t)
                            current_sequence.append(t)
                            idx -= 1

                    # idx -= 1


                return self._make_xy(current_sequence, self.encode(self.eos_token)[0])
                #
                #
                # # Формируем последовательность заново
                # current_sequence = []
                # for msg in self.data[:sequence_idx + 1]:
                #     role = msg["role"].lower()
                #     content = msg["content"]
                #     token = self.user_token if role == "user" else self.bot_token
                #     encoded = self.encode(f"{token}{content}")
                #     current_sequence.extend(encoded)
                #     if role == "bot":
                #         current_sequence.extend(self.encode(self.eos_token))
                #         if sequence_idx == len(self.data) - 1 or self.data[sequence_idx + 1]["role"].lower() == "user":
                #             break
                #
                # # Выбираем подпоследовательность
                # local_idx = idx - cumulative_length
                # start = local_idx
                # x = torch.tensor(current_sequence[start:start + self.block_size], dtype=torch.long)
                # y = torch.tensor(current_sequence[start + 1:start + self.block_size + 1], dtype=torch.long)
                # return x, y

        return ChatDataset(data, self.block_size, self.tokenizer, self.special_tokens, self.pad_token_id)


class ChatLoader2(BaseGPTLoader):
    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        super().__init__()

    def _initialize_tokenizer(self):
        return BPETokenizer(self.tokenizer_path)

    def _set_block_size(self) -> int:
        return 125

    def _set_special_tokens(self) -> List[str]:
        return ["[PAD]", "[USER]", "[BOT]", "[EOS]", "[UNK]"]

    def _set_stop_token(self) -> Optional[str]:
        return "[EOS]"

    def load_dataset(self, file_path: str) -> BaseGPTDataset:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._create_dataset(data)

    def _create_dataset(self, data: Union[str, List[dict]]) -> BaseGPTDataset:
        class ChatDataset(BaseGPTDataset):
            def __init__(self, data: List[dict], block_size: int, tokenizer, special_tokens, pad_token_id):
                self.data = data
                self.tokenizer = tokenizer
                self.user_token = special_tokens[1]
                self.bot_token = special_tokens[2]
                self.eos_token = special_tokens[3]
                super().__init__(block_size, pad_token_id)

            def _calculate_samples_len(self) -> List[int]:
                if not isinstance(self.data, list):
                    raise ValueError("ChatDataset requires a list of messages")
                sequences = []
                for msg in self.data:
                    if msg["role"].lower() != "bot":
                        continue
                    content = msg["content"]
                    encoded = self.encode(f"{self.bot_token}{content}{self.eos_token}")
                    sequences.append(encoded)
                return [len(seq) for seq in sequences]

            def encode(self, text: str) -> List[int]:
                return self.tokenizer.encode(text)

            def decode(self, tokens: List[int]) -> str:
                return self.tokenizer.decode(tokens)

            def __getitem__(self, idx):
                sequence_idx = 0
                cumulative_length = 0
                for length in self.sequence_lengths:
                    if idx < cumulative_length + max(1, length - self.block_size + 1):
                        break
                    cumulative_length += max(1, length - self.block_size + 1)
                    sequence_idx += 1

                bot_msgs = [msg for msg in self.data if msg["role"].lower() == "bot"]
                if sequence_idx >= len(bot_msgs):
                    raise IndexError("Invalid sequence index")
                content = bot_msgs[sequence_idx]["content"]
                current_sequence = self.encode(f"{self.bot_token}{content}{self.eos_token}")

                local_idx = idx - cumulative_length
                start = local_idx
                sequence = current_sequence[start:start + self.block_size]
                x = self._pad_sequence(sequence)
                y_sequence = current_sequence[start + 1:start + self.block_size + 1]
                y = self._pad_sequence(y_sequence)
                if len(y) < self.block_size:
                    y.extend([-1] * (self.block_size - len(y)))
                return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

        return ChatDataset(data, self.block_size, self.tokenizer, self.special_tokens, self.pad_token_id)

def create_loader(loader_type: str, tokenizer_path: Optional[str] = None) -> BaseGPTLoader:
    loaders = {
        "byte": ByteLoader,
        "bpe": lambda: BPELoader(tokenizer_path),
        "chat1": lambda: ChatLoader1(tokenizer_path),
        "chat2": lambda: ChatLoader2(tokenizer_path),
    }
    if loader_type not in loaders:
        raise ValueError(f"Unsupported loader_type: {loader_type}")
    return loaders[loader_type]()