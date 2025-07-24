from random import triangular

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC


def train_tokenizer(files: list[str], save_path="bpe_tokenizer.json"):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=5000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[USER]", "[BOT]"]
    )

    tokenizer.train(files=files, trainer=trainer)
    tokenizer.save(f"tokenizer/{save_path}")
    print(f"✅ Tokenizer saved to tokenizer/{save_path}")


if __name__ == "__main__":
    FILES = [
        "tiny.txt"
    ]
    train_tokenizer(FILES, input("tok. name: "))