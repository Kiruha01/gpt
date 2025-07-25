import json
import tempfile

from core.loaders import BPELoader, ChatLoader1

with tempfile.NamedTemporaryFile() as f:
    data = json.load(
                open("datasets/dialog.json"))
    print(data)
    t = ChatLoader1.create_text_data(data).encode("utf-8")
    print(t)
    f.write(t)
    f.seek(0)
    BPELoader.train_tokenizer([f.name])