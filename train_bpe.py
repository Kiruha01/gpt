import json
import tempfile

from core.loaders import BPELoader, ChatLoader1

with open("temp.txt", "w") as f:
    data = json.load(open("datasets/alina.json", encoding="utf-8"))
    # print(data)
    t = ChatLoader1.create_text_data(data)
    print(t)
    f.write(t)

BPELoader.train_tokenizer(["temp.txt"])