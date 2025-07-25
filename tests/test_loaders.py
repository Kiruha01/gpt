from pathlib import Path

import pytest

from loaders import BPELoader, ByteLoader, ChatLoader1



def test_byte():
    Path("dt.txt").write_text("""Раз два три четыре""")

    # ByteLoader.trainen_tokenizer(["dt.txt"])

    loader = ByteLoader()

    ds = loader._create_dataset("""Раз два три четыре""")
    print(ds[0][0])

    assert (loader.tokenizer.decode(ds[0][0]), loader.tokenizer.decode(ds[0][1])) == (
        "Раз д�",
        "�аз дв"
    )
    assert (loader.tokenizer.decode(ds[1][0]), loader.tokenizer.decode(ds[1][1])) == (
        "�аз дв",
        "аз дв�"
    )

@pytest.mark.WIP
def test_chat():
    data = [
        {"role": "user", "content": "HI"},
        {"role": "bot", "content": "Hi, bro"},
        {"role": "user", "content": "Go"},
        {"role": "bot", "content": "GO"},
    ]

    Path("dt.txt").write_text(ChatLoader1.create_text_data(data))

    BPELoader.train_tokenizer(["dt.txt"])

    loader = ChatLoader1("bpe_tokenizer.json")

    ds = loader._create_dataset(data)

    assert (loader.tokenizer.decode(ds[0][0]), loader.tokenizer.decode(ds[0][1])) == (
        "[USER] HI\n[BOT]",
        "[USER] HI\n[BOT] "
    )
    assert (loader.tokenizer.decode(ds[1][0]), loader.tokenizer.decode(ds[1][1])) == (
        "[USER] HI\n[BOT] ",
        "[USER] HI\n[BOT] Hi"
    )
    assert (loader.tokenizer.decode(ds[2][0]), loader.tokenizer.decode(ds[2][1])) == (
        "[USER] HI\n[BOT] Hi",
        "[USER] HI\n[BOT] Hi,"
    )
    assert (loader.tokenizer.decode(ds[3][0]), loader.tokenizer.decode(ds[3][1])) == (
        "[USER] HI\n[BOT] Hi,",
        "[USER] HI\n[BOT] Hi, bro"
    )
    assert (loader.tokenizer.decode(ds[4][0]), loader.tokenizer.decode(ds[4][1])) == (
        "[USER] HI\n[BOT] Hi, bro",
        "[USER] HI\n[BOT] Hi, bro[EOS]",
    )
    assert (loader.tokenizer.decode(ds[5][0]), loader.tokenizer.decode(ds[5][1])) == (
        "[USER] HI\n[BOT] Hi, bro[USER] Go\n[BOT]",
        "[USER] HI\n[BOT] Hi, bro[USER] Go\n[BOT] ",
    )
    assert (loader.tokenizer.decode(ds[6][0]), loader.tokenizer.decode(ds[6][1])) == (
        "[USER] HI\n[BOT] Hi, bro[USER] Go\n[BOT] ",
        "[USER] HI\n[BOT] Hi, bro[USER] Go\n[BOT] GO",
    )
    assert (loader.tokenizer.decode(ds[7][0]), loader.tokenizer.decode(ds[7][1])) == (
        "[USER] HI\n[BOT] Hi, bro[USER] Go\n[BOT] GO",
        "[USER] HI\n[BOT] Hi, bro[USER] Go\n[BOT] GO[EOS]",
    )


