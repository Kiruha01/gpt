from pathlib import Path

import pytest

from core.loaders import BPELoader, ByteLoader, ChatLoader1



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
    loader.block_size = 10

    ds = loader._create_dataset(data)

    assert (loader.tokenizer.decode(ds[0][0]), loader.tokenizer.decode(ds[0][1])) == (
        "[PAD][PAD][PAD][PAD][PAD][USER] HI\n[BOT]",
        "[PAD][PAD][PAD][PAD][PAD][USER] HI\n[BOT] "
    )
    assert (loader.tokenizer.decode(ds[1][0]), loader.tokenizer.decode(ds[1][1])) == (
        "[PAD][PAD][PAD][PAD][USER] HI\n[BOT] ",
        "[PAD][PAD][PAD][PAD][USER] HI\n[BOT] Hi"
    )
    assert (loader.tokenizer.decode(ds[2][0]), loader.tokenizer.decode(ds[2][1])) == (
        "[PAD][PAD][PAD][USER] HI\n[BOT] Hi",
        "[PAD][PAD][PAD][USER] HI\n[BOT] Hi,"
    )
    assert (loader.tokenizer.decode(ds[3][0]), loader.tokenizer.decode(ds[3][1])) == (
        "[PAD][PAD][USER] HI\n[BOT] Hi,",
        "[PAD][PAD][USER] HI\n[BOT] Hi, bro"
    )
    assert (loader.tokenizer.decode(ds[4][0]), loader.tokenizer.decode(ds[4][1])) == (
        "[PAD][USER] HI\n[BOT] Hi, bro",
        "[PAD][USER] HI\n[BOT] Hi, bro[EOS]",
    )
    assert (loader.tokenizer.decode(ds[5][0]), loader.tokenizer.decode(ds[5][1])) == (
        "[BOT] Hi, bro[USER] Go\n[BOT]",
        "[PAD] Hi, bro[USER] Go\n[BOT] ",
    )
    assert (loader.tokenizer.decode(ds[6][0]), loader.tokenizer.decode(ds[6][1])) == (
        " Hi, bro[USER] Go\n[BOT] ",
        "[PAD]Hi, bro[USER] Go\n[BOT] GO",
    )
    assert (loader.tokenizer.decode(ds[7][0]), loader.tokenizer.decode(ds[7][1])) == (
        "Hi, bro[USER] Go\n[BOT] GO",
        "[PAD], bro[USER] Go\n[BOT] GO[EOS]",
    )


