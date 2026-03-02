import os
import tempfile

import pytest

from soynlp.pipeline.tasks.read_text import ReadTextTask, ReadTextTaskArgs


def test_read_text(corpus_path: str):
    assert os.path.exists(corpus_path)
    task = ReadTextTask(ReadTextTaskArgs(path=corpus_path, text_key="text"))
    corpus = task({})["corpus"]
    assert corpus[0] == {"text": "안녕하세요"}
    assert corpus[1] == {"text": "soynlp 입니다"}


@pytest.fixture
def corpus_path():
    texts = [
        "안녕하세요",
        "soynlp 입니다",
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        corpus_path = os.path.join(temp_dir, "data.jsonl")
        with open(corpus_path, "w", encoding="utf-8") as file:
            for text in texts:
                file.write(f"{text}\n")
        yield corpus_path
