import json
import os
import tempfile

import pytest

from soynlp.pipeline.tasks.read_json import ReadJsonTask, ReadJsonTaskArgs


def test_read_json(corpus_path: str):
    assert os.path.exists(corpus_path)
    task = ReadJsonTask(ReadJsonTaskArgs(path=corpus_path))
    corpus = task({})["corpus"]
    assert corpus[0] == {"text": "안녕하세요", "label": True}
    assert corpus[1] == {"text": "soynlp 입니다", "label": True}


@pytest.fixture
def corpus_path():
    examples = [
        {"text": "안녕하세요", "label": True},
        {"text": "soynlp 입니다", "label": True},
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        corpus_path = os.path.join(temp_dir, "data.jsonl")
        with open(corpus_path, "w", encoding="utf-8") as file:
            for example in examples:
                file.write(f"{json.dumps(example, ensure_ascii=False)}\n")
        yield corpus_path
