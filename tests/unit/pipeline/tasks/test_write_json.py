import json
import os
import tempfile

import pytest

from soynlp.pipeline.tasks.write_json import WriteJsonTask, WriteJsonTaskArgs


@pytest.fixture
def examples():
    return [
        {"text": "안녕하세요", "label": True},
        {"text": "soynlp 입니다", "label": True},
    ]


class TestWriteJsonTask:
    def test_write_json(self, examples):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.jsonl")
            task = WriteJsonTask(WriteJsonTaskArgs(path=path, in_key="corpus"))
            result = task({"corpus": examples})
            assert "corpus" in result
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0]) == examples[0]

    def test_file_exists_error(self, examples):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.jsonl")
            with open(path, "w") as f:
                f.write("existing\n")
            task = WriteJsonTask(WriteJsonTaskArgs(path=path, in_key="corpus"))
            with pytest.raises(FileExistsError):
                task({"corpus": examples})

    def test_missing_key_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.jsonl")
            task = WriteJsonTask(WriteJsonTaskArgs(path=path, in_key="corpus"))
            with pytest.raises(ValueError, match="Not found"):
                task({})
