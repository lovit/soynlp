import os
import tempfile

import pytest

from soynlp.pipeline.tasks.write_text import WriteTextTask, WriteTextTaskArgs


@pytest.fixture
def examples():
    return [
        {"text": "안녕하세요"},
        {"text": "soynlp 입니다"},
    ]


class TestWriteTextTask:
    def test_write_text(self, examples):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.txt")
            task = WriteTextTask(WriteTextTaskArgs(path=path, in_key="corpus"))
            result = task({"corpus": examples})
            assert "corpus" in result
            with open(path, encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
            assert lines == ["안녕하세요", "soynlp 입니다"]

    def test_file_exists_error(self, examples):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.txt")
            with open(path, "w") as f:
                f.write("existing\n")
            task = WriteTextTask(WriteTextTaskArgs(path=path, in_key="corpus"))
            with pytest.raises(FileExistsError):
                task({"corpus": examples})

    def test_missing_key_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output.txt")
            task = WriteTextTask(WriteTextTaskArgs(path=path, in_key="corpus"))
            with pytest.raises(ValueError, match="Not found"):
                task({})
