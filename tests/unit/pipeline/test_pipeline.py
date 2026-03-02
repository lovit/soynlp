import os
import tempfile

from soynlp.configs.config import from_dict
from soynlp.pipeline.pipeline import Pipeline


class TestPipeline:
    def test_dummy_pipeline(self):
        config = from_dict(
            {
                "pipeline": [
                    {"name": "Dummy", "args": {"name": "test"}},
                ]
            }
        )
        pipeline = Pipeline()
        result = pipeline(config)
        assert isinstance(result, dict)

    def test_read_write_text_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            output_path = os.path.join(tmpdir, "output.txt")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("안녕하세요\n")
                f.write("반갑습니다\n")

            config = from_dict(
                {
                    "pipeline": [
                        {"name": "ReadText", "args": {"path": input_path}},
                        {"name": "WriteText", "args": {"path": output_path}},
                    ]
                }
            )
            pipeline = Pipeline()
            pipeline(config)

            assert os.path.exists(output_path)
            with open(output_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
            assert lines == ["안녕하세요", "반갑습니다"]

    def test_read_normalize_write_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.txt")
            output_path = os.path.join(tmpdir, "output.txt")
            with open(input_path, "w", encoding="utf-8") as f:
                f.write("안녕하세요ㅋㅋㅋㅋㅋㅋㅋ\n")
                f.write("반갑습니다!!!!!!\n")

            config = from_dict(
                {
                    "pipeline": [
                        {"name": "ReadText", "args": {"path": input_path}},
                        {"name": "Normalize", "args": {}},
                        {"name": "WriteText", "args": {"path": output_path}},
                    ]
                }
            )
            pipeline = Pipeline()
            pipeline(config)

            assert os.path.exists(output_path)
            with open(output_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
            assert "ㅋㅋㅋㅋㅋㅋㅋ" not in lines[0]
