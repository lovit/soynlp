import os
import tempfile

from soynlp.configs.config import Config, TaskConfig, from_dict, from_yaml


class TestConfig:
    def test_from_dict(self):
        data = {
            "pipeline": [
                {"name": "Dummy", "args": {"name": "test"}},
            ]
        }
        config = from_dict(data)
        assert isinstance(config, Config)
        assert len(config.pipeline) == 1
        assert config.pipeline[0].name == "Dummy"
        assert config.pipeline[0].args == {"name": "test"}

    def test_from_yaml(self):
        yaml_content = """
pipeline:
  - name: ReadText
    args:
      path: /tmp/test.txt
  - name: Dummy
    args:
      name: test
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            with open(path, "w") as f:
                f.write(yaml_content)
            config = from_yaml(path)
            assert isinstance(config, Config)
            assert len(config.pipeline) == 2
            assert config.pipeline[0].name == "ReadText"
            assert config.pipeline[1].name == "Dummy"

    def test_task_config(self):
        tc = TaskConfig(name="Dummy", args={"key": "value"})
        assert tc.name == "Dummy"
        assert tc.args == {"key": "value"}

    def test_multiple_tasks(self):
        data = {
            "pipeline": [
                {"name": "ReadText", "args": {"path": "test.txt"}},
                {"name": "Normalize", "args": {}},
                {"name": "WriteText", "args": {"path": "out.txt"}},
            ]
        }
        config = from_dict(data)
        assert len(config.pipeline) == 3
        names = [tc.name for tc in config.pipeline]
        assert names == ["ReadText", "Normalize", "WriteText"]
