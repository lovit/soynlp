"""Auto-discover and test all integration examples."""

import importlib.util
import os

import pytest

from soynlp.configs.config import from_yaml
from soynlp.pipeline import Pipeline

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _discover_examples():
    for name in sorted(os.listdir(EXAMPLES_DIR)):
        example_dir = os.path.join(EXAMPLES_DIR, name)
        verify_path = os.path.join(example_dir, "verify.py")
        if os.path.isdir(example_dir) and os.path.isfile(verify_path):
            yield pytest.param(example_dir, id=name)


def _load_verify_module(example_dir):
    spec = importlib.util.spec_from_file_location("verify", os.path.join(example_dir, "verify.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("example_dir", _discover_examples())
def test_example(example_dir):
    answers_dir = os.path.join(example_dir, "answers")
    config_path = os.path.join(example_dir, "config.yaml")
    verify_module = _load_verify_module(example_dir)

    if os.path.isfile(config_path):
        prev_cwd = os.getcwd()
        try:
            os.chdir(ROOT_DIR)
            config = from_yaml(config_path)
            pipeline = Pipeline()
            parameters = pipeline(config)
        finally:
            os.chdir(prev_cwd)
        verify_module.verify(parameters, answers_dir)
    else:
        verify_module.verify(answers_dir)
