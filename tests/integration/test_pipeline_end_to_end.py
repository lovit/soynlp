"""Integration test: End-to-end pipeline (ReadJson -> Normalize -> ExtractNoun)."""

import os

from soynlp.configs.config import from_yaml
from soynlp.pipeline import Pipeline

from .conftest import ROOT_DIR, read_answer_lines

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "examples/pipeline_end_to_end/pipeline.yaml")


def test_pipeline_end_to_end():
    prev_cwd = os.getcwd()
    try:
        os.chdir(ROOT_DIR)
        config = from_yaml(CONFIG_PATH)
        pipeline = Pipeline()
        parameters = pipeline(config)
    finally:
        os.chdir(prev_cwd)

    assert "corpus" in parameters
    assert "nouns" in parameters

    nouns = parameters["nouns"]
    top_nouns = sorted(nouns.items(), key=lambda x: -x[1].frequency)[:50]
    actual_lines = [f"{noun}\t{score.frequency}\t{score.score:.4f}" for noun, score in top_nouns]

    expected_lines = read_answer_lines("pipeline_end_to_end", "pipeline_nouns.txt")
    assert actual_lines == expected_lines
