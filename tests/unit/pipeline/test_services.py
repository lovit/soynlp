import json
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from soynlp.artifacts import load_artifact, load_matrix
from soynlp.cli import main
from soynlp.configs.config import from_dict
from soynlp.pipeline import Pipeline
from soynlp.services import text_processor


def test_saved_resource_and_python_cli_pipeline_agree(tmp_path, capsys):
    output = tmp_path / "nouns.json"
    texts = ["한국어를 공부합니다", "한국어는 재미있습니다"] * 10
    config = from_dict(
        {
            "pipeline": [
                {"name": "Extract", "args": {"kind": "nouns", "out_key": "nouns"}},
                {"name": "WriteArtifact", "args": {"path": str(output), "in_key": "nouns"}},
                {"name": "ReadArtifact", "args": {"path": str(output), "out_key": "loaded"}},
                {"name": "TextTokenize", "args": {"method": "l", "resource_key": "loaded", "result_key": "tokens"}},
            ]
        }
    )
    initial = {"corpus": [{"id": i, "text": text} for i, text in enumerate(texts)]}
    result = Pipeline()(config, initial)
    resource = load_artifact(output)
    expected = text_processor("tokenize", "l", resource=resource)(texts[0])
    assert result["corpus"][0]["tokens"] == expected
    assert result["corpus"][0]["id"] == 0
    main(["tokenize", "--method", "l", "--resource", str(output), "--text", texts[0]])
    assert json.loads(capsys.readouterr().out)["tokens"] == expected


def test_pipeline_matrix_saved_output_matches_cli(tmp_path):
    output = tmp_path / "matrix.npz"
    config = from_dict(
        {
            "pipeline": [
                {"name": "Vectorize", "args": {"method": "contexts", "parameters": {"min_tf": 1}}},
                {"name": "PMI", "args": {}},
                {"name": "SVD", "args": {"parameters": {"n_components": 1}}},
                {"name": "WriteMatrix", "args": {"path": str(output)}},
            ]
        }
    )
    result = Pipeline()(config, {"corpus": [{"text": "a b c"}, {"text": "a c"}]})
    matrix, metadata = load_matrix(output)
    np.testing.assert_array_equal(matrix, result["matrix"])
    source = tmp_path / "corpus.txt"
    source.write_text("a b c\na c\n")
    counts = tmp_path / "counts.npz"
    probabilities = tmp_path / "pmi.npz"
    embedding = tmp_path / "embedding.npz"
    main(["vectorize", "contexts", "--input", str(source), "--parameters", '{"min_tf":1}', "-o", str(counts)])
    main(["matrix", "pmi", "--input", str(counts), "-o", str(probabilities)])
    main(["matrix", "svd", "--input", str(probabilities), "--parameters", '{"n_components":1}', "-o", str(embedding)])
    cli_matrix, cli_metadata = load_matrix(embedding)
    np.testing.assert_allclose(matrix, cli_matrix)
    assert metadata.data["row_vocabulary"] == cli_metadata.data["row_vocabulary"]


def test_validation_rejects_unknown_args_missing_keys_and_methods_without_execution(tmp_path):
    output = tmp_path / "not-created.jsonl"
    valid = from_dict(
        {
            "pipeline": [
                {"name": "ReadText", "args": {"path": "/nonexistent-input.txt"}},
                {"name": "WriteJson", "args": {"path": str(output)}},
            ]
        }
    )
    with patch("builtins.open", side_effect=AssertionError("must not read files")):
        assert len(Pipeline.validate(valid)) == 2
    assert not output.exists()
    for tasks in [
        [{"name": "Unknown", "args": {}}],
        [{"name": "WriteJson", "args": {"path": str(output)}}],
        [{"name": "ReadText", "args": {"path": "input", "typo": 1}}],
        [{"name": "TextNormalize", "args": {"method": "unknown"}}],
    ]:
        with pytest.raises(ValueError):
            Pipeline.validate(from_dict({"pipeline": tasks}))


def test_write_json_refuses_score_mapping(tmp_path):
    path = tmp_path / "scores.jsonl"
    config = from_dict({"pipeline": [{"name": "WriteJson", "args": {"path": str(path), "in_key": "nouns"}}]})
    with pytest.raises(TypeError, match="save_artifact"):
        Pipeline()(config, {"nouns": {"한국어": {"score": 1.0, "frequency": 5}}})
    assert not path.exists()


def test_cli_validate_tasks_and_run(tmp_path, capsys):
    source = tmp_path / "input.txt"
    source.write_text("한국어ㅋㅋㅋㅋ\n")
    config = tmp_path / "pipeline.yaml"
    output = tmp_path / "output.jsonl"
    config.write_text(
        yaml.safe_dump(
            {
                "pipeline": [
                    {"name": "ReadDocuments", "args": {"path": str(source)}},
                    {"name": "TextNormalize", "args": {"method": "repeat"}},
                    {"name": "WriteDocuments", "args": {"path": str(output), "format": "jsonl"}},
                ]
            }
        )
    )
    main(["pipeline", "validate", "-c", str(config)])
    assert not output.exists() and len(json.loads(capsys.readouterr().out)) == 3
    main(["pipeline", "run", "-c", str(config)])
    assert json.loads(output.read_text())["result"] == "한국어ㅋㅋ"
    main(["pipeline", "tasks"])
    assert "ReadArtifact" in json.loads(capsys.readouterr().out)
