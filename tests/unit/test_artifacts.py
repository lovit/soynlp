import json
from collections import namedtuple

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from soynlp.artifacts import Artifact, load_artifact, load_matrix, save_artifact, save_matrix
from soynlp.documents import read_documents, write_documents


def test_score_roundtrip_and_overwrite_protection(tmp_path):
    score = namedtuple("Score", "frequency score")
    path = tmp_path / "nouns.json"
    save_artifact(Artifact("nouns", {"한국어": score(3, 0.8)}, {"method": "lr"}), path)
    assert load_artifact(path, expected_kind="nouns").data == {"한국어": {"frequency": 3, "score": 0.8}}
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_artifact(Artifact("nouns", {}), path)
    assert path.read_bytes() == before
    save_artifact(Artifact("nouns", {}), path, overwrite=True)
    assert load_artifact(path).data == {}
    with pytest.raises(ValueError, match="kind"):
        load_artifact(path, expected_kind="words")


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_unknown_schema(tmp_path, version):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": version, "kind": "nouns", "parameters": {}, "data": {}}))
    with pytest.raises(ValueError, match="schema_version"):
        load_artifact(path)


@pytest.mark.parametrize("sparse", [True, False])
def test_matrix_roundtrip_and_matching_metadata(tmp_path, sparse):
    path = tmp_path / "bow.npz"
    value = np.array([[1, 0], [2, 3]])
    save_matrix(path, csr_matrix(value) if sparse else value, vocabulary=["한국어", "공부"])
    matrix, metadata = load_matrix(path)
    np.testing.assert_array_equal(matrix.toarray() if sparse else matrix, value)
    assert metadata.data["vocabulary"] == ["한국어", "공부"]
    with pytest.raises(FileExistsError):
        save_matrix(path, value)
    path.write_bytes(b"wrong file")
    with pytest.raises(ValueError, match="do not match"):
        load_matrix(path)


def test_matrix_sidecar_protection_and_vocabulary_validation(tmp_path):
    path = tmp_path / "matrix.npz"
    path.with_suffix(".json").write_text("existing")
    with pytest.raises(FileExistsError):
        save_matrix(path, np.zeros((1, 2)))
    assert not path.exists()
    with pytest.raises(ValueError, match="vocabulary"):
        save_matrix(tmp_path / "other.npz", np.zeros((1, 2)), vocabulary=["a"])


def test_streaming_documents_metadata_and_failed_write(tmp_path):
    path = tmp_path / "docs.jsonl"
    records = [{"id": 10, "text": "한국어", "meta": {"source": "sample"}}, {"id": 11, "text": ""}]
    write_documents(iter(records), path)
    assert list(read_documents(path, format="jsonl")) == records
    with pytest.raises(TypeError, match="save_artifact"):
        write_documents({"한국어": 3}, tmp_path / "scores.jsonl")
    bad = tmp_path / "bad.jsonl"
    with pytest.raises(TypeError):
        write_documents(iter([records[0], "bad record"]), bad)
    assert not bad.exists()


def test_document_validation_has_line_number(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text": "valid"}\n{"text": 42}\n')
    with pytest.raises(ValueError, match="Line 2"):
        list(read_documents(path, format="jsonl"))
