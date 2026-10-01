import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from soynlp.artifacts import load_matrix
from soynlp.cli import main
from soynlp.utils import pmi
from soynlp.vectorizer import sent_to_word_contexts_matrix


def test_context_shape_empty_and_reused_vocabulary():
    matrix, vocabulary = sent_to_word_contexts_matrix(["a b", "z"], min_tf=1, verbose=False)
    assert matrix.shape == (3, 3)
    assert len(vocabulary) == 3
    empty, vocabulary = sent_to_word_contexts_matrix([], min_tf=1, verbose=False)
    assert empty.shape == (0, 0) and vocabulary == []
    reused, vocabulary = sent_to_word_contexts_matrix(["a b unknown"], vocabulary=["z", "b", "a"], verbose=False)
    assert vocabulary == ["z", "b", "a"] and reused.shape == (3, 3)
    assert reused[1, 2] == 1


def test_saved_matrix_workflow_and_vocabulary_reuse(tmp_path, capsys):
    source = tmp_path / "corpus.txt"
    source.write_text("a b c\na b\na d\nb c d\n")
    counts = tmp_path / "contexts.npz"
    transformed = tmp_path / "pmi.npz"
    vectors = tmp_path / "embedding.npz"
    main(["vectorize", "contexts", "--input", str(source), "--parameters", '{"min_tf":1}', "-o", str(counts)])
    matrix, metadata = load_matrix(counts)
    assert matrix.shape == (4, 4)
    main(["matrix", "pmi", "--input", str(counts), "-o", str(transformed)])
    main(["matrix", "svd", "--input", str(transformed), "-o", str(vectors)])
    rows, row_metadata = load_matrix(vectors)
    columns, _ = load_matrix(tmp_path / "embedding-columns.npz")
    assert rows.shape == columns.shape == (4, 2)
    assert row_metadata.parameters is not None
    assert len(row_metadata.parameters["singular_values"]) == 2
    main(["matrix", "similar", "--input", str(vectors), "--query", "a", "--topk", "2"])
    assert len(json.loads(capsys.readouterr().out)["data"]) == 2
    bag = tmp_path / "bow.npz"
    main(["vectorize", "bow", "--input", str(source), "--vocabulary", str(counts), "-o", str(bag)])
    matrix, bag_metadata = load_matrix(bag)
    assert matrix.shape == (4, 4) and bag_metadata.data["vocabulary"] == metadata.data["vocabulary"]
    assert matrix.sum() == 10


def test_matrix_errors_and_svd_overwrite_guard(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["vectorize", "bow", "--text", "한국어"])
    assert error.value.code == 2
    source = tmp_path / "counts.npz"
    main(["vectorize", "contexts", "--text", "a b c", "--parameters", '{"min_tf":1}', "-o", str(source)])
    output = tmp_path / "embedding.npz"
    (tmp_path / "embedding-columns.json").write_text("existing metadata")
    with pytest.raises(SystemExit) as error:
        main(["matrix", "svd", "--input", str(source), "-o", str(output)])
    assert error.value.code == 1 and not output.exists()


def test_zero_pmi_and_invalid_counts():
    matrix, px, py = pmi(csr_matrix((2, 3)))
    assert matrix.shape == (2, 3) and matrix.nnz == 0
    np.testing.assert_array_equal(px, np.zeros((1, 2)))
    np.testing.assert_array_equal(py, np.zeros((1, 3)))
    with pytest.raises(ValueError, match="nonnegative"):
        pmi(csr_matrix([[-1, 1]]))
