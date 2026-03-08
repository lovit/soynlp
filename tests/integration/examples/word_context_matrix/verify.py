"""Verify word-context co-occurrence matrix results."""

import os

from soynlp.utils import CorpusLoader, most_similar
from soynlp.vectorizer import sent_to_word_contexts_matrix

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
NEWS_DATA = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")


def _read_answer(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def verify(answers_dir: str) -> None:
    loader = CorpusLoader(NEWS_DATA, format="jsonl")
    sents = [item["text"] for item in loader]

    matrix, idx2vocab = sent_to_word_contexts_matrix(
        sents,
        windows=3,
        min_tf=20,
        dynamic_weight=True,
        verbose=False,
    )
    vocab2idx = {word: idx for idx, word in enumerate(idx2vocab)}

    # Verify matrix stats
    n_rows, n_cols = matrix.shape  # type: ignore[misc]
    stats_lines = [
        f"matrix_shape: {matrix.shape}",
        f"vocabulary_size: {len(idx2vocab)}",
        f"non_zero_elements: {matrix.nnz}",
        f"density: {matrix.nnz / (n_rows * n_cols):.6f}",
        "",
        "top_50_vocab:",
    ]
    for word in idx2vocab[:50]:
        stats_lines.append(f"  {word}")
    actual_stats = "\n".join(stats_lines) + "\n"
    expected_stats = _read_answer(f"{answers_dir}/matrix_stats.txt")
    assert actual_stats == expected_stats

    # Verify similar words — structural checks (non-deterministic order)
    query_words = ["경찰", "대통령", "삼성", "서울", "사건", "정부", "국회", "기자"]
    for query in query_words:
        if query not in vocab2idx:
            continue
        similars = most_similar(query, matrix, vocab2idx, idx2vocab, topk=10)

        assert len(similars) == 10, f"'{query}' should have 10 similar words, got {len(similars)}"

        words_returned = [word for word, _ in similars]
        scores = [sim for _, sim in similars]

        assert len(set(words_returned)) == 10, f"'{query}' has duplicate similar words"
        assert query not in words_returned, f"'{query}' should not be in its own similar words"

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"'{query}' similar words not in descending order"

        assert all(0 < s <= 1.0 for s in scores), f"'{query}' has scores out of range"
