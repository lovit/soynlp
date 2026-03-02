"""Integration test: Word-context co-occurrence matrix."""

from soynlp.utils import CorpusLoader, most_similar
from soynlp.vectorizer import sent_to_word_contexts_matrix

from .conftest import NEWS_DATA, read_answer


def test_word_context_matrix():
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
    expected_stats = read_answer("word_context_matrix", "matrix_stats.txt")
    assert actual_stats == expected_stats

    # Verify similar words
    query_words = ["경찰", "대통령", "삼성", "서울", "사건", "정부", "국회", "기자"]
    similar_lines: list[str] = []
    for query in query_words:
        if query not in vocab2idx:
            continue
        similars = most_similar(query, matrix, vocab2idx, idx2vocab, topk=10)
        similar_lines.append(f"# {query}")
        for word, sim in similars:
            similar_lines.append(f"  {word}\t{sim:.4f}")
        similar_lines.append("")
    actual_similar = "\n".join(similar_lines) + "\n"
    expected_similar = read_answer("word_context_matrix", "similar_words.txt")
    assert actual_similar == expected_similar
