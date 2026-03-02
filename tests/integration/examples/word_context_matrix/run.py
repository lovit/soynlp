"""Build word-context co-occurrence matrix from news corpus.

Uses sent_to_word_contexts_matrix to create sparse matrix,
then finds similar words using cosine distance.

Usage:
    uv run python tests/integration/examples/word_context_matrix/run.py
"""

import os

from soynlp.utils import CorpusLoader, most_similar
from soynlp.vectorizer import sent_to_word_contexts_matrix

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_PATH = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    # Load corpus as sentences (strings)
    loader = CorpusLoader(DATA_PATH, format="jsonl")
    sents = [item["text"] for item in loader]
    print(f"Loaded {len(sents)} sentences")

    # Build word-context matrix
    print("\nBuilding word-context co-occurrence matrix...")
    matrix, idx2vocab = sent_to_word_contexts_matrix(
        sents,
        windows=3,
        min_tf=20,
        dynamic_weight=True,
        verbose=True,
    )
    print(f"Matrix shape: {matrix.shape}")
    print(f"Vocabulary size: {len(idx2vocab)}")
    print(f"Non-zero elements: {matrix.nnz}")

    # Build lookup tables
    vocab2idx = {word: idx for idx, word in enumerate(idx2vocab)}

    # Find similar words for query terms
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    query_words = ["경찰", "대통령", "삼성", "서울", "사건", "정부", "국회", "기자"]

    output_path = os.path.join(OUTPUT_DIR, "similar_words.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for query in query_words:
            if query not in vocab2idx:
                print(f"  '{query}' not in vocabulary, skipping")
                continue

            similars = most_similar(query, matrix, vocab2idx, idx2vocab, topk=10)
            f.write(f"# {query}\n")
            print(f"\n  Similar to '{query}':")
            for word, sim in similars:
                f.write(f"  {word}\t{sim:.4f}\n")
                print(f"    {word}: {sim:.4f}")
            f.write("\n")

    print(f"\nSimilar words saved to {output_path}")

    # Save vocabulary stats
    stats_path = os.path.join(OUTPUT_DIR, "matrix_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"matrix_shape: {matrix.shape}\n")
        f.write(f"vocabulary_size: {len(idx2vocab)}\n")
        f.write(f"non_zero_elements: {matrix.nnz}\n")
        n_rows, n_cols = matrix.shape  # type: ignore[misc]
        f.write(f"density: {matrix.nnz / (n_rows * n_cols):.6f}\n")
        f.write("\ntop_50_vocab:\n")
        for word in idx2vocab[:50]:
            f.write(f"  {word}\n")
    print(f"Matrix stats saved to {stats_path}")


if __name__ == "__main__":
    main()
