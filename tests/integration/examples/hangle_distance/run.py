"""Hangle character operations and distance calculations.

Tests decompose/compose, jamo_levenshtein distance,
and ConvolutionHangleEncoder on Korean words extracted from news.

Usage:
    uv run python tests/integration/examples/hangle_distance/run.py
"""

import os

from soynlp.hangle import (
    ConvolutionHangleEncoder,
    character_is_complete_korean,
    compose,
    cosine_distance,
    decompose,
    jaccard_distance,
    jamo_levenshtein,
    levenshtein,
)
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_PATH = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def test_decompose_compose():
    """Test decompose → compose roundtrip."""
    print("=== Decompose/Compose Roundtrip ===")
    test_chars = list("한글테스트입니다가나다라마바사아자차카타파하")
    results = []
    for ch in test_chars:
        parts = decompose(ch)
        if parts is not None:
            recomposed = compose(*parts)
            ok = recomposed == ch
            results.append((ch, parts, recomposed, ok))
    print(f"  Tested {len(results)} characters, all roundtrip OK: {all(r[3] for r in results)}")
    return results


def test_distance_calculations():
    """Test various distance metrics on Korean word pairs."""
    print("\n=== Distance Calculations ===")
    word_pairs = [
        ("대통령", "대통렬"),  # typo: 령→렬
        ("삼성전자", "삼성전기"),  # similar company
        ("경찰", "경철"),  # typo
        ("연합뉴스", "연합뉴우스"),  # elongation
        ("서울", "서울시"),  # extension
        ("대한민국", "대한민극"),  # typo
        ("아이오아이", "아이오에이"),  # similar name
        ("국회의원", "국회위원"),  # similar word
    ]

    results = []
    for w1, w2 in word_pairs:
        lev = levenshtein(w1, w2)
        jamo_lev = jamo_levenshtein(w1, w2)
        cos = cosine_distance(w1, w2)
        jac = jaccard_distance(w1, w2)
        results.append((w1, w2, lev, jamo_lev, cos, jac))
        print(f"  '{w1}' vs '{w2}':")
        print(f"    levenshtein={lev:.2f}, jamo_levenshtein={jamo_lev:.2f}, cosine={cos:.4f}, jaccard={jac:.4f}")

    return results


def test_encoder():
    """Test ConvolutionHangleEncoder on sentences."""
    print("\n=== ConvolutionHangleEncoder ===")
    encoder = ConvolutionHangleEncoder()
    print(f"  Encoder dimension: {encoder.dim}")

    test_sents = [
        "한글 인코딩 테스트",
        "서울에서 부산까지",
        "대통령이 기자회견을 열었다",
    ]

    results = []
    for sent in test_sents:
        encoded = encoder.encode(sent)
        results.append((sent, encoded.shape))
        print(f"  '{sent}' → shape={encoded.shape}")

    # Roundtrip test
    print("\n  Onehot roundtrip test:")
    for sent in test_sents:
        onehot = encoder.sent_to_onehot(sent)
        decoded = encoder.onehot_to_sent(onehot)
        ok = decoded == sent
        print(f"    '{sent}' → onehot → '{decoded}' (match={ok})")

    return results


def find_similar_words_in_corpus():
    """Find similar words from corpus using jamo_levenshtein."""
    print("\n=== Finding similar words in corpus ===")
    loader = CorpusLoader(DATA_PATH, format="jsonl")

    # Collect unique eojeols from first 1000 documents
    words = set()
    for i, item in enumerate(loader):
        if i >= 1000:
            break
        for eojeol in item["text"].split():
            if 2 <= len(eojeol) <= 5 and all(character_is_complete_korean(c) for c in eojeol):
                words.add(eojeol)

    print(f"  Collected {len(words)} unique Korean words")

    # Find nearest neighbors for query words
    queries = ["대통령", "경찰", "서울"]
    word_list = list(words)

    results = {}
    for query in queries:
        if query not in words:
            continue
        distances = [(w, jamo_levenshtein(query, w)) for w in word_list if w != query]
        distances.sort(key=lambda x: x[1])
        results[query] = distances[:5]
        print(f"  Most similar to '{query}':")
        for w, d in distances[:5]:
            print(f"    {w}: jamo_distance={d:.2f}")

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_decompose_compose()
    distance_results = test_distance_calculations()
    test_encoder()
    similar_results = find_similar_words_in_corpus()

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "distance_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Distance Calculations\n")
        f.write(f"{'word1':<15} {'word2':<15} {'levenshtein':>12} {'jamo_lev':>10} {'cosine':>10} {'jaccard':>10}\n")
        for w1, w2, lev, jlev, cos, jac in distance_results:
            f.write(f"{w1:<15} {w2:<15} {lev:>12.2f} {jlev:>10.2f} {cos:>10.4f} {jac:>10.4f}\n")

        f.write("\n# Similar words from corpus\n")
        for query, neighbors in similar_results.items():
            f.write(f"\n{query}:\n")
            for w, d in neighbors:
                f.write(f"  {w}\t{d:.2f}\n")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
