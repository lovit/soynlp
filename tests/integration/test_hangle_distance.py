"""Integration test: Hangle distance calculations and similar word search."""

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

from .conftest import NEWS_DATA, read_answer


def test_decompose_compose_roundtrip():
    test_chars = list("한글테스트입니다가나다라마바사아자차카타파하")
    for ch in test_chars:
        parts = decompose(ch)
        assert parts is not None
        recomposed = compose(*parts)
        assert recomposed == ch


def test_encoder_roundtrip():
    encoder = ConvolutionHangleEncoder()
    test_sents = ["한글 인코딩 테스트", "서울에서 부산까지", "대통령이 기자회견을 열었다"]
    for sent in test_sents:
        encoded = encoder.encode(sent)
        assert encoded.shape[0] > 0
        onehot = encoder.sent_to_onehot(sent)
        decoded = encoder.onehot_to_sent(onehot)
        assert decoded == sent


def test_distance_calculations():
    """Test deterministic distance calculations against answer file."""
    expected = read_answer("hangle_distance", "distance_results.txt")
    # Extract only the distance table (before "# Similar words" section)
    expected_table = expected.split("\n# Similar words")[0].strip()

    word_pairs = [
        ("대통령", "대통렬"),
        ("삼성전자", "삼성전기"),
        ("경찰", "경철"),
        ("연합뉴스", "연합뉴우스"),
        ("서울", "서울시"),
        ("대한민국", "대한민극"),
        ("아이오아이", "아이오에이"),
        ("국회의원", "국회위원"),
    ]
    lines = ["# Distance Calculations"]
    lines.append(f"{'word1':<15} {'word2':<15} {'levenshtein':>12} {'jamo_lev':>10} {'cosine':>10} {'jaccard':>10}")
    for w1, w2 in word_pairs:
        lev = levenshtein(w1, w2)
        jlev = jamo_levenshtein(w1, w2)
        cos = cosine_distance(w1, w2)
        jac = jaccard_distance(w1, w2)
        lines.append(f"{w1:<15} {w2:<15} {lev:>12.2f} {jlev:>10.2f} {cos:>10.4f} {jac:>10.4f}")

    actual_table = "\n".join(lines)
    assert actual_table == expected_table


def test_similar_words_from_corpus():
    """Test similar word search structure (non-deterministic ordering, so check distances only)."""
    loader = CorpusLoader(NEWS_DATA, format="jsonl")
    words: set[str] = set()
    for i, item in enumerate(loader):
        if i >= 1000:
            break
        for eojeol in item["text"].split():
            if 2 <= len(eojeol) <= 5 and all(character_is_complete_korean(c) for c in eojeol):
                words.add(eojeol)

    queries = ["대통령", "경찰", "서울"]
    for query in queries:
        assert query in words
        distances = [(w, jamo_levenshtein(query, w)) for w in words if w != query]
        distances.sort(key=lambda x: x[1])
        top5 = distances[:5]
        assert len(top5) == 5
        # Distances should be non-negative and sorted
        for i in range(len(top5) - 1):
            assert top5[i][1] <= top5[i + 1][1]
