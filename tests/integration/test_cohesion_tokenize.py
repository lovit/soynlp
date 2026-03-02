"""Integration test: Cohesion score extraction and tokenization."""

from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.utils import CorpusLoader
from soynlp.word import WordExtractor

from .conftest import NEWS_DATA, read_answer_lines

EXAMPLE_SENTENCES = [
    "국회의원선거법안이통과되었습니다",
    "아이오아이가프로듀스101telecom에서탄생했다",
    "청와대에서대통령이기자회견을열었다",
    "삼성전자갤럭시노트7배터리폭발사건",
    "연합뉴스기자가보도한내용입니다",
]


def test_cohesion_scores():
    """Verify cohesion extraction produces enough high-scoring words."""
    corpus = CorpusLoader(NEWS_DATA, format="jsonl", verbose=False)
    word_extractor = WordExtractor(verbose=False)
    words = word_extractor.extract(corpus, min_frequency=5)
    cohesion_scores = {word: score.leftside for word, score in words["cohesion"].items() if score.leftside > 0.1}

    # Should extract a reasonable number of words
    assert len(cohesion_scores) > 100
    # All scores should be positive
    assert all(s > 0.1 for s in cohesion_scores.values())
    # Top scores should be 1.0
    top_scores = sorted(cohesion_scores.values(), reverse=True)[:10]
    assert top_scores[0] == 1.0


def test_tokenization():
    """Verify tokenization results match expected output."""
    corpus = CorpusLoader(NEWS_DATA, format="jsonl", verbose=False)
    word_extractor = WordExtractor(verbose=False)
    words = word_extractor.extract(corpus, min_frequency=5)
    cohesion_scores = {word: score.leftside for word, score in words["cohesion"].items() if score.leftside > 0.1}

    tokenizer = MaxScoreTokenizer(scores=cohesion_scores)
    tokenize_lines = []
    for sent in EXAMPLE_SENTENCES:
        tokens = tokenizer.tokenize(sent)
        result = " / ".join(tokens)
        tokenize_lines.append(f"{sent}\t{result}")

    expected_tokenize = read_answer_lines("cohesion_tokenize", "tokenized.txt")
    assert tokenize_lines == expected_tokenize
