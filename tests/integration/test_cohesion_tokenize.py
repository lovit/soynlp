"""Integration test: Cohesion score extraction and tokenization."""

from soynlp.tokenizer import MaxScoreTokenizer

from .conftest import read_answer_lines

EXAMPLE_SENTENCES = [
    "국회의원선거법안이통과되었습니다",
    "아이오아이가프로듀스101telecom에서탄생했다",
    "청와대에서대통령이기자회견을열었다",
    "삼성전자갤럭시노트7배터리폭발사건",
    "연합뉴스기자가보도한내용입니다",
]


def test_cohesion_scores(news_word_scores):
    """Verify cohesion extraction produces enough high-scoring words."""
    cohesion_scores = {word: score.leftside for word, score in news_word_scores["cohesion"].items() if score.leftside > 0.1}

    assert len(cohesion_scores) > 100
    assert all(s > 0.1 for s in cohesion_scores.values())
    top_scores = sorted(cohesion_scores.values(), reverse=True)[:10]
    assert top_scores[0] == 1.0


def test_tokenization(news_word_scores):
    """Verify tokenization results match expected output."""
    cohesion_scores = {word: score.leftside for word, score in news_word_scores["cohesion"].items() if score.leftside > 0.1}

    tokenizer = MaxScoreTokenizer(scores=cohesion_scores)
    tokenize_lines = []
    for sent in EXAMPLE_SENTENCES:
        tokens = tokenizer.tokenize(sent)
        result = " / ".join(tokens)
        tokenize_lines.append(f"{sent}\t{result}")

    expected_tokenize = read_answer_lines("cohesion_tokenize", "tokenized.txt")
    assert tokenize_lines == expected_tokenize
