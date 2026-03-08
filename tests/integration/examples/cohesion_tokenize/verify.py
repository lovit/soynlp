"""Verify cohesion-based tokenization results."""

from soynlp.tokenizer import MaxScoreTokenizer

EXAMPLE_SENTENCES = [
    "국회의원선거법안이통과되었습니다",
    "아이오아이가프로듀스101telecom에서탄생했다",
    "청와대에서대통령이기자회견을열었다",
    "삼성전자갤럭시노트7배터리폭발사건",
    "연합뉴스기자가보도한내용입니다",
]


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


def verify(parameters: dict, answers_dir: str) -> None:
    word_cohesion = parameters["word_cohesion"]
    cohesion_scores = {word: score.leftside for word, score in word_cohesion.items() if score.leftside > 0.1}

    assert len(cohesion_scores) > 100
    assert all(s > 0.1 for s in cohesion_scores.values())
    top_scores = sorted(cohesion_scores.values(), reverse=True)[:10]
    assert top_scores[0] == 1.0

    tokenizer = MaxScoreTokenizer(scores=cohesion_scores)
    tokenize_lines = []
    for sent in EXAMPLE_SENTENCES:
        tokens = tokenizer.tokenize(sent)
        result = " / ".join(tokens)
        tokenize_lines.append(f"{sent}\t{result}")

    expected = _read_lines(f"{answers_dir}/tokenized.txt")
    assert tokenize_lines == expected
