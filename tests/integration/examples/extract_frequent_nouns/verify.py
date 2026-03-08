"""Verify frequent noun extraction results."""


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


def verify(parameters: dict, answers_dir: str) -> None:
    nouns = parameters["nouns"]
    top_nouns = sorted(nouns.items(), key=lambda x: -x[1].frequency)[:100]
    lines = [f"{noun}\t{score.frequency}\t{score.score:.4f}" for noun, score in top_nouns]

    expected = _read_lines(f"{answers_dir}/top_nouns.txt")
    assert lines == expected
