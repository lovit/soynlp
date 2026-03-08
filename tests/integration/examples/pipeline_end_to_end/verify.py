"""Verify end-to-end pipeline results (ReadJson -> Normalize -> ExtractNoun)."""


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


def verify(parameters: dict, answers_dir: str) -> None:
    assert "corpus" in parameters
    assert "nouns" in parameters

    nouns = parameters["nouns"]
    top_nouns = sorted(nouns.items(), key=lambda x: -x[1].frequency)[:50]
    actual_lines = [f"{noun}\t{score.frequency}\t{score.score:.4f}" for noun, score in top_nouns]

    expected_lines = _read_lines(f"{answers_dir}/pipeline_nouns.txt")
    assert actual_lines == expected_lines
