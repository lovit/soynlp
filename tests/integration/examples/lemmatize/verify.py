"""Verify lemmatization and conjugation results."""

import os

from soynlp.lemmatizer import Lemmatizer, conjugate, lemma_candidate

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
STEM_DIR = os.path.join(ROOT_DIR, "soynlp/lemmatizer/dictionary/default/Stem")
EOMI_PATH = os.path.join(ROOT_DIR, "soynlp/lemmatizer/dictionary/default/Eomi/Eomi.txt")


def _read_answer(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load_stems() -> set[str]:
    stems: set[str] = set()
    for pos in ["Adjective", "Verb"]:
        path = os.path.join(STEM_DIR, f"{pos}.txt")
        with open(path, encoding="utf-8") as f:
            for line in f:
                stems.add(line.split()[0])
    return stems


def _load_endings() -> set[str]:
    endings: set[str] = set()
    with open(EOMI_PATH, encoding="utf-8") as f:
        for line in f:
            endings.add(line.split()[0])
    return endings


def _build_results() -> str:
    conjugation_cases = [
        ("하", "ㄴ다"),
        ("하", "았다"),
        ("먹", "었다"),
        ("먹", "는"),
        ("좋", "은"),
        ("좋", "아서"),
        ("가", "ㄴ다"),
        ("오", "았다"),
        ("크", "ㄴ"),
        ("예쁘", "ㄴ"),
    ]
    lines = ["# Conjugation"]
    for stem, ending in conjugation_cases:
        forms = conjugate(stem, ending)
        forms_str = ", ".join(sorted(forms))
        lines.append(f"{stem} + {ending}\t{forms_str}")

    stems = _load_stems()
    endings = _load_endings()
    lemmatizer = Lemmatizer(stems=stems, endings=endings)
    test_words = [
        "했다",
        "한다",
        "하는",
        "하면",
        "먹었다",
        "먹는",
        "먹으면",
        "좋았다",
        "좋은",
        "좋아서",
        "갔다",
        "간다",
        "가면",
        "예쁜",
        "예뻤다",
        "달린다",
        "달렸다",
        "만들었다",
        "만드는",
    ]
    lines.append("")
    lines.append("# Lemmatization")
    for word in test_words:
        lemmas = lemmatizer.lemmatize(word)
        if lemmas:
            lemma_str = ", ".join(f"({s}, {e})" for s, e in sorted(lemmas))
        else:
            lemma_str = "(no match)"
        lines.append(f"{word}\t{lemma_str}")

    candidate_cases = [("했", "다"), ("먹었", "다"), ("좋", "은"), ("예쁜", ""), ("갔", "다")]
    lines.append("")
    lines.append("# Lemma Candidates")
    for left, right in candidate_cases:
        candidates = lemma_candidate(left, right)
        cand_str = ", ".join(f"({s}, {e})" for s, e in sorted(candidates))
        lines.append(f"L={left}, R={right}\t{cand_str}")

    return "\n".join(lines) + "\n"


def verify(answers_dir: str) -> None:
    expected = _read_answer(f"{answers_dir}/lemmatization_results.txt")
    actual = _build_results()
    assert actual == expected
