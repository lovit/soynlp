"""Lemmatization and conjugation on Korean predicators.

Tests Lemmatizer with stems/endings from the built-in dictionary,
and conjugate function for surface form generation.

Usage:
    uv run python tests/integration/examples/lemmatize/run.py
"""

import os

from soynlp.lemmatizer import Lemmatizer, conjugate, lemma_candidate

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def load_default_stems():
    """Load default stem dictionary."""
    stems = set()
    for pos in ["Adjective", "Verb"]:
        path = os.path.join(ROOT_DIR, f"soynlp/lemmatizer/dictionary/default/Stem/{pos}.txt")
        with open(path, encoding="utf-8") as f:
            for line in f:
                word = line.split()[0]
                stems.add(word)
    return stems


def load_default_endings():
    """Load default ending dictionary."""
    path = os.path.join(ROOT_DIR, "soynlp/lemmatizer/dictionary/default/Eomi/Eomi.txt")
    endings = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            word = line.split()[0]
            endings.add(word)
    return endings


def test_conjugation():
    """Test conjugation of various stem + ending combinations."""
    print("=== Conjugation Tests ===")
    test_cases = [
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
    results = []
    for stem, ending in test_cases:
        forms = conjugate(stem, ending)
        forms_str = ", ".join(sorted(forms))
        results.append((stem, ending, forms_str))
        print(f"  {stem} + {ending} → {forms_str}")
    return results


def test_lemmatization():
    """Test lemmatization with built-in dictionaries."""
    print("\n=== Lemmatization Tests ===")
    stems = load_default_stems()
    endings = load_default_endings()
    print(f"  Loaded {len(stems)} stems, {len(endings)} endings")

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

    results = []
    for word in test_words:
        lemmas = lemmatizer.lemmatize(word)
        if lemmas:
            lemma_str = ", ".join(f"({s}, {e})" for s, e in sorted(lemmas))
        else:
            lemma_str = "(no match)"
        results.append((word, lemma_str))
        print(f"  {word} → {lemma_str}")
    return results


def test_lemma_candidate():
    """Test raw lemma candidate generation (without dictionary filtering)."""
    print("\n=== Lemma Candidate Tests ===")
    test_cases = [
        ("했", "다"),
        ("먹었", "다"),
        ("좋", "은"),
        ("예쁜", ""),
        ("갔", "다"),
    ]
    results = []
    for left, right in test_cases:
        candidates = lemma_candidate(left, right)
        cand_str = ", ".join(f"({s}, {e})" for s, e in sorted(candidates))
        results.append((left, right, cand_str))
        print(f"  L='{left}', R='{right}' → {cand_str}")
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conjugation_results = test_conjugation()
    lemmatization_results = test_lemmatization()
    candidate_results = test_lemma_candidate()

    # Save all results
    output_path = os.path.join(OUTPUT_DIR, "lemmatization_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Conjugation\n")
        for stem, ending, forms in conjugation_results:
            f.write(f"{stem} + {ending}\t{forms}\n")

        f.write("\n# Lemmatization\n")
        for word, lemmas in lemmatization_results:
            f.write(f"{word}\t{lemmas}\n")

        f.write("\n# Lemma Candidates\n")
        for left, right, cands in candidate_results:
            f.write(f"L={left}, R={right}\t{cands}\n")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
