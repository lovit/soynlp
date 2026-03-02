"""Extract predicators (verbs/adjectives) from news corpus.

Uses LRNounExtractor for noun extraction first, then PredicatorExtractor
to find verbs and adjectives.

Usage:
    uv run python tests/integration/examples/extract_predicators/run.py
"""

import os

from soynlp.noun import LRNounExtractor
from soynlp.predicator import PredicatorExtractor
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_PATH = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    # Load corpus
    loader = CorpusLoader(DATA_PATH, format="jsonl", verbose=True)
    sents = [item["text"] for item in loader]
    print(f"Loaded {len(sents)} sentences")

    # Step 1: Extract nouns first (required by PredicatorExtractor)
    print("\n=== Step 1: Noun extraction ===")
    noun_extractor = LRNounExtractor(verbose=True)
    nouns = noun_extractor.extract(sents, min_noun_frequency=10)
    noun_set = set(nouns.keys())
    print(f"Extracted {len(noun_set)} nouns")

    # Step 2: Extract predicators
    print("\n=== Step 2: Predicator extraction ===")
    predicator_extractor = PredicatorExtractor(
        nouns=noun_set,
        verbose=True,
    )
    adjectives, verbs = predicator_extractor.train_extract(
        sents,
        min_eojeol_frequency=2,
        min_predicator_frequency=5,
    )
    print(f"Extracted {len(adjectives)} adjectives, {len(verbs)} verbs")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save top adjectives
    adj_path = os.path.join(OUTPUT_DIR, "top_adjectives.txt")
    top_adj = sorted(adjectives.items(), key=lambda x: -x[1].frequency)[:50]
    with open(adj_path, "w", encoding="utf-8") as f:
        for word, pred in top_adj:
            lemmas = ", ".join(f"{s}+{e}" for s, e in list(pred.lemma)[:3])
            f.write(f"{word}\t{pred.frequency}\t{lemmas}\n")
    print(f"\nTop-50 adjectives saved to {adj_path}")

    # Save top verbs
    verb_path = os.path.join(OUTPUT_DIR, "top_verbs.txt")
    top_verbs = sorted(verbs.items(), key=lambda x: -x[1].frequency)[:50]
    with open(verb_path, "w", encoding="utf-8") as f:
        for word, pred in top_verbs:
            lemmas = ", ".join(f"{s}+{e}" for s, e in list(pred.lemma)[:3])
            f.write(f"{word}\t{pred.frequency}\t{lemmas}\n")
    print(f"Top-50 verbs saved to {verb_path}")

    # Print samples
    print("\nTop-10 adjectives:")
    for word, pred in top_adj[:10]:
        lemmas = ", ".join(f"{s}+{e}" for s, e in list(pred.lemma)[:3])
        print(f"  {word} (freq={pred.frequency}): {lemmas}")

    print("\nTop-10 verbs:")
    for word, pred in top_verbs[:10]:
        lemmas = ", ".join(f"{s}+{e}" for s, e in list(pred.lemma)[:3])
        print(f"  {word} (freq={pred.frequency}): {lemmas}")


if __name__ == "__main__":
    main()
