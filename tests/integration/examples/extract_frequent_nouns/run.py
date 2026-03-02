"""Extract frequent nouns from news corpus using LRNounExtractor.

Usage:
    uv run python tests/integration/examples/extract_frequent_nouns/run.py
"""

import os

from soynlp.noun import LRNounExtractor
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_PATH = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    corpus = CorpusLoader(DATA_PATH, format="jsonl", verbose=True)
    print(f"Corpus size: {len(corpus)} documents")

    noun_extractor = LRNounExtractor(verbose=True)
    nouns = noun_extractor.extract(corpus, min_noun_frequency=10)
    print(f"Extracted {len(nouns)} nouns")

    top_nouns = sorted(nouns.items(), key=lambda x: -x[1].frequency)[:100]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "top_nouns.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for noun, score in top_nouns:
            f.write(f"{noun}\t{score.frequency}\t{score.score:.4f}\n")

    print(f"Top-100 nouns saved to {output_path}")
    print("\nTop-20 nouns:")
    for noun, score in top_nouns[:20]:
        print(f"  {noun}: frequency={score.frequency}, score={score.score:.4f}")


if __name__ == "__main__":
    main()
