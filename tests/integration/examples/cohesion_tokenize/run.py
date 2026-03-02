"""Extract cohesion scores and tokenize example sentences.

Usage:
    uv run python tests/integration/examples/cohesion_tokenize/run.py
"""

import os

from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.utils import CorpusLoader
from soynlp.word import WordExtractor

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_PATH = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

EXAMPLE_SENTENCES = [
    "국회의원선거법안이통과되었습니다",
    "아이오아이가프로듀스101telecom에서탄생했다",
    "청와대에서대통령이기자회견을열었다",
    "삼성전자갤럭시노트7배터리폭발사건",
    "연합뉴스기자가보도한내용입니다",
]


def main():
    corpus = CorpusLoader(DATA_PATH, format="jsonl", verbose=True)
    print(f"Corpus size: {len(corpus)} documents")

    # Extract word scores using WordExtractor
    word_extractor = WordExtractor(verbose=True)
    words = word_extractor.extract(corpus, min_frequency=5)
    cohesion_scores = {word: score.leftside for word, score in words["cohesion"].items() if score.leftside > 0.1}
    print(f"Extracted {len(cohesion_scores)} words with cohesion > 0.1")

    # Save top cohesion scores
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    top_cohesions = sorted(cohesion_scores.items(), key=lambda x: -x[1])[:100]
    cohesion_path = os.path.join(OUTPUT_DIR, "top_cohesion.txt")
    with open(cohesion_path, "w", encoding="utf-8") as f:
        for word, score in top_cohesions:
            f.write(f"{word}\t{score:.4f}\n")
    print(f"Top-100 cohesion scores saved to {cohesion_path}")

    # Tokenize example sentences
    tokenizer = MaxScoreTokenizer(scores=cohesion_scores)
    tokenize_path = os.path.join(OUTPUT_DIR, "tokenized.txt")
    with open(tokenize_path, "w", encoding="utf-8") as f:
        for sent in EXAMPLE_SENTENCES:
            tokens = tokenizer.tokenize(sent)
            result = " / ".join(tokens)
            f.write(f"{sent}\t{result}\n")
            print(f"  {sent}")
            print(f"    -> {result}")

    print(f"\nTokenization results saved to {tokenize_path}")


if __name__ == "__main__":
    main()
