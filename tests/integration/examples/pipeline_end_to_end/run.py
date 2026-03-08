"""End-to-end pipeline test: ReadJson → Normalize → ExtractNouns.

Usage:
    uv run python tests/integration/examples/pipeline_end_to_end/run.py
"""

import os

from soynlp.configs.config import from_yaml
from soynlp.pipeline import Pipeline

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    # Change to root dir so relative paths in YAML work
    os.chdir(ROOT_DIR)

    config = from_yaml(CONFIG_PATH)
    pipeline = Pipeline()

    print("Running pipeline: ReadJson → Normalize → ExtractNoun")
    print(f"Config: {CONFIG_PATH}")
    parameters = pipeline(config)

    # Verify results
    assert "corpus" in parameters, "corpus not in parameters"
    assert "nouns" in parameters, "nouns not in parameters"

    nouns = parameters["nouns"]
    print(f"\nExtracted {len(nouns)} nouns from pipeline")

    # Save top nouns
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    top_nouns = sorted(nouns.items(), key=lambda x: -x[1].frequency)[:50]
    output_path = os.path.join(OUTPUT_DIR, "pipeline_nouns.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for noun, score in top_nouns:
            f.write(f"{noun}\t{score.frequency}\t{score.score:.4f}\n")

    print(f"Top-50 nouns saved to {output_path}")
    print("\nTop-10 nouns:")
    for noun, score in top_nouns[:10]:
        print(f"  {noun}: frequency={score.frequency}, score={score.score:.4f}")

    # Verify corpus was normalized
    corpus = parameters["corpus"]
    print(f"\nCorpus size: {len(corpus)} documents")
    print(f"Sample normalized text (first 100 chars): {corpus[0]['text'][:100]}...")


if __name__ == "__main__":
    main()
