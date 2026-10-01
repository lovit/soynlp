"""Run from the repository root: uv run --python 3.12 python examples/refactor_demo.py."""

import json
import tempfile
from pathlib import Path

from soynlp import Artifact, load_artifact, load_matrix, save_artifact, save_matrix
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2
from soynlp.postagger import KoreanPOSTagger
from soynlp.predicator import PredicatorExtractor
from soynlp.services import text_processor
from soynlp.utils import pmi, svd
from soynlp.vectorizer import sent_to_word_contexts_matrix
from soynlp.word import BigramExtractor, WordExtractor


def main() -> None:
    sentences = [
        "한국어를 공부합니다",
        "한국어는 재미있습니다",
        "아이오아이가 아이돌입니다",
        "아이오아이는 아이돌 그룹입니다",
    ] * 20
    nouns = LRNounExtractor(verbose=False).extract(sentences)
    assert nouns and LRNounExtractor_v2(verbose=False).train_extract(sentences)
    words = WordExtractor(min_frequency=2, verbose_points=0)
    words.train(sentences)
    word_scores = words.extract()
    bigrams = BigramExtractor(min_frequency=2, score="pmi", verbose=False).extract(sentences)
    adjectives, verbs = PredicatorExtractor(nouns=nouns, verbose=False).train_extract(sentences)
    assert KoreanPOSTagger().tag(sentences[0])
    assert ("먹", "었다") in text_processor("lemmatize")("먹었다")
    with tempfile.TemporaryDirectory(prefix="soynlp-demo-") as directory:
        path = Path(directory)
        resource = path / "nouns.json"
        save_artifact(Artifact("nouns", nouns), resource)
        loaded = load_artifact(resource, expected_kind="nouns")
        tokens = text_processor("tokenize", "l", resource=loaded)(sentences[0])
        contexts, vocabulary = sent_to_word_contexts_matrix(sentences, min_tf=1, verbose=False)
        probabilities, _, _ = pmi(contexts)
        embeddings, singular_values, _ = svd(probabilities, n_components=2, random_state=0)
        save_matrix(path / "contexts.npz", contexts, vocabulary=vocabulary, row_vocabulary=vocabulary)
        restored, metadata = load_matrix(path / "contexts.npz")
        assert restored.shape == contexts.shape and metadata.data["vocabulary"] == vocabulary
        assert len(singular_values) == 2
        print(
            json.dumps(
                {
                    "nouns": len(nouns),
                    "words": len(word_scores),
                    "bigrams": len(bigrams),
                    "adjectives": len(adjectives),
                    "verbs": len(verbs),
                    "tokens": tokens,
                    "embedding_shape": list(embeddings.shape),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
