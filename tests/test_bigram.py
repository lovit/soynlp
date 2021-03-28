from pprint import pprint
from soynlp.word import BigramExtractor


def test_bigram_extractor():
    train_data = ["a b c d e", "a b c a b c", "a d e b", "a b"]
    for method, threshold in [("frequency", 1), ("pmi", 1.0), ("mikolov", 0.1)]:
        bigram_extractor = BigramExtractor(min_frequency=1, score=method, verbose=False)
        bigrams = bigram_extractor.extract(train_data, threshold=threshold)
        print(f"\nmethod={method}")
        pprint(bigrams)
        assert "a - b" in bigrams
