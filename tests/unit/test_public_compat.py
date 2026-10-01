"""Compare public compatibility APIs with results produced by master 503eaee."""

import json
from pathlib import Path

import pytest

import soynlp
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2, NewsNounExtractor, NounScore
from soynlp.predicator import Predicator
from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer, NounLMatchTokenizer
from soynlp.utils import DoublespaceLineCorpus, EojeolCounter
from soynlp.word import Bigram, WordExtractor

FIXTURES = Path(__file__).parent / "fixtures" / "compat"


@pytest.fixture
def master():
    return json.loads((FIXTURES / "master.json").read_text())


def test_root_imports_and_tuple_results():
    for name in ["hangle", "normalizer", "noun", "word", "tokenizer", "lemmatizer", "postagger", "vectorizer"]:
        assert getattr(soynlp, name)
    assert soynlp.DoublespaceLineCorpus is DoublespaceLineCorpus
    assert tuple(NounScore(3, 0.8)) == (3, 0.8)
    assert NounScore(3, 0.8)._asdict() == {"frequency": 3, "score": 0.8}
    assert tuple(Predicator(2, [("먹", "다")])) == (2, [("먹", "다")])


def test_master_word_algorithm_and_model(master, tmp_path):
    extractor = WordExtractor(min_frequency=2, verbose_points=0)
    extractor.train(master["sents"])
    assert {w: s._asdict() for w, s in extractor.extract().items()} == master["words"]
    loaded = WordExtractor()
    loaded.load(str(FIXTURES / "word-model.pkl"))
    assert loaded.extract() == extractor.extract()
    model_path = tmp_path / "word-model"
    loaded.save(str(model_path))
    restored = WordExtractor()
    restored.load(str(model_path))
    assert restored.extract() == loaded.extract()


def test_master_noun_v2_algorithm(master):
    extractor = LRNounExtractor_v2(verbose=False)
    nouns = extractor.train_extract(master["sents"], min_noun_frequency=2)
    assert {w: list(s) for w, s in nouns.items()} == master["nouns_v2"]


def test_legacy_noun_v1_training_and_news(master):
    extractor = LRNounExtractor(verbose=False)
    extractor.train(master["sents"], min_noun_frequency=2)
    nouns = extractor.extract(min_noun_frequency=2)
    assert extractor.is_trained
    assert all(len(s) == 3 for s in nouns.values())
    assert NewsNounExtractor(verbose=False).train_extract(master["sents"], min_frequency=2) is not None


def test_master_bigram(master):
    extractor = Bigram(min_frequency=1, verbose=False)
    extractor.train(master["sents"])
    assert {str(w): list(s) for w, s in extractor.extract().items()} == master["bigrams"]


def test_master_token_layouts(master):
    text = "한국어를 공부합니다"
    scores = {"한국어": 1.0}
    assert [list(s) for s in LTokenizer(scores).tokenize(text, flatten=False)] == master["l"]
    tokens = MaxScoreTokenizer(scores).tokenize(text, flatten=False)
    assert [[list(s) for s in group] for group in tokens] == master["max"]
    assert NounLMatchTokenizer({"한국어"}).tokenize(text) == master["noun_l"]


def test_line_corpus(master, tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("\n".join(master["sents"][:2]))
    assert list(DoublespaceLineCorpus(str(path))) == master["sents"][:2]
    assert len(DoublespaceLineCorpus(str(path))) == 2


def test_public_eojeol_counter_legacy_calls(master):
    counter = EojeolCounter(master["sents"])
    initial = counter.count_sum
    graph = counter.to_lrgraph(l_max_length=10, r_max_length=9)
    assert graph.get_r("한국어")
    nouns = LRNounExtractor_v2(verbose=False).train_extract(counter, min_noun_frequency=2)
    assert {w: list(score) for w, score in nouns.items()} == master["nouns_v2"]
    counter.remove_covered_eojeols({"한국어를"})
    assert counter.coverage == 1 - counter.num_of_uncovered_eojeols / initial
    assert "한국어를" not in counter.get_uncovered_eojeols()
