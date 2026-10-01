import itertools
from pathlib import Path

import pytest

from soynlp.core import LRGraph
from soynlp.pos import ChatPOSExtractor, NewsPOSExtractor
from soynlp.postagger import Dictionary, LRMaxScoreTagger, POSExtractor
from soynlp.utils import CorpusLoader
from soynlp.vectorizer import BaseVectorizer


def test_graph_counter_roundtrip_and_snapshot():
    graph = LRGraph.from_sents(["학교에는 학교에는 학생이"])
    assert dict(graph.to_EojeolCounter().items()) == {"학교에는": 2, "학생이": 1}
    graph.remove_eojeol("학교에는", 1)
    assert dict(graph.to_EojeolCounter().items())["학교에는"] == 1
    assert dict(graph.to_EojeolCounter(reset_lrgraph=True).items())["학교에는"] == 2
    assert dict(graph.to_EojeolCounter().items())["학교에는"] == 1


@pytest.fixture(scope="module")
def news_sentences():
    path = Path(__file__).parents[1] / "integration/data/news-text/2016-10-20.jsonl"
    return [row["text"] for row in itertools.islice(CorpusLoader(str(path)), 300)]


@pytest.mark.parametrize("extractor_class", [NewsPOSExtractor, ChatPOSExtractor])
@pytest.mark.parametrize("extract_eomi", [True, False])
def test_pos_extraction_real_corpus(extractor_class, extract_eomi, news_sentences):
    result = extractor_class(verbose=False, extract_eomi=extract_eomi).train_extract(news_sentences)
    assert {"Noun", "Verb", "Adjective", "Eomi"} <= result.keys()
    assert result["Noun"]


def test_composite_pos_extractor(news_sentences):
    nouns, predicators = POSExtractor(verbose=False).extract(news_sentences)
    assert nouns
    assert len(predicators) == 2


def test_lr_tagger_dictionary_and_unknown_spans(tmp_path):
    dictionary = Dictionary({"Noun": {"한국어"}, "Josa": {"를"}})
    tagger = LRMaxScoreTagger(sents=["한국어를 공부합니다"] * 5, dictionary=dictionary)
    tagged = tagger.pos("한국어를 미등록어")
    assert tagged[0] == ("한국어", "Noun")
    assert "".join(word for word, tag in tagged) == "한국어를미등록어"
    tagger.add_words_into_dictionary({"공부"}, "Noun")
    tagger.save_domain_dictionary(str(tmp_path))
    restored = LRMaxScoreTagger(domain_dictionary_folders=str(tmp_path), use_base_dictionary=False)
    assert restored.dictionary.pos_L("공부") == "Noun"


def test_empty_vectorization_and_dictionary():
    vectorizer = BaseVectorizer(verbose=False)
    assert vectorizer.fit_transform([]).shape == (0, 0)
    assert Dictionary({}).max_length == 0
