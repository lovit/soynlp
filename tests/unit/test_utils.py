import json

import pytest

from soynlp.core.lrgraph import LRGraph
from soynlp.utils import CorpusLoader, EojeolCounter


def test_lrgraph_construct():
    sents = ["이것은 예문이다", "이것도 예문이고", "저것은 예문이다"]
    lrgraph = LRGraph.from_sents(sents)

    assert lrgraph._lr == {
        "예": {"문이고": 1, "문이다": 2},
        "예문": {"이고": 1, "이다": 2},
        "예문이": {"고": 1, "다": 2},
        "예문이고": {"": 1},
        "예문이다": {"": 2},
        "이": {"것도": 1, "것은": 1},
        "이것": {"도": 1, "은": 1},
        "이것도": {"": 1},
        "이것은": {"": 1},
        "저": {"것은": 1},
        "저것": {"은": 1},
        "저것은": {"": 1},
    }

    assert lrgraph._rl == {
        "것도": {"이": 1},
        "것은": {"이": 1, "저": 1},
        "고": {"예문이": 1},
        "다": {"예문이": 2},
        "도": {"이것": 1},
        "문이고": {"예": 1},
        "문이다": {"예": 2},
        "은": {"이것": 1, "저것": 1},
        "이고": {"예문": 1},
        "이다": {"예문": 2},
    }

    lrgraph = LRGraph.from_sents(sents, max_l_length=3)

    assert lrgraph._lr == {
        "예": {"문이고": 1, "문이다": 2},
        "예문": {"이고": 1, "이다": 2},
        "예문이": {"고": 1, "다": 2},
        "이": {"것도": 1, "것은": 1},
        "이것": {"도": 1, "은": 1},
        "이것도": {"": 1},
        "이것은": {"": 1},
        "저": {"것은": 1},
        "저것": {"은": 1},
        "저것은": {"": 1},
    }

    assert lrgraph._rl == {
        "것도": {"이": 1},
        "것은": {"이": 1, "저": 1},
        "고": {"예문이": 1},
        "다": {"예문이": 2},
        "도": {"이것": 1},
        "문이고": {"예": 1},
        "문이다": {"예": 2},
        "은": {"이것": 1, "저것": 1},
        "이고": {"예문": 1},
        "이다": {"예문": 2},
    }


def test_lrgraph_add_pair():
    lrgraph = LRGraph({}, max_l_length=3, max_r_length=3)
    lrgraph.add_lr_pair("abc", "de")
    assert lrgraph._lr == {"abc": {"de": 1}}
    lrgraph.add_lr_pair("abcd", "de")
    assert lrgraph._lr == {"abc": {"de": 1}}


def test_lrgraph_add_eojeol():
    lrgraph = LRGraph({}, max_l_length=3, max_r_length=3)
    lrgraph.add_eojeol("abcde")
    assert lrgraph._lr == {"ab": {"cde": 1}, "abc": {"de": 1}}
    lrgraph.add_eojeol("abcde", frequency=3)
    assert lrgraph._lr == {"ab": {"cde": 4}, "abc": {"de": 4}}

    lrgraph = LRGraph({}, max_l_length=3, max_r_length=4)
    lrgraph.add_eojeol("abcde")
    assert lrgraph._lr == {"a": {"bcde": 1}, "ab": {"cde": 1}, "abc": {"de": 1}}
    lrgraph.add_eojeol("abcde", frequency=3)
    assert lrgraph._lr == {"a": {"bcde": 4}, "ab": {"cde": 4}, "abc": {"de": 4}}


def test_lrgraph_remove_lr_pair():
    lrgraph = LRGraph({}, max_l_length=3, max_r_length=4)
    lrgraph.add_eojeol("abcde", frequency=4)
    assert lrgraph._lr == {"a": {"bcde": 4}, "ab": {"cde": 4}, "abc": {"de": 4}}
    assert lrgraph._rl == {"bcde": {"a": 4}, "cde": {"ab": 4}, "de": {"abc": 4}}
    lrgraph.remove_lr_pair("ab", "cde", 3)
    assert lrgraph._lr == {"a": {"bcde": 4}, "ab": {"cde": 1}, "abc": {"de": 4}}
    assert lrgraph._rl == {"bcde": {"a": 4}, "cde": {"ab": 1}, "de": {"abc": 4}}


def test_lrgraph_get_r():
    lrgraph = LRGraph({}, max_l_length=3, max_r_length=4)
    lrgraph.add_eojeol("이것은", 1)
    lrgraph.add_eojeol("이것도", 2)
    lrgraph.add_eojeol("이것이", 3)
    lrgraph.add_eojeol("이것을", 4)
    assert lrgraph.get_r("이것", topk=3) == [("을", 4), ("이", 3), ("도", 2)]
    assert lrgraph.get_r("이것", topk=2) == [("을", 4), ("이", 3)]
    assert lrgraph.get_r("이것", topk=-1) == [("을", 4), ("이", 3), ("도", 2), ("은", 1)]


def test_lrgraph_get_l():
    lrgraph = LRGraph({}, max_l_length=3, max_r_length=4)
    lrgraph.add_eojeol("너의", 1)
    lrgraph.add_eojeol("나의", 2)
    lrgraph.add_eojeol("모두의", 3)
    lrgraph.add_eojeol("시작의", 4)
    assert lrgraph.get_l("의", topk=3) == [("시작", 4), ("모두", 3), ("나", 2)]
    assert lrgraph.get_l("의", topk=2) == [("시작", 4), ("모두", 3)]
    assert lrgraph.get_l("의", topk=-1) == [("시작", 4), ("모두", 3), ("나", 2), ("너", 1)]


def test_eojeol_counter():
    sents = ["이것은 어절 입니다", "이것은 예문 입니다", "이것도 예문 이고요"]
    eojeol_counter = EojeolCounter(sents=sents)
    assert sorted(eojeol_counter.items()) == [
        ("어절", 1),
        ("예문", 2),
        ("이것도", 1),
        ("이것은", 2),
        ("이고요", 1),
        ("입니다", 2),
    ]

    lrgraph = eojeol_counter.to_lrgraph()
    assert lrgraph.get_r("이것") == [("은", 2), ("도", 1)]

    eojeol_counter.remove_eojeols("이것은")
    assert sorted(eojeol_counter.items()) == [
        ("어절", 1),
        ("예문", 2),
        ("이것도", 1),
        ("이고요", 1),
        ("입니다", 2),
    ]

    eojeol_counter.remove_eojeols({"이것도", "입니다"})
    assert sorted(eojeol_counter.items()) == [("어절", 1), ("예문", 2), ("이고요", 1)]


def test_eojeol_counter_with_dicts():
    sents = [{"text": "이것은 어절 입니다"}, {"text": "이것은 예문 입니다"}, {"text": "이것도 예문 이고요"}]
    eojeol_counter = EojeolCounter(sents=sents)
    assert sorted(eojeol_counter.items()) == [
        ("어절", 1),
        ("예문", 2),
        ("이것도", 1),
        ("이것은", 2),
        ("이고요", 1),
        ("입니다", 2),
    ]


# --- CorpusLoader tests ---


@pytest.fixture
def jsonl_file(tmp_path):
    path = tmp_path / "corpus.jsonl"
    lines = [
        json.dumps({"text": "이것은 예문입니다"}, ensure_ascii=False),
        json.dumps({"text": "두번째 문장입니다"}, ensure_ascii=False),
        json.dumps({"text": "세번째 문장이에요"}, ensure_ascii=False),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


@pytest.fixture
def text_file(tmp_path):
    path = tmp_path / "corpus.txt"
    lines = ["이것은 예문입니다", "두번째 문장입니다", "세번째 문장이에요"]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def test_corpus_loader_jsonl(jsonl_file):
    loader = CorpusLoader(jsonl_file, format="jsonl")
    items = list(loader)
    assert len(items) == 3
    assert items[0] == {"text": "이것은 예문입니다"}
    assert items[1] == {"text": "두번째 문장입니다"}


def test_corpus_loader_text(text_file):
    loader = CorpusLoader(text_file, format="text")
    items = list(loader)
    assert len(items) == 3
    assert items[0] == {"text": "이것은 예문입니다"}


def test_corpus_loader_text_custom_key(text_file):
    loader = CorpusLoader(text_file, format="text", text_key="content")
    items = list(loader)
    assert items[0] == {"content": "이것은 예문입니다"}


def test_corpus_loader_len(jsonl_file):
    loader = CorpusLoader(jsonl_file, format="jsonl")
    assert len(loader) == 3


def test_corpus_loader_invalid_format(jsonl_file):
    with pytest.raises(ValueError, match="format must be"):
        CorpusLoader(jsonl_file, format="csv")


def test_corpus_loader_file_not_found():
    with pytest.raises(FileNotFoundError):
        CorpusLoader("/nonexistent/path.jsonl", format="jsonl")


def test_corpus_loader_skips_empty_lines(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("첫번째\n\n두번째\n\n", encoding="utf-8")
    loader = CorpusLoader(str(path), format="text")
    items = list(loader)
    assert len(items) == 2
    assert len(loader) == 2


def test_corpus_loader_with_eojeol_counter(jsonl_file):
    loader = CorpusLoader(jsonl_file, format="jsonl")
    counter = EojeolCounter(sents=loader)
    assert counter["이것은"] == 1
    assert counter["예문입니다"] == 1


# --- EojeolCounter multiprocessing tests ---

_MULTI_SENTS = [
    "아이오아이가 평가단에게 높은 점수를 받았습니다",
    "아이오아이는 아이돌 그룹입니다",
    "자연어처리는 어렵습니다",
    "자연어처리를 공부합니다",
    "명사추출은 중요한 작업입니다",
] * 100


def test_eojeol_counter_multi_equals_single():
    """n_workers=4 결과가 단일 프로세스 결과와 동일하다."""
    single = EojeolCounter(sents=_MULTI_SENTS, n_workers=1)
    multi = EojeolCounter(sents=_MULTI_SENTS, n_workers=4)
    assert dict(single.items()) == dict(multi.items())


def test_eojeol_counter_auto_workers():
    """n_workers=-1이면 CPU 코어 수를 자동 사용한다."""
    single = EojeolCounter(sents=_MULTI_SENTS, n_workers=1)
    auto = EojeolCounter(sents=_MULTI_SENTS, n_workers=-1)
    assert dict(single.items()) == dict(auto.items())


def test_eojeol_counter_multi_with_min_count():
    """n_workers > 1일 때 min_count 필터가 올바르게 적용된다."""
    single = EojeolCounter(sents=_MULTI_SENTS, min_count=5, n_workers=1)
    multi = EojeolCounter(sents=_MULTI_SENTS, min_count=5, n_workers=4)
    assert dict(single.items()) == dict(multi.items())
