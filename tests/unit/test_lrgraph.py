import os
import tempfile

import pytest

from soynlp.core.lrgraph import LRGraph, corpus_to_lrgraph


class TestLRGraph:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            LRGraph({}, max_l_length=0)
        with pytest.raises(ValueError):
            LRGraph({}, max_r_length=0)
        with pytest.raises(ValueError):
            LRGraph({}, max_l_length="a")  # type: ignore[arg-type]

    def test_bidirectional_graph(self):
        lrgraph = LRGraph({"이것": {"은": 2, "도": 1}})
        assert lrgraph.get_r("이것") == [("은", 2), ("도", 1)]
        assert lrgraph.get_l("은") == [("이것", 2)]
        assert lrgraph.get_l("도") == [("이것", 1)]

    def test_add_lr_pair(self):
        lrgraph = LRGraph({})
        lrgraph.add_lr_pair("이것", "은", 3)
        assert lrgraph.get_r("이것") == [("은", 3)]
        assert lrgraph.get_l("은") == [("이것", 3)]

    def test_add_lr_pair_respects_max_length(self):
        lrgraph = LRGraph({}, max_l_length=2, max_r_length=2)
        lrgraph.add_lr_pair("이것은", "예문입니다", 1)
        assert lrgraph.get_r("이것은") == []

    def test_add_eojeol(self):
        lrgraph = LRGraph({})
        lrgraph.add_eojeol("이것은", 1)
        assert ("것은", 1) in lrgraph.get_r("이")
        assert ("은", 1) in lrgraph.get_r("이것")

    def test_remove_lr_pair(self):
        lrgraph = LRGraph({"이것": {"은": 3}})
        lrgraph.remove_lr_pair("이것", "은", 1)
        assert lrgraph.get_r("이것") == [("은", 2)]
        lrgraph.remove_lr_pair("이것", "은", 5)
        assert lrgraph.get_r("이것") == []

    def test_remove_eojeol(self):
        lrgraph = LRGraph({"이": {"것은": 2}, "이것": {"은": 2}})
        lrgraph.remove_eojeol("이것은", 1)
        assert lrgraph.get_r("이것") == [("은", 1)]

    def test_get_r_topk(self):
        lrgraph = LRGraph({"L": {"a": 3, "b": 2, "c": 1}})
        assert len(lrgraph.get_r("L", topk=2)) == 2
        assert len(lrgraph.get_r("L", topk=-1)) == 3

    def test_get_l_topk(self):
        lrgraph = LRGraph({"a": {"R": 3}, "b": {"R": 2}, "c": {"R": 1}})
        assert len(lrgraph.get_l("R", topk=2)) == 2

    def test_freeze_and_reset(self):
        lrgraph = LRGraph({"이것": {"은": 2}})
        lrgraph.add_lr_pair("이것", "도", 1)
        assert ("도", 1) in lrgraph.get_r("이것")
        lrgraph.reset_lrgraph()
        r_items = dict(lrgraph.get_r("이것", topk=-1))
        assert "도" not in r_items

    def test_freeze_deepcopy(self):
        lrgraph = LRGraph({"이것": {"은": 2}})
        lrgraph.add_lr_pair("이것", "도", 1)
        lrgraph.freeze()
        lrgraph.add_lr_pair("이것", "이", 5)
        lrgraph.reset_lrgraph()
        r_items = dict(lrgraph.get_r("이것", topk=-1))
        assert "도" in r_items
        assert "이" not in r_items

    def test_save_and_load(self):
        lrgraph = LRGraph({"이것": {"은": 2, "도": 1, "": 3}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lrgraph.txt")
            lrgraph.save(path)
            loaded = LRGraph.load(path)
            assert dict(loaded.get_r("이것", topk=-1)) == {"은": 2, "도": 1, "": 3}

    def test_from_sents(self):
        sents = ["이것은 예문 입니다", "이것도 예문 입니다"]
        lrgraph = LRGraph.from_sents(sents)
        r_items = dict(lrgraph.get_r("이것", topk=-1))
        assert r_items["은"] == 1
        assert r_items["도"] == 1


class TestCorpusToLrgraph:
    def test_basic(self):
        texts = ["이것은 예문입니다"]
        lrgraph = corpus_to_lrgraph(texts)
        r_items = dict(lrgraph.get_r("이것", topk=-1))
        assert r_items["은"] == 1

    def test_frequency_counts_correctly(self):
        texts = ["이것은 이것은"]
        lrgraph = corpus_to_lrgraph(texts)
        r_items = dict(lrgraph.get_r("이것", topk=-1))
        assert r_items["은"] == 2

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            corpus_to_lrgraph([], l_max_length=0)
        with pytest.raises(ValueError):
            corpus_to_lrgraph([], r_max_length=-1)
