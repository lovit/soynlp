import tempfile

import pytest

from soynlp.postagger import (
    LR,
    Dictionary,
    EojeolTemplateMatcher,
    MorphTag,
    SimpleEojeolEvaluator,
    SimpleTagger,
)


@pytest.fixture
def sample_dict():
    pos_dict = {
        "Noun": {"나", "너", "사과", "학교", "학생"},
        "Verb": {"먹다", "가다"},
        "Josa": {"는", "를", "이", "가", "에서"},
        "Adverb": {"빨리", "천천히"},
    }
    return Dictionary(pos_dict)


class TestDictionary:
    def test_get_pos(self, sample_dict):
        assert "Noun" in sample_dict.get_pos("사과")
        assert sample_dict.get_pos("없는단어") == []

    def test_word_is_tag(self, sample_dict):
        assert sample_dict.word_is_tag("사과", "Noun") is True
        assert sample_dict.word_is_tag("사과", "Verb") is False

    def test_add_words(self, sample_dict):
        sample_dict.add_words("Noun", {"컴퓨터"})
        assert sample_dict.word_is_tag("컴퓨터", "Noun") is True

    def test_add_words_new_tag_requires_force(self, sample_dict):
        with pytest.raises(ValueError):
            sample_dict.add_words("NewTag", {"단어"})
        sample_dict.add_words("NewTag", {"단어"}, force=True)
        assert sample_dict.word_is_tag("단어", "NewTag") is True

    def test_remove_words(self, sample_dict):
        sample_dict.remove_words("Noun", {"사과"})
        assert sample_dict.word_is_tag("사과", "Noun") is False

    def test_remove_tag(self, sample_dict):
        sample_dict.remove_words("Adverb")
        assert "Adverb" not in sample_dict.pos_dict

    def test_save_load(self, sample_dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            sample_dict.save(f.name)
            loaded = Dictionary(f.name)
            assert loaded.word_is_tag("사과", "Noun") is True
            assert loaded.max_length == sample_dict.max_length


class TestLRDataclass:
    def test_fields(self):
        lr = LR(l="사과", l_tag="Noun", r="를", r_tag="Josa", b=0, m=2, e=3)
        assert lr.l == "사과"
        assert lr.l_tag == "Noun"
        assert lr.r == "를"
        assert lr.r_tag == "Josa"
        assert lr.b == 0
        assert lr.m == 2
        assert lr.e == 3

    def test_frozen(self):
        lr = LR(l="사과", l_tag="Noun", r="를", r_tag="Josa", b=0, m=2, e=3)
        with pytest.raises(AttributeError):
            lr.l = "바나나"  # type: ignore[misc]


class TestEojeolTemplateMatcher:
    def test_generate_single(self, sample_dict):
        matcher = EojeolTemplateMatcher(sample_dict)
        candidates = matcher.generate("사과")
        assert len(candidates) >= 1
        assert any(c[0].l == "사과" and c[0].l_tag == "Noun" for c in candidates)

    def test_generate_noun_josa(self, sample_dict):
        matcher = EojeolTemplateMatcher(sample_dict)
        candidates = matcher.generate("사과를")
        found = any(len(c) == 1 and c[0].l == "사과" and c[0].r == "를" and c[0].r_tag == "Josa" for c in candidates)
        assert found


class TestSimpleEojeolEvaluator:
    def test_evaluate(self):
        evaluator = SimpleEojeolEvaluator()
        candidate = [LR("사과", "Noun", "", None, 0, 2, 2)]
        score = evaluator.evaluate(candidate)
        assert isinstance(score, float)

    def test_select_best(self):
        evaluator = SimpleEojeolEvaluator()
        c1 = [LR("사과", "Noun", "", None, 0, 2, 2)]
        c2 = [LR("사과를", None, "", None, 0, 3, 3)]
        best = evaluator.select_best([c1, c2])
        assert best is not None


class TestMorphTag:
    def test_fields(self):
        mt = MorphTag(surface="사과", tag="Noun")
        assert mt.surface == "사과"
        assert mt.tag == "Noun"

    def test_unknown_tag(self):
        mt = MorphTag(surface="모름", tag=None)
        assert mt.tag is None

    def test_frozen(self):
        mt = MorphTag(surface="사과", tag="Noun")
        with pytest.raises(AttributeError):
            mt.surface = "바나나"  # type: ignore[misc]


class TestSimpleTagger:
    def test_tag_returns_morph_tag_list(self, sample_dict):
        matcher = EojeolTemplateMatcher(sample_dict)
        evaluator = SimpleEojeolEvaluator()
        tagger = SimpleTagger(matcher, evaluator)
        result = tagger.tag("사과")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(m, MorphTag) for m in result)

    def test_tag_noun_josa(self, sample_dict):
        from typing import cast

        matcher = EojeolTemplateMatcher(sample_dict)
        evaluator = SimpleEojeolEvaluator()
        tagger = SimpleTagger(matcher, evaluator)
        result = cast(list[MorphTag], tagger.tag("사과를"))
        surfaces = [m.surface for m in result]
        assert "사과" in surfaces or "사과를" in surfaces

    def test_tag_not_flatten(self, sample_dict):
        from typing import cast

        matcher = EojeolTemplateMatcher(sample_dict)
        evaluator = SimpleEojeolEvaluator()
        tagger = SimpleTagger(matcher, evaluator)
        result = cast(list[list[MorphTag]], tagger.tag("나는 학교에서", flatten=False))
        assert isinstance(result, list)
        assert all(isinstance(eojeol, list) for eojeol in result)
        assert all(isinstance(m, MorphTag) for eojeol in result for m in eojeol)
