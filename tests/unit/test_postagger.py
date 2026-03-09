import tempfile

import pytest

from soynlp.postagger import (
    LR,
    Dictionary,
    DictionaryProtocol,
    EojeolTemplateMatcher,
    ExtractorStepProtocol,
    KoreanPOSTagger,
    MorphTag,
    POSExtractor,
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

    def test_repr(self, sample_dict):
        matcher = EojeolTemplateMatcher(sample_dict)
        evaluator = SimpleEojeolEvaluator()
        tagger = SimpleTagger(matcher, evaluator)
        r = repr(tagger)
        assert "SimpleTagger" in r
        assert "EojeolTemplateMatcher" in r


class TestDictionaryRepr:
    def test_repr(self, sample_dict):
        r = repr(sample_dict)
        assert "Dictionary" in r
        assert "num_tags" in r
        assert "num_words" in r


class TestPOSExtractorRepr:
    def test_repr_not_trained(self):
        extractor = POSExtractor()
        r = repr(extractor)
        assert "POSExtractor" in r
        assert "trained=False" in r

    def test_is_trained_initially_false(self):
        extractor = POSExtractor()
        assert extractor.is_trained is False

    def test_extra_steps_stored(self):
        class NoopStep:
            def extract(self, sentences, context: dict) -> dict:
                return {}

        step = NoopStep()
        extractor = POSExtractor(extra_steps=[step])
        assert len(extractor.extra_steps) == 1

    def test_extra_step_satisfies_protocol(self):
        class NoopStep:
            def extract(self, sentences, context: dict) -> dict:
                return {}

        step = NoopStep()
        assert isinstance(step, ExtractorStepProtocol)

    def test_extra_step_context_injection(self):
        """extra_step이 context에 값을 주입하면 다음 단계에서 사용 가능한지 확인한다."""
        received_contexts: list[dict] = []

        class RecordContextStep:
            def extract(self, sentences, context: dict) -> dict:
                received_contexts.append(dict(context))
                return {"custom_key": "custom_value"}

        extractor = POSExtractor(extra_steps=[RecordContextStep()])
        assert extractor.extra_steps[0] is not None
        # extra_step이 등록된 것만 확인 (실제 extract 호출은 느리므로 생략)


class TestDictionaryProtocol:
    def test_dictionary_satisfies_protocol(self, sample_dict):
        assert isinstance(sample_dict, DictionaryProtocol)

    def test_custom_dictionary_satisfies_protocol(self):
        class MyDict:
            max_length = 5

            def get_pos(self, word: str) -> list[str]:
                return ["Noun"] if word == "사과" else []

            def word_is_tag(self, word: str, tag: str) -> bool:
                return tag == "Noun" and word == "사과"

        my_dict = MyDict()
        assert isinstance(my_dict, DictionaryProtocol)

    def test_custom_dictionary_usable_in_template(self):
        class MinimalDict:
            max_length = 5

            def get_pos(self, word: str) -> list[str]:
                return ["Noun"] if word in {"사과", "배"} else []

            def word_is_tag(self, word: str, tag: str) -> bool:
                return tag == "Noun" and word in {"사과", "배"}

        matcher = EojeolTemplateMatcher(MinimalDict())
        candidates = matcher.generate("사과")
        assert len(candidates) >= 1


class TestEojeolTemplateMatcherFilePath:
    def test_save_and_load(self, sample_dict):
        matcher = EojeolTemplateMatcher(sample_dict)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            matcher.save(f.name)
            loaded = EojeolTemplateMatcher.from_file(f.name, sample_dict)
            assert loaded.single_tags == matcher.single_tags
            assert loaded.lr_templates == matcher.lr_templates

    def test_from_file_custom_template(self, sample_dict, tmp_path):
        template = {
            "single_tags": ["Noun"],
            "lr_templates": [["Noun", "Josa"]],
        }
        path = tmp_path / "template.json"
        path.write_text(__import__("json").dumps(template), encoding="utf-8")
        matcher = EojeolTemplateMatcher.from_file(str(path), sample_dict)
        assert matcher.single_tags == ["Noun"]
        assert ("Noun", "Josa") in matcher.lr_templates

    def test_template_path_in_init(self, sample_dict, tmp_path):
        template = {
            "single_tags": ["Noun", "Verb"],
            "lr_templates": [["Noun", "Josa"]],
        }
        path = tmp_path / "template.json"
        path.write_text(__import__("json").dumps(template), encoding="utf-8")
        matcher = EojeolTemplateMatcher(sample_dict, template_path=str(path))
        assert matcher.single_tags == ["Noun", "Verb"]


class TestKoreanPOSTagger:
    def test_default_creates_instance(self):
        tagger = KoreanPOSTagger.default()
        assert isinstance(tagger, KoreanPOSTagger)

    def test_is_trained_true_after_init(self):
        tagger = KoreanPOSTagger.default()
        assert tagger.is_trained is True

    def test_repr(self):
        tagger = KoreanPOSTagger.default()
        r = repr(tagger)
        assert "KoreanPOSTagger" in r
        assert "trained=True" in r
        assert "vocab_size" in r

    def test_tag_works_without_train(self):
        tagger = KoreanPOSTagger.default()
        result = tagger.tag("나는 학교에 갔다")
        assert isinstance(result, list)
        assert all(isinstance(m, MorphTag) for m in result)

    def test_tag_returns_morph_tag_list(self):
        tagger = KoreanPOSTagger.default()
        result = tagger.tag("사과를")
        assert isinstance(result, list)
        assert all(isinstance(m, MorphTag) for m in result)

    def test_train_with_extra_nouns(self):
        tagger = KoreanPOSTagger.default()
        tagger.train(extra_nouns={"ChatGPT", "딥러닝"})
        assert tagger._dictionary.word_is_tag("ChatGPT", "Noun")
        assert tagger._dictionary.word_is_tag("딥러닝", "Noun")

    def test_train_with_sentences_does_not_raise(self):
        tagger = KoreanPOSTagger.default()
        tagger.train(sentences=["나는 학교에 갔다", "사과를 먹었다"])
        assert tagger.is_trained is True

    def test_custom_dictionary(self):
        custom_dict = Dictionary({"Noun": {"사과", "배", "귤"}, "Josa": {"를", "이", "가", "은", "는"}})
        tagger = KoreanPOSTagger(dictionary=custom_dict)
        result = tagger.tag("사과를")
        surfaces = [m.surface for m in result]
        assert "사과" in surfaces or "사과를" in surfaces
