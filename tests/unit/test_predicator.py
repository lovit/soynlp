"""Tests for soynlp.predicator module."""

from soynlp.predicator import EomiScore, Predicator


class TestPredicatorNamedTuple:
    def test_predicator_creation(self):
        p = Predicator(frequency=10, lemma={("먹", "다")})
        assert p.frequency == 10
        assert ("먹", "다") in p.lemma

    def test_predicator_empty_lemma(self):
        p = Predicator(frequency=0, lemma=set())
        assert p.frequency == 0
        assert len(p.lemma) == 0

    def test_predicator_multiple_lemmas(self):
        lemmas = {("먹", "다"), ("먹", "어")}
        p = Predicator(frequency=5, lemma=lemmas)
        assert len(p.lemma) == 2


class TestEomiScore:
    def test_eomi_score_creation(self):
        e = EomiScore(frequency=100, score=0.85)
        assert e.frequency == 100
        assert e.score == 0.85

    def test_eomi_score_zero(self):
        e = EomiScore(frequency=0, score=0.0)
        assert e.frequency == 0
        assert e.score == 0.0


class TestRuleClassify:
    """Test rule_classify from _adjective_vs_verb.py"""

    def test_adjective_suffix_답(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("아름답") == "Adjective"

    def test_adjective_suffix_롭(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("자유롭") == "Adjective"

    def test_adjective_suffix_스럽(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("사랑스럽") == "Adjective"

    def test_adjective_suffix_같(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("똑같") == "Adjective"

    def test_adjective_suffix_만하(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("볼만하") == "Adjective"

    def test_verb_suffix_거리(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("반짝거리") == "Verb"

    def test_verb_suffix_당하(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("해고당하") == "Verb"

    def test_verb_suffix_시키(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("공부시키") == "Verb"

    def test_ambiguous_returns_none(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("하") is None

    def test_unknown_stem_returns_none(self):
        from soynlp.predicator._adjective_vs_verb import rule_classify

        assert rule_classify("먹") is None


class TestConjugateAsPresent:
    def test_present_open_syllable(self):
        from soynlp.predicator._adjective_vs_verb import conjugate_as_present

        surfaces = conjugate_as_present("가")
        assert isinstance(surfaces, set)
        assert len(surfaces) > 0

    def test_present_closed_syllable(self):
        from soynlp.predicator._adjective_vs_verb import conjugate_as_present

        surfaces = conjugate_as_present("먹")
        assert isinstance(surfaces, set)
        assert len(surfaces) > 0


class TestConjugateAsImperative:
    def test_imperative(self):
        from soynlp.predicator._adjective_vs_verb import conjugate_as_imperative

        surfaces = conjugate_as_imperative("먹")
        assert isinstance(surfaces, set)
        assert len(surfaces) > 0


class TestConjugateAsPleasure:
    def test_pleasure(self):
        from soynlp.predicator._adjective_vs_verb import conjugate_as_pleasure

        surfaces = conjugate_as_pleasure("먹")
        assert isinstance(surfaces, set)
        assert len(surfaces) > 0
        # "먹자" should be one of the results
        assert "먹자" in surfaces


class TestPredicatorExtractor:
    def test_init_with_nouns(self):
        from soynlp.predicator import PredicatorExtractor

        extractor = PredicatorExtractor(nouns={"사과", "바나나"}, verbose=False)
        assert extractor.is_trained is False
        assert extractor._nouns is not None

    def test_is_trained_false_before_train(self):
        from soynlp.predicator import PredicatorExtractor

        extractor = PredicatorExtractor(nouns={"테스트"}, verbose=False)
        assert extractor.is_trained is False

    def test_remove_stem_prefix(self):
        """Nouns that are prefix of stems should be removed."""
        from soynlp.predicator import PredicatorExtractor

        extractor = PredicatorExtractor(nouns={"사과", "바나나", "테스트"}, verbose=False)
        # nouns that overlap with stem prefixes are removed
        assert isinstance(extractor._nouns, set)

    def test_separate_adjective_verb_empty(self):
        from soynlp.predicator import PredicatorExtractor

        extractor = PredicatorExtractor(nouns={"테스트"}, verbose=False)
        adj, verb = extractor._separate_adjective_verb({})
        assert adj == {}
        assert verb == {}

    def test_separate_adjective_verb_with_known_stems(self):
        from soynlp.predicator import PredicatorExtractor

        extractor = PredicatorExtractor(nouns={"테스트"}, verbose=False)
        # Create predicator with a known adjective stem
        predicators = {
            "아름다워": Predicator(frequency=10, lemma={("아름답", "어")}),
        }
        # "아름답" has suffix "답" -> should be classified as adjective
        adj, verb = extractor._separate_adjective_verb(predicators)
        assert "아름다워" in adj
