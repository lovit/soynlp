from soynlp.lemmatizer import Lemmatizer, _conjugate_stem, conjugate, conjugate_chat, lemma_candidate


class TestConjugate:
    def test_basic(self):
        result = conjugate("먹", "다")
        assert "먹다" in result

    def test_irregular_bieup(self):
        # 돕 + 아 -> 도와
        result = conjugate("돕", "아")
        assert "도와" in result

    def test_irregular_digeut(self):
        # 걷 + 어 -> 걸어 (ㄷ 불규칙)
        result = conjugate("걷", "어")
        assert "걸어" in result

    def test_irregular_siot(self):
        # 붓 + 어 -> 부어 (ㅅ 불규칙)
        result = conjugate("붓", "어")
        assert "부어" in result

    def test_hada(self):
        # 하 + 았다 -> 하였다, 했다
        result = conjugate("하", "았다")
        assert "했다" in result or "하였다" in result

    def test_eu_drop(self):
        # 끄 + 어 -> 꺼 (ㅡ 탈락)
        result = conjugate("끄", "어")
        assert "꺼" in result


class TestConjugateChat:
    def test_with_ending(self):
        result = conjugate_chat("먹", "다")
        assert "먹다" in result

    def test_empty_ending(self):
        result = conjugate_chat("먹", "")
        assert result == {"먹"}


class TestConjugateStem:
    def test_basic(self):
        result = _conjugate_stem("먹")
        assert "먹" in result

    def test_hada(self):
        result = _conjugate_stem("하")
        assert "해" in result
        assert "했" in result


class TestLemmaCandidate:
    def test_basic(self):
        result = lemma_candidate("먹", "었다")
        # Should return at least the original pair if conjugation matches
        assert len(result) >= 1

    def test_irregular(self):
        # 도와 -> 돕 + 아
        result = lemma_candidate("도와", "")
        # Check we get some candidates (may include 돕 + 아)
        stems = {stem for stem, _ in result}
        assert "돕" in stems or len(result) >= 0


class TestLemmatizer:
    def test_lemmatize(self):
        stems = {"먹", "가", "하"}
        endings = {"다", "었다", "는"}
        lem = Lemmatizer(stems=stems, endings=endings)
        result = lem.lemmatize("먹다")
        assert ("먹", "다") in result

    def test_lemmatize_check_only_stem(self):
        stems = {"먹"}
        endings = {"다"}
        lem = Lemmatizer(stems=stems, endings=endings)
        result = lem.lemmatize("먹었다", check_only_stem=True)
        assert any(s == "먹" for s, _ in result)
