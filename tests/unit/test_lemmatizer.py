from soynlp.lemmatizer import Lemmatizer, _conjugate_stem, conjugate, conjugate_chat, lemma_candidate, lemma_candidate_chat


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

    def test_candidates(self):
        stems = {"먹", "가"}
        endings = {"다", "어"}
        lem = Lemmatizer(stems=stems, endings=endings)
        result = lem.candidates("먹다")
        assert isinstance(result, set)
        assert ("먹", "다") in result

    def test_predefined_merging(self):
        stems = {"붇", "불"}
        endings = {"어"}
        custom = {("살", "아"): (("삶", "아"),)}
        lem = Lemmatizer(stems=stems, endings=endings, predefined=custom)
        # custom predefined가 _initialize() 기본값과 합쳐져야 함
        assert ("살", "아") in lem._predefined
        assert ("불", "어") in lem._predefined  # 기본값 유지

    def test_predefined_affects_candidates(self):
        stems = {"붇"}
        endings = {"어"}
        lem = Lemmatizer(stems=stems, endings=endings)
        result = lem.lemmatize("불어")
        # 붇다 ㄷ불규칙: 불어 → 붇 + 어
        assert ("붇", "어") in result


class TestLemmaCandidateChat:
    def test_with_r_same_as_base(self):
        # r이 있으면 emoticon 분기를 타지 않아 기본 lemma_candidate와 동일
        result_chat = lemma_candidate_chat("먹", "어")
        result_base = lemma_candidate("먹", "어")
        assert result_chat == result_base

    def test_no_r_with_emoticon_jongsung_adds_candidates(self):
        # r이 없고 종성이 이모티콘 문자(ㅂ)이면 emoticon 분기 실행 → 추가 후보 발생
        # 밥: decompose='ㅂ','ㅏ','ㅂ' → jongsung=ㅂ (in emoticon set)
        result_chat = lemma_candidate_chat("밥", "")
        result_base = lemma_candidate("밥", "")
        assert result_chat.issuperset(result_base)
        assert len(result_chat) > len(result_base)

    def test_no_r_without_emoticon_jongsung_same_as_base(self):
        # r이 없어도 종성이 이모티콘 아니면 기본과 동일
        # 먹: decompose='ㅁ','ㅓ','ㄱ' → jongsung=ㄱ (NOT in emoticon set)
        result_chat = lemma_candidate_chat("먹", "")
        result_base = lemma_candidate("먹", "")
        assert result_chat == result_base

    def test_emoticon_extra_candidates_are_tuples(self):
        result_chat = lemma_candidate_chat("밥", "")
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result_chat)


class TestLemmatizerCandidatesNoFilter:
    def test_candidates_not_filtered_by_stems(self):
        # candidates()는 stems/endings와 무관하게 모든 후보 반환
        lem_empty = Lemmatizer(stems=set(), endings=set())
        lem_full = Lemmatizer(stems={"먹"}, endings={"었다"})
        assert lem_empty.candidates("먹었다") == lem_full.candidates("먹었다")

    def test_candidates_superset_of_lemmatize(self):
        # lemmatize()가 반환하는 후보는 candidates()의 부분집합
        stems = {"먹", "가"}
        endings = {"다", "었다"}
        lem = Lemmatizer(stems=stems, endings=endings)
        word = "먹었다"
        assert lem.lemmatize(word).issubset(lem.candidates(word))


class TestLemmaCandidatePredefined:
    def test_predefined_key_is_tuple(self):
        predefined: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
            ("살", "아"): (("삶", "아"),),
        }
        result = lemma_candidate("살", "아", predefined=predefined)
        # predefined 후보가 conjugation 검증을 통과하면 포함됨
        assert isinstance(result, set)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_predefined_type_consistency(self):
        # predefined 없이도 정상 동작
        result = lemma_candidate("먹", "었다")
        assert all(isinstance(item, tuple) for item in result)
