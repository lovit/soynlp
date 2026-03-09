import pytest

from soynlp.core.lrgraph import LRGraph
from soynlp.noun.postprocessing import expand_suffix_nouns


def _make_lrgraph(eojeols: list[str]) -> LRGraph:
    # from_sents 사용: _lr_origin 이 올바르게 초기화됨
    return LRGraph.from_sents(eojeols)


class TestExpandSuffixNouns:
    """expand_suffix_nouns() 의 동작을 검증한다."""

    def test_high_medium_suffix_added(self):
        """생산성 높음/중간 접미사(화, 성, 학 등)가 붙은 파생어가 추가된다."""
        nouns: dict[str, tuple[int, float]] = {"디지털": (100, 0.9)}
        eojeols = ["디지털화는"] * 50 + ["디지털화의"] * 30
        lrgraph = _make_lrgraph(eojeols)

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert "디지털화" in result
        assert result["디지털화"][1] == 1.0

    def test_already_extracted_noun_not_overwritten(self):
        """이미 명사로 추출된 경우에는 추가하지 않는다."""
        nouns: dict[str, tuple[int, float]] = {
            "자동화": (200, 0.95),
            "자동": (100, 0.8),
        }
        eojeols = ["자동화는"] * 50
        lrgraph = _make_lrgraph(eojeols)

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert result["자동화"] == (200, 0.95)  # 기존 값 유지

    def test_low_suffix_with_noun_length_2_added(self):
        """길이 2 이상의 명사 + 생산성 낮음 접미사(꾼, 쟁이, 질)가 추가된다."""
        nouns: dict[str, tuple[int, float]] = {"사기": (100, 0.9)}
        eojeols = ["사기꾼이"] * 50 + ["사기꾼을"] * 30
        lrgraph = _make_lrgraph(eojeols)

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert "사기꾼" in result
        assert result["사기꾼"][1] == 1.0

    def test_low_suffix_with_noun_length_1_not_added(self):
        """길이 1인 명사 + 생산성 낮음 접미사는 추가하지 않는다."""
        nouns: dict[str, tuple[int, float]] = {"일": (100, 0.9)}
        eojeols = ["일꾼은"] * 50
        lrgraph = _make_lrgraph(eojeols)

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert "일꾼" not in result

    def test_suffix_noun_not_in_lrgraph_not_added(self):
        """코퍼스에 등장하지 않는 파생어는 추가하지 않는다."""
        nouns: dict[str, tuple[int, float]] = {"언어": (100, 0.9)}
        lrgraph = _make_lrgraph([])  # 빈 lrgraph

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert "언어학" not in result

    def test_min_noun_frequency_filter(self):
        """min_noun_frequency 미만인 파생어는 추가하지 않는다."""
        nouns: dict[str, tuple[int, float]] = {"언어": (100, 0.9)}
        # 언어학이 2번만 등장
        eojeols = ["언어학은"] * 2
        lrgraph = _make_lrgraph(eojeols)

        result_freq3 = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=3)
        result_freq1 = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert "언어학" not in result_freq3
        assert "언어학" in result_freq1

    def test_frequency_correctly_assigned(self):
        """추가된 파생어의 frequency 는 lrgraph 의 총 빈도로 설정된다."""
        nouns: dict[str, tuple[int, float]] = {"경제": (100, 0.9)}
        eojeols = ["경제학은"] * 30 + ["경제학의"] * 20 + ["경제학을"] * 10
        lrgraph = _make_lrgraph(eojeols)

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert "경제학" in result
        freq, score = result["경제학"]
        assert freq == 60  # 30 + 20 + 10
        assert score == 1.0

    def test_josa_suffix_excluded(self):
        """josaset에 포함된 '가'는 접미사 목록에서 제외된다 (오분류 방지)."""
        nouns: dict[str, tuple[int, float]] = {"학생": (100, 0.9)}
        eojeols = ["학생가는"] * 50  # '학생가'가 코퍼스에 등장해도
        lrgraph = _make_lrgraph(eojeols)

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert "학생가" not in result  # josa '가'이므로 추가되지 않아야 함

    @pytest.mark.parametrize(
        "suffix",
        ["화", "성", "적", "자", "들", "상", "기", "학", "론", "계", "형", "주의", "권", "력", "감", "관", "제"],
    )
    def test_high_medium_suffixes_supported(self, suffix: str):
        """생산성 높음/중간 접미사 전체가 지원된다."""
        noun = "기반"
        nouns: dict[str, tuple[int, float]] = {noun: (100, 0.9)}
        candidate = noun + suffix
        eojeols = [candidate + "은"] * 10
        lrgraph = _make_lrgraph(eojeols)

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert candidate in result, f"'{candidate}' 이 결과에 없음 (suffix='{suffix}')"

    @pytest.mark.parametrize("suffix", ["꾼", "쟁이", "질"])
    def test_low_suffixes_supported(self, suffix: str):
        """생산성 낮음 접미사(꾼, 쟁이, 질)가 Noun(길이>=2) 조건으로 지원된다."""
        noun = "도둑"
        nouns: dict[str, tuple[int, float]] = {noun: (100, 0.9)}
        candidate = noun + suffix
        eojeols = [candidate + "은"] * 10
        lrgraph = _make_lrgraph(eojeols)

        result = expand_suffix_nouns(nouns, lrgraph, min_noun_frequency=1)

        assert candidate in result, f"'{candidate}' 이 결과에 없음 (suffix='{suffix}')"


class TestExpandSuffixNounsDisabled:
    """expand_suffixes=False 옵션이 올바르게 동작함을 검증한다."""

    def test_expand_suffixes_false_skips_expansion(self):
        """postprocessing에서 expand_suffixes=False 이면 파생어를 추가하지 않는다."""
        from soynlp.noun.lr import postprocessing

        nouns: dict[str, tuple[int, float]] = {"언어": (100, 0.9)}
        eojeols = ["언어학은"] * 50 + ["언어학의"] * 30
        lrgraph = LRGraph.from_sents(eojeols)

        result_with = postprocessing(nouns.copy(), lrgraph, set(), 0.3, False, expand_suffixes=True, min_noun_frequency=1)
        result_without = postprocessing(nouns.copy(), lrgraph, set(), 0.3, False, expand_suffixes=False, min_noun_frequency=1)

        assert "언어학" in result_with
        assert "언어학" not in result_without
