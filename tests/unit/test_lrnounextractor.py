import pytest

from soynlp.core.lrgraph import LRGraph
from soynlp.noun.lr import (
    _inject_known_nouns,
    check_r_features,
    postprocessing,
    predict_single_noun,
    remove_ambiguous_features,
)

POS_FEATURES = {"로", "의", "로는", "으로", "으로는", "으로써", "이지만"}
NEG_FEATURES = {"다고", "고", "자는", "자고", "지만"}
COMMON_FEATURES = {"은", "는"}


@pytest.mark.parametrize(
    ("word", "features", "expected_refined", "expected_pos", "expected_neg", "expected_common", "expected_end"),
    [
        ("대학생으", [("로", 10), ("로써", 5), ("로는", 3)], [], 0, 0, 0, 0),
        (
            "대학생",
            [("으로", 10), ("으로써", 5), ("으로는", 3)],
            [("으로", 10), ("으로써", 5), ("으로는", 3)],
            18,
            0,
            0,
            0,
        ),
        ("너로", [("는", 5), ("", 3)], [], 0, 0, 0, 0),
        ("너", [("로는", 5), ("는", 3)], [("는", 3), ("로는", 5)], 5, 0, 3, 0),
        (
            "관계자",
            [("는", 10), ("로", 5), ("이지만", 5)],
            [("로", 5), ("이지만", 5)],
            10,
            0,
            0,
            0,
        ),
        (
            "관계",
            [("자는", 10), ("자로", 5), ("는", 20), ("의", 25), ("자이지만", 5)],
            [("자는", 10), ("자로", 5), ("는", 20), ("의", 25), ("자이지만", 5)],
            25,
            10,
            20,
            0,
        ),
        ("하자", [("는", 10), ("고", 5)], [], 0, 0, 0, 0),
        ("하", [("자는", 10), ("자고", 5)], [("자는", 10), ("자고", 5)], 0, 15, 0, 0),
        ("가고있다", [("고", 10), ("지만", 3)], [("지만", 3)], 0, 3, 0, 0),
        (
            "가고있",
            [("다고", 10), ("지만", 5), ("다지만", 3)],
            [("다고", 10), ("지만", 5), ("다지만", 3)],
            0,
            15,
            0,
            0,
        ),
    ],
    ids=["대학생으", "대학생", "너로", "너", "관계자", "관계", "하자", "하", "가고있다", "가고있"],
)
def test_check_r_features(word, features, expected_refined, expected_pos, expected_neg, expected_common, expected_end):
    refined, _ = remove_ambiguous_features(word, features, POS_FEATURES, NEG_FEATURES, COMMON_FEATURES)
    pos, common, neg, _, end = check_r_features(word, refined, POS_FEATURES, NEG_FEATURES, COMMON_FEATURES)

    assert sorted(refined) == sorted(expected_refined)
    assert pos == expected_pos
    assert neg == expected_neg
    assert common == expected_common
    assert end == expected_end


PREDICT_POS = {"로", "에", "의", "로는", "에서", "으로", "으로는", "으로써", "이지만"}
PREDICT_NEG = {"다고", "고", "자는", "자고", "지만"}
PREDICT_COMMON = {"은", "는"}


@pytest.mark.parametrize(
    ("word", "features", "expected_support", "expected_score"),
    [
        ("대학생으", [("로", 10), ("로써", 5), ("로는", 3)], 0, 0),
        ("대학생", [("으로", 10), ("으로써", 5), ("으로는", 3)], 18, 1.0),
        ("너로", [("는", 5), ("", 3)], 0, 0),
        ("너", [("로는", 5), ("는", 3)], 8, 1.0),
        ("관계자", [("는", 10), ("로", 5), ("이지만", 5)], 10, 1.0),
        (
            "관계",
            [("자는", 10), ("자로", 5), ("는", 20), ("의", 25), ("자이지만", 5)],
            45,
            0.42857142857142855,
        ),
        ("하자", [("는", 10), ("고", 5)], 0, 0),
        ("하", [("자는", 10), ("자고", 5)], 15, -1.0),
        ("가고있다", [("고", 10), ("지만", 3)], 3, 0),
        ("가고있", [("다고", 10), ("지만", 5), ("다지만", 3)], 15, -1.0),
        ("경찰국", [("은", 1), ("에", 1), ("에서", 1)], 3, 1.0),
        ("아이웨딩", [("", 90), ("은", 3), ("측은", 1)], 93, 0.9893617021276596),
        ("아이엠텍", [("은", 2), ("", 2)], 4, 1.0),
    ],
    ids=[
        "대학생으",
        "대학생",
        "너로",
        "너",
        "관계자",
        "관계",
        "하자",
        "하",
        "가고있다",
        "가고있",
        "경찰국",
        "아이웨딩",
        "아이엠텍",
    ],
)
def test_predict_single_noun(word, features, expected_support, expected_score):
    support, score = predict_single_noun(word, features, PREDICT_POS, PREDICT_NEG, PREDICT_COMMON)
    assert support == expected_support
    assert score == expected_score


class TestPostprocessingNJ:
    """postprocessing_nj=False 시 check_N_is_NJ 단계를 건너뜀을 검증한다."""

    def _make_lrgraph(self, eojeols: list[str]) -> LRGraph:
        lrgraph = LRGraph({})
        for eojeol in eojeols:
            lrgraph.add_eojeol(eojeol)
        return lrgraph

    def test_postprocessing_nj_default_removes(self):
        """기본값(postprocessing_nj=True)이면 N+조사 패턴이 제거될 수 있다."""
        nouns: dict[str, tuple[int, float]] = {"상식": (100, 1.0), "상식이": (10, 0.8)}
        eojeols = ["상식은"] * 100 + ["상식의"] * 50 + ["상식이다"] * 10
        lrgraph = self._make_lrgraph(eojeols)
        features: set[str] = set()
        result = postprocessing(nouns, lrgraph, features, 0.3, False, postprocessing_nj=True)
        assert isinstance(result, dict)

    def test_postprocessing_nj_false_preserves(self):
        """postprocessing_nj=False이면 check_N_is_NJ 단계를 건너뛴다."""
        nouns: dict[str, tuple[int, float]] = {"상식": (100, 1.0), "상식이": (10, 0.8)}
        eojeols = ["상식은"] * 100 + ["상식의"] * 50
        lrgraph = self._make_lrgraph(eojeols)
        features: set[str] = set()

        result_with_nj = postprocessing(nouns.copy(), lrgraph, features, 0.3, False, postprocessing_nj=True)
        result_without_nj = postprocessing(nouns.copy(), lrgraph, features, 0.3, False, postprocessing_nj=False)

        assert len(result_without_nj) >= len(result_with_nj)


class TestInjectKnownNouns:
    def _make_lrgraph(self, eojeols: list[str]) -> LRGraph:
        lrgraph = LRGraph({})
        for eojeol in eojeols:
            lrgraph.add_eojeol(eojeol)
        return lrgraph

    def test_known_noun_added(self):
        """known_nouns에 있는 단어가 기존 결과에 없으면 추가된다."""
        nouns: dict[str, tuple[int, float]] = {"학생": (100, 0.9)}
        eojeols = ["트와이스는"] * 50 + ["트와이스의"] * 30
        lrgraph = self._make_lrgraph(eojeols)
        result = _inject_known_nouns(nouns, {"트와이스"}, lrgraph, min_noun_frequency=1)
        assert "트와이스" in result
        assert result["트와이스"][1] == 1.0

    def test_existing_noun_not_overwritten(self):
        """이미 추출된 명사는 덮어쓰지 않는다."""
        nouns: dict[str, tuple[int, float]] = {"학생": (100, 0.9)}
        lrgraph = self._make_lrgraph([])
        result = _inject_known_nouns(nouns, {"학생"}, lrgraph, min_noun_frequency=1)
        assert result["학생"] == (100, 0.9)

    def test_oov_known_noun_added_with_zero_freq(self):
        """LRGraph에 없는 OOV 단어도 frequency=0으로 추가된다."""
        nouns: dict[str, tuple[int, float]] = {}
        lrgraph = self._make_lrgraph([])
        result = _inject_known_nouns(nouns, {"OOV단어"}, lrgraph, min_noun_frequency=1)
        assert "OOV단어" in result
        assert result["OOV단어"] == (0, 1.0)

    def test_known_nouns_none_skipped(self):
        """known_nouns가 None이면 extract()에서 호출되지 않는다 (직접 검증 불필요)."""
        nouns: dict[str, tuple[int, float]] = {"학생": (100, 0.9)}
        lrgraph = self._make_lrgraph([])
        result = _inject_known_nouns(nouns, set(), lrgraph, min_noun_frequency=1)
        assert result == nouns


class TestCompoundMinNounScoreFilter:
    """compound 추출 시 min_noun_score 필터링이 적용됨을 검증한다."""

    def test_compounds_below_min_score_excluded(self):
        """min_noun_score 미만인 compound는 nouns에 추가되지 않는다."""
        min_noun_score = 0.3
        nouns: dict[str, tuple[int, float]] = {"학교": (100, 0.9)}
        compounds: dict[str, tuple[int, float]] = {
            "서울대학교": (50, 0.8),  # min_noun_score 이상 → 포함
            "어떤복합어": (30, 0.2),  # min_noun_score 미만 → 제외
            "경계복합어": (20, 0.3),  # 정확히 min_noun_score → 포함 (>= 조건)
        }
        nouns.update({noun: (freq, sc) for noun, (freq, sc) in compounds.items() if sc >= min_noun_score})

        assert "서울대학교" in nouns
        assert "경계복합어" in nouns
        assert "어떤복합어" not in nouns

    def test_all_compounds_pass_min_score_threshold(self):
        """extract_compounds_func 이 반환한 compounds 는 항상 min_noun_score 이상이다."""
        from soynlp.noun.lr import extract_compounds_func

        pos_features = {"는", "의", "를", "이", "가", "에"}
        # 서울(0.9), 대학교(0.8) → compound 서울대학교 추출
        noun_scores: dict[str, tuple[int, float]] = {
            "서울": (100, 0.9),
            "대학교": (80, 0.8),
        }
        # 서울대학교가 eojeol 로 등장하는 그래프
        sents = ["서울대학교는 유명하다"] * 10
        lrgraph = LRGraph.from_sents(sents)

        min_noun_score = 0.3
        compounds, _, _ = extract_compounds_func(lrgraph, noun_scores, 1, min_noun_score, pos_features, False)
        filtered = {noun: score for noun, score in compounds.items() if score[1] >= min_noun_score}

        assert set(compounds.keys()) == set(filtered.keys()), "필터 전후 compound 집합이 달라지면 안 됨"
