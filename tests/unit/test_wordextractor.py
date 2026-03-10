from collections import defaultdict

import pytest

from soynlp.word import WordExtractor
from soynlp.word.word import (
    AccessorVariety,
    BranchingEntropy,
    CohesionScore,
    calculate_cohesion,
    calculate_cohesion_batch,
    count_substrings,
    get_entropy,
)


def test_score_dataclass():
    assert CohesionScore("토크나이저", 0.8, 0.5) == CohesionScore(subword="토크나이저", leftside=0.8, rightside=0.5)
    assert BranchingEntropy("토크나이저", 1.53, 2.25) == BranchingEntropy(subword="토크나이저", leftside=1.53, rightside=2.25)
    assert AccessorVariety("토크나이저", 0.8, 0.5) == AccessorVariety(subword="토크나이저", leftside=0.8, rightside=0.5)


def test_counting_substrings():
    train_data = ["여름이는 여름을 좋아한다", "올겨울에는 겨울에 갔다"]
    L, R, prev_sub, sub_next = count_substrings(
        train_data=train_data,
        L=defaultdict(int),
        R=defaultdict(int),
        prev_sub=defaultdict(int),
        sub_next=defaultdict(int),
        max_left_length=3,
        max_right_length=2,
        min_frequency=1,
        prune_per_lines=-1,
        cohesion_only=False,
        verbose=True,
    )

    assert L == {
        "여": 2,
        "여름": 2,
        "여름이": 1,
        "여름을": 1,
        "좋": 1,
        "좋아": 1,
        "좋아한": 1,
        "올": 1,
        "올겨": 1,
        "올겨울": 1,
        "겨": 1,
        "겨울": 1,
        "겨울에": 1,
        "갔": 1,
        "갔다": 1,
    }

    assert R == {
        "는": 2,
        "이는": 1,
        "을": 1,
        "름을": 1,
        "다": 2,
        "한다": 1,
        "에는": 1,
        "에": 1,
        "울에": 1,
    }

    assert prev_sub == {
        ("다", "여"): 1,
        ("다", "여름"): 1,
        ("다", "여름이"): 1,
        ("는", "여"): 1,
        ("는", "여름"): 1,
        ("는", "여름을"): 1,
        ("을", "좋"): 1,
        ("을", "좋아"): 1,
        ("을", "좋아한"): 1,
        ("다", "올"): 1,
        ("다", "올겨"): 1,
        ("다", "올겨울"): 1,
        ("는", "겨"): 1,
        ("는", "겨울"): 1,
        ("는", "겨울에"): 1,
        ("에", "갔"): 1,
        ("에", "갔다"): 1,
    }

    assert sub_next == {
        ("는", "여"): 1,
        ("이는", "여"): 1,
        ("여름을", "좋"): 1,
        ("을", "좋"): 1,
        ("름을", "좋"): 1,
        ("다", "여"): 1,
        ("한다", "여"): 1,
        ("는", "겨"): 1,
        ("에는", "겨"): 1,
        ("겨울에", "갔"): 1,
        ("에", "갔"): 1,
        ("울에", "갔"): 1,
        ("갔다", "올"): 1,
        ("다", "올"): 1,
    }


def test_counting_substrings_min_frequency():
    train_data = ["여름이는 여름을 좋아한다", "올겨울에는 겨울에 갔다"]
    L, _, _, _ = count_substrings(
        train_data=train_data,
        L=defaultdict(int),
        R=defaultdict(int),
        prev_sub=defaultdict(int),
        sub_next=defaultdict(int),
        max_left_length=3,
        max_right_length=2,
        min_frequency=2,
        prune_per_lines=-1,
        cohesion_only=False,
        verbose=True,
    )
    assert L == {"여": 2, "여름": 2}


_COHESION_L = {
    "아": 30000,
    "아이": 4910,
    "아이폰": 700,
    "아이폰의": 100,
    "아이돌": 350,
    "아이오": 307,
    "아이오아": 270,
    "아이오아이": 270,
    "아이오아이는": 40,
}
_COHESION_R = {
    "이오아이는": 40,
    "오아이는": 40,
    "아이는": 350,
    "이는": 9500,
    "는": 54000,
    "이폰의": 100,
    "이폰": 700,
    "아이돌": 50,
    "간아이돌": 50,
}


@pytest.mark.parametrize(
    ("word", "expected_l", "expected_r"),
    [
        ("아이", 0.16366666666666665, 0),
        ("아이오", 0.10115993936995679, 0),
        ("아이오아", 0.20800838230519042, 0),
        ("아이오아이", 0.3080070288241023, 0),
        ("아이오아이는", 0.26606499942619716, 0.0),
        ("아이폰", 0.15275252316519466, 0),
        ("아이폰의", 0.14938015821857217, 0),
        ("아이돌", 0.10801234497346433, 0),
        ("주간아이돌", 0, 0),
    ],
    ids=["아이", "아이오", "아이오아", "아이오아이", "아이오아이는", "아이폰", "아이폰의", "아이돌", "주간아이돌"],
)
def test_cohesion_score(word, expected_l, expected_r):
    l_score, r_score = calculate_cohesion(word, _COHESION_L, _COHESION_R)
    assert abs(l_score - expected_l) < 1e-6
    assert abs(r_score - expected_r) < 1e-6


def test_cohesion_score_batch():
    train_data = [
        "여름이는 여름을 좋아한다",
        "올겨울에는 겨울에 갔다",
        "겨울이는 겨울이를 겨울겨울",
        "여름이 여름에 여름을 여름여름",
        "여지가 있다",
    ]
    L, R, _, _ = count_substrings(
        train_data=train_data,
        L=defaultdict(int),
        R=defaultdict(int),
        prev_sub=defaultdict(int),
        sub_next=defaultdict(int),
        max_left_length=3,
        max_right_length=2,
        min_frequency=2,
        prune_per_lines=-1,
        cohesion_only=False,
        verbose=True,
    )

    extracteds_L = calculate_cohesion_batch(L, R, 0.8, 0.0)
    extracteds_R = calculate_cohesion_batch(L, R, 0.0, 0.1)

    assert "여름" not in extracteds_R
    assert "겨울" not in extracteds_R
    assert "이는" in extracteds_R
    assert "여름" in extracteds_L
    assert "겨울" in extracteds_L
    assert "이는" not in extracteds_L


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([3, 4, 3], 1.0888999),
        ([100, 1, 1], 0.11010),
    ],
    ids=["uniform_ish", "skewed"],
)
def test_get_entropy(counts, expected):
    assert abs(get_entropy(counts) - expected) < 0.0001


def test_get_entropy_all_zeros():
    # 전부 0인 리스트 → ZeroDivisionError 없이 0.0 반환
    assert get_entropy([0, 0, 0]) == 0.0


def test_get_entropy_empty():
    assert get_entropy([]) == 0.0


# --- WordExtractor multiprocessing tests ---

_WORD_SENTS = [
    "아이오아이가 평가단에게 높은 점수를 받았습니다",
    "아이오아이는 아이돌 그룹입니다",
    "자연어처리는 어렵고 재미있는 분야입니다",
    "자연어처리를 열심히 공부합니다",
    "명사추출은 중요한 자연어처리 작업입니다",
] * 200


def test_word_extractor_multi_L_equals_single():
    """n_workers=4 결과의 L 카운터가 단일 프로세스와 동일하다."""
    ext_single = WordExtractor(verbose=False)
    ext_single.extract(_WORD_SENTS, min_frequency=1, n_workers=1)

    ext_multi = WordExtractor(verbose=False)
    ext_multi.extract(_WORD_SENTS, min_frequency=1, n_workers=4)

    assert ext_single.L == ext_multi.L
    assert ext_single.R == ext_multi.R


def test_word_extractor_auto_workers():
    """n_workers=-1이면 CPU 코어 수를 자동 사용한다."""
    ext = WordExtractor(verbose=False)
    result = ext.extract(_WORD_SENTS, min_frequency=1, n_workers=-1)
    assert "cohesion" in result
    assert len(result["cohesion"]) > 0
