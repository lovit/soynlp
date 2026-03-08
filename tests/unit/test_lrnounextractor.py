import pytest

from soynlp.noun.lr import (
    check_r_features,
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
