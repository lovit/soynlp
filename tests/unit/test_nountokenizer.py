import pytest

from soynlp.tokenizer import NounMatchTokenizer

SENTENCE = "아이오아이의아이들은 오이오이를 좋아하는 아이들이오"
SENTENCE_WITH_PREFIX = "헐아이오아이의아이들은 오이오이를 좋아하는 아이들이오"


@pytest.mark.parametrize(
    ("noun_scores", "sentence", "concat", "must_be_L", "expected"),
    [
        (
            {"아이": 0.5, "아이오": 0.7, "아이오아이": 0.8, "오이": 0.7},
            SENTENCE,
            True,
            False,
            ["아이오아이", "아이", "오이오이", "아이"],
        ),
        (
            {"아이", "아이오", "아이오아이", "오이"},
            SENTENCE,
            True,
            False,
            ["아이오아이", "아이", "오이오이", "아이"],
        ),
        (
            {"아이": 1.0, "아이오": 1.0, "아이오아이": 1.0, "오이": 1.0},
            SENTENCE,
            False,
            False,
            ["아이오아이", "아이", "오이", "오이", "아이"],
        ),
        (
            {"아이": 1.0, "아이오": 1.0, "아이오아이": 1.0, "오이": 1.0},
            SENTENCE_WITH_PREFIX,
            False,
            False,
            ["아이오아이", "아이", "오이", "오이", "아이"],
        ),
        (
            {"아이": 1.0, "아이오": 1.0, "아이오아이": 1.0, "오이": 1.0},
            SENTENCE_WITH_PREFIX,
            False,
            True,
            ["오이", "아이"],
        ),
    ],
    ids=[
        "dict_scores_concat",
        "set_scores_concat",
        "no_concat",
        "prefix_no_concat",
        "prefix_must_be_L",
    ],
)
def test_nounmatch_tokenizer(noun_scores, sentence, concat, must_be_L, expected):
    tokenizer = NounMatchTokenizer(noun_scores)
    nouns = tokenizer.tokenize(sentence, return_words=True, concat_compound=concat, must_be_L=must_be_L)
    assert nouns == expected
