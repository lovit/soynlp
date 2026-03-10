import pytest

from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer, RegexTokenizer
from soynlp.tokenizer.tokenizer_builder import EojeolPatternTrainer


def test_regex_tokenizer_empty_string():
    tokenizer = RegexTokenizer()
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   ") == []


def test_regex_tokenizer():
    sentence = "abc123가나다 alphabet!!3.14한글 hank`s report"
    expected_words = ["abc", "123", "가나다", "alphabet", "!!", "3.14", "한글", "hank`s", "report"]
    expected_offsets = [0, 3, 6, 10, 18, 20, 24, 27, 34]

    tokenizer = RegexTokenizer()
    words = tokenizer.tokenize(sentence, return_words=True)
    tokens = tokenizer.tokenize(sentence, return_words=False)
    offsets = [t.begin for t in tokens]

    assert words == expected_words
    assert offsets == expected_offsets


@pytest.mark.parametrize(
    ("scores", "sentence", "tolerance", "remove_r", "expected"),
    [
        (
            {"파스": 0.65, "파스타": 0.7, "좋아": 0.3},
            "파스타가 좋아요 파스타가좋아요",
            0.0,
            False,
            ["파스타", "가", "좋아", "요", "파스타", "가좋아요"],
        ),
        (
            {"파스": 0.65, "파스타": 0.7, "좋아": 0.3},
            "파스타가 좋아요 파스타가좋아요",
            0.0,
            True,
            ["파스타", "좋아", "파스타"],
        ),
        (
            {"파스": 0.75, "파스타": 0.7, "좋아": 0.3},
            "파스타가 좋아요 파스타가좋아요",
            0.06,
            False,
            ["파스타", "가", "좋아", "요", "파스타", "가좋아요"],
        ),
        (
            {"파스": 0.75, "파스타": 0.7, "좋아": 0.3},
            "파스타가 좋아요 파스타가좋아요",
            0.0,
            False,
            ["파스", "타가", "좋아", "요", "파스", "타가좋아요"],
        ),
    ],
    ids=["basic", "remove_r", "tolerance_override", "higher_prefix_wins"],
)
def test_l_tokenizer(scores, sentence, tolerance, remove_r, expected):
    tokenizer = LTokenizer(scores)
    words = tokenizer.tokenize(sentence, tolerance=tolerance, remove_r=remove_r)
    assert words == expected


def test_maxscore_tokenizer():
    scores = {"파스": 0.65, "파스타": 0.7, "좋아": 0.3, "스타": 0.65}
    sentence = "파스타짱좋아 파스타짱 짱좋아요 짱짱맨 파스좋아!"
    expected_words = ["파스타", "짱", "좋아", "파스타", "짱", "짱", "좋아", "요", "짱짱맨", "파스", "좋아", "!"]
    expected_begin = [0, 3, 4, 7, 10, 12, 13, 15, 17, 21, 23, 25]

    tokenizer = MaxScoreTokenizer(scores)
    tokens = tokenizer.tokenize(sentence, return_words=False)
    words = [t.word for t in tokens]
    begin = [t.begin for t in tokens]

    assert words == expected_words
    assert begin == expected_begin


_TRAINER_SENTS = [
    "나는 학생 입니다",
    "나는 사과를 먹었다",
    "학생이 사과를 먹었다",
    "나는 나는 학생 학생",
] * 200


class TestEojeolPatternTrainerMultiprocessing:
    def test_train_multi_equals_single(self):
        """n_workers=4로 train한 lrgraph가 단일 프로세스와 동일하다."""
        trainer_single = EojeolPatternTrainer(min_frequency=1, verbose=False)
        trainer_single.train(_TRAINER_SENTS, n_workers=1)

        trainer_multi = EojeolPatternTrainer(min_frequency=1, verbose=False)
        trainer_multi.train(_TRAINER_SENTS, n_workers=4)

        assert trainer_single.wordset_l == trainer_multi.wordset_l
        assert trainer_single.wordset_r == trainer_multi.wordset_r
        assert trainer_single.lrgraph == trainer_multi.lrgraph
        assert trainer_single.rlgraph == trainer_multi.rlgraph

    def test_train_multi_completes(self):
        """n_workers=4 train이 에러 없이 완료되고 lrgraph가 비어있지 않다."""
        trainer = EojeolPatternTrainer(min_frequency=1, verbose=False)
        trainer.train(_TRAINER_SENTS, n_workers=4)

        assert trainer.lrgraph is not None
        assert len(trainer.lrgraph) > 0
        assert trainer.rlgraph is not None
