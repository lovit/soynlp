from collections import defaultdict
from soynlp.word.word import (
    CohesionScore,
    BranchingEntropy,
    AccessorVariety,
    get_entropy,
    count_substrings
)


def test_score_dataclass():
    assert CohesionScore("토크나이저", 0.8, 0.5) == CohesionScore(
        subword="토크나이저", leftside=0.8, rightside=0.5
    )
    assert BranchingEntropy("토크나이저", 1.53, 2.25) == BranchingEntropy(
        subword="토크나이저", leftside=1.53, rightside=2.25
    )
    assert AccessorVariety("토크나이저", 0.8, 0.5) == AccessorVariety(
        subword="토크나이저", leftside=0.8, rightside=0.5
    )


def test_couting_substrings():
    train_data = ["여름이는 여름을 좋아한다", "올겨울에는 겨울에 갔다"]
    L, R, prev_L, R_next = count_substrings(
        train_data=train_data,
        L=defaultdict(int),
        R=defaultdict(int),
        prev_L=defaultdict(int),
        R_next=defaultdict(int),
        max_left_length=3,
        max_right_length=2,
        min_frequency=1,
        prune_per_lines=-1,
        cohesion_only=False,
        verbose=True
    )

    # ["여름이는", "좋아한다", "올겨울에", "올겨울에는"] 는 길이가 4 이상이므로 포함되지 않음
    L == {
        '여': 2,
        '여름': 2,
        '여름이': 1,
        '여름을': 1,
        '좋': 1,
        '좋아': 1,
        '좋아한': 1,
        '올': 1,
        '올겨': 1,
        '올겨울': 1,
        '겨': 1,
        '겨울': 1,
        '겨울에': 1,
        '갔': 1,
        '갔다': 1
    }

    # ["갔다"]는 L 이기에 포함되지 않음
    assert R == {
        '는': 2,
        '이는': 1,
        '을': 1,
        '름을': 1,
        '다': 2,
        '한다': 1,
        '에는': 1,
        '에': 1,
        '울에': 1
    }

    assert prev_L == {
        '다 여': 1,
        '다 여름': 1,
        '다 여름이': 1,
        '는 여': 1,
        '는 여름': 1,
        '는 여름을': 1,
        '을 좋': 1,
        '을 좋아': 1,
        '을 좋아한': 1,
        '다 올': 1,
        '다 올겨': 1,
        '다 올겨울': 1,
        '는 겨': 1,
        '는 겨울': 1,
        '는 겨울에': 1,
        '에 갔': 1,
        '에 갔다': 1
    }

    # "갔다"는 L 이므로 포함되지 않으며, "갔다 - 올겨울에는" 으로부터 "다 올" 이 계산되었다.
    assert R_next == {
        '는 여': 1,
        '이는 여': 1,
        '을 좋': 1,
        '름을 좋': 1,
        '다 여': 1,
        '한다 여': 1,
        '는 겨': 1,
        '에는 겨': 1,
        '에 갔': 1,
        '울에 갔': 1,
        '다 올': 1
    }

def test_get_entropy():
    assert abs(get_entropy([3, 4, 3]) - 1.0888999) < 0.0001
    assert abs(get_entropy([100, 1, 1]) - 0.11010) < 0.0001
