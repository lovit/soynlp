from soynlp.word.word import CohesionScore, BranchingEntropy, AccessorVariety, get_entropy


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


def test_get_entropy():
    assert abs(get_entropy([3, 4, 3]) - 1.0888999) < 0.0001
    assert abs(get_entropy([100, 1, 1]) - 0.11010) < 0.0001
