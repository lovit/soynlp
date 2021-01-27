from soynlp.word.word import CohesionScore, BranchingEntropy, AccessVariety


def test_score_dataclass():
    assert CohesionScore("토크나이저", 0.8, 0.5) == CohesionScore(
        subword="토크나이저", leftside=0.8, rightside=0.5
    )
    assert BranchingEntropy("토크나이저", 1.53, 2.25) == BranchingEntropy(
        subword="토크나이저", leftside=1.53, rightside=2.25
    )
    assert AccessVariety("토크나이저", 0.8, 0.5) == AccessVariety(
        subword="토크나이저", leftside=0.8, rightside=0.5
    )
