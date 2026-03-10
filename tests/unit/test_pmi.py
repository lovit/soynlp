import warnings

import numpy as np
from scipy.sparse import csr_matrix

from soynlp.utils import pmi


def test_pmi():
    x = csr_matrix(np.ones((4, 6)))
    assert (pmi(x)[0].todense() - np.zeros((4, 6))).sum() == 0

    x = csr_matrix(np.ones((4, 6)))
    x[3, 4] = 10
    pmi_mat, px, py = pmi(x)
    assert abs(pmi_mat[0, 0] - 0.3185) < 0.001
    assert abs(pmi_mat[3, 4] - 0.526) < 0.01

    with np.printoptions(precision=4, suppress=True, threshold=5):
        print(f"\nX: \n{x.todense()}")
        print(f"\nPMI: \n{pmi_mat.todense()}")
        print(f"\nPx: {px}")
        print(f"\nPy: {py}")


def test_pmi_import_from_utils():
    """soynlp.utils.pmi가 정상적으로 import되고 동작한다."""
    from soynlp.utils.pmi import pmi as pmi_direct

    x = csr_matrix(np.ones((3, 4)))
    pmi_mat, px, py = pmi_direct(x)
    assert pmi_mat.shape == (3, 4)


def test_pmi_deprecated_word_import():
    """soynlp.word.pmi는 DeprecationWarning을 발생시키며 동일한 결과를 반환한다."""
    from soynlp.word.pmi import pmi as pmi_word

    x = csr_matrix(np.ones((3, 4)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pmi_word(x)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "soynlp.utils.pmi" in str(w[0].message)

    # 결과는 새 경로와 동일해야 함
    expected = pmi(x)
    assert (result[0] - expected[0]).nnz == 0
