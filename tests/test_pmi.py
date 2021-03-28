import numpy as np
from scipy.sparse import csr_matrix
from soynlp.word import pmi


def test_pmi():
    x = csr_matrix(np.ones((4, 6)))
    assert (pmi(x)[0].todense() - np.zeros((4, 6))).sum() == 0

    x = csr_matrix(np.ones((4, 6)))
    x[3, 4] = 10
    pmi_mat, px, py = pmi(x)
    assert abs(pmi_mat[0, 0] - 0.3185) < 0.001  # 0.3185
    assert abs(pmi_mat[3, 4] - 0.526) < 0.01  # 0.5261

    with np.printoptions(precision=4, suppress=True, threshold=5):
        print(f"\nX: \n{x.todense()}")
        print(f"\nPMI: \n{pmi_mat.todense()}")
        print(f"\nPx: {px}")
        print(f"\nPy: {py}")
