import numpy as np
from scipy.sparse import csr_matrix, diags


def _as_diag(px, alpha):
    px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray([0 if v == 0 else 1 / (v + alpha) for v in px_diag.data[0]])
    return px_diag


def _logarithm_and_ppmi(exp_pmi, min_exp_pmi):
    n, m = exp_pmi.shape

    rows, cols = exp_pmi.nonzero()
    data = exp_pmi.data

    indices = np.where(data >= min_exp_pmi)[0]
    rows = rows[indices]
    cols = cols[indices]
    data = data[indices]

    data = np.log(data)
    exp_pmi_ = csr_matrix((data, (rows, cols)), shape=(n, m))
    return exp_pmi_


def pmi(X, py=None, min_pmi=0, alpha=0.0, beta=1):
    """Transform `X` to Positive-PMI matrix (CSR sparse matrix)

    Args:
        X (scipy.sparse.csr_matrix) :
            shape = (n items, n features)
        py (numpy.ndarray, optional) :
            shape = (1, word), probability of context words.
            If `py` is None, `pmi` function uses normalized row sum of `X`
        min_pmi (float) :
            Minimum value of pmi.
        alpha (float) :
            Smoothing factor. Default is `0.0`
        beta (float) :
            Smoothing factor. Default is `1.0`

    Returns:
        pmi (scipy.sparse.csr_matrix)
        px (numpy.ndarray)
        py (numpy.ndarray)
    """

    assert 0 < beta <= 1

    px = np.asarray((X.sum(axis=1) / X.sum()).reshape(-1))
    pxy = X / X.sum()
    if py is None:
        py = np.asarray((X.sum(axis=0) / X.sum()).reshape(-1))
    if beta < 1:
        py = py**beta
        py /= py.sum()
    assert py.shape[1] == pxy.shape[1]

    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)
    pmi_mat = _logarithm_and_ppmi(exp_pmi, min_exp_pmi)

    return pmi_mat, px, py
