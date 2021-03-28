import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags


def _as_diag(px, alpha):
    px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray(
        [0 if v == 0 else 1 / (v + alpha) for v in px_diag.data[0]]
    )
    return px_diag


def _logarithm_and_ppmi(exp_pmi, min_exp_pmi):
    n, m = exp_pmi.shape

    # type of `exp_pmi` and `exp_pmi.data` are scipy.sparse.csr_matrix and numpy.ndarray respective.
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
    """
    Transform `X` to Positive-PMI matrix (CSR sparse matrix)

    Args:
        X (scipy.sparse.csr_matrix) :
            shape = (n items, n features)
        py (numpy.ndarray, optional) :
            shape = (1, word), probability of context words.
            If `py` is None, `pmi` function uses normalized row sum of `X`
        min_pmi (float) :
            Minimum value of pmi.
            All the values that smaller than min_pmi are force to be remove.
        alpha (float) :
            Smoothing factor.
            pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha))
            Default is `0.0
        beta (float) :
            Smoothing factor. pmi(x,y) = log ( Pxy / (Px x Py^beta) )
            Default is `1.0`

    Returns:
        pmi (scipy.sparse.csr_matrix) :
            shape = (n items, n features) pmi value sparse matrix
        px (numpy.ndarray) :
            Probability of rows (items)
        py (numpy.ndarray) :
            Probability of columns (features)

    Examples:
        >>> pmi_mat, px, py = pmi(X, py=None, min_pmi=0, alpha=0, beta=1.0)
    """

    assert 0 < beta <= 1

    # convert x to probability matrix & marginal probability
    px = np.asarray((X.sum(axis=1) / X.sum()).reshape(-1))
    pxy = X / X.sum()
    if py is None:
        py = np.asarray((X.sum(axis=0) / X.sum()).reshape(-1))
    if beta < 1:
        py = py ** beta
        py /= py.sum()
    assert py.shape[1] == pxy.shape[1]

    # transform `px` and `py` to diagonal matrix using scipy.sparse.diags
    # pmi_alpha (x,y) = p(x,y) / ( p(x) x (p(y) + alpha) )
    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)
    pmi = _logarithm_and_ppmi(exp_pmi, min_exp_pmi)

    return pmi, px, py
