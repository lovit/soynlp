import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse import dok_matrix
from sklearn.metrics import pairwise_distances
from soynlp.utils import get_process_memory
from soynlp.vectorizer import sent_to_word_contexts_matrix

def _as_diag(px, alpha):
    px_diag = diags(px.tolist()[0])
    px_diag.data[0] = np.asarray([0 if v == 0 else 1 / (v + alpha) for v in px_diag.data[0]])
    return px_diag

def _logarithm_and_ppmi(exp_pmi, min_exp_pmi):
    n, m = exp_pmi.shape

    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    rows, cols = exp_pmi.nonzero()
    data = exp_pmi.data

    indices = np.where(data >= min_exp_pmi)[0]
    rows = rows[indices]
    cols = cols[indices]
    data = data[indices]

    # apply logarithm
    data = np.log(data)

    # new matrix
    exp_pmi_ = csr_matrix((data, (rows, cols)), shape=(n, m))
    return exp_pmi_

def pmi(X, py=None, min_pmi=0, alpha=0.0, beta=1):
    """
    :param X: scipy.sparse.csr_matrix
        (word, contexts) sparse matrix
    :param py: numpy.ndarray
        (1, word) shape, probability of context words.
    :param min_pmi: float
        Minimum value of pmi. all the values that smaller than min_pmi
        are reset to zero.
        Default is zero.
    :param alpha: float
        Smoothing factor. pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha))
        Default is 0.0
    :param beta: float
        Smoothing factor. pmi(x,y) = log ( Pxy / (Px x Py^beta) )
        Default is 1.0

    Returns
    ----------
    pmi : scipy.sparse.csr_matrix
        (word, contexts) pmi value sparse matrix
    px : numpy.ndarray
        Probability of rows (items)
    py : numpy.ndarray
        Probability of columns (features)

    Usage
    -----
        >>> pmi, px, py = pmi_memory_friendly(X, py=None, min_pmi=0, alpha=0, beta=1.0)
    """

    assert 0 < beta <= 1

    # convert x to probability matrix & marginal probability
    px = np.asarray((X.sum(axis=1) / X.sum()).reshape(-1))
    if py is None:
        py = np.asarray((X.sum(axis=0) / X.sum()).reshape(-1))
    if beta < 1:
        py = py ** beta
        py /= py.sum()
    pxy = X / X.sum()

    # transform px and py to diagonal matrix
    # using scipy.sparse.diags
    # pmi_alpha (x,y) = p(x,y) / ( p(x) x (p(y) + alpha) )
    px_diag = _as_diag(px, 0)
    py_diag = _as_diag(py, alpha)
    exp_pmi = px_diag.dot(pxy).dot(py_diag)

    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)
    pmi = _logarithm_and_ppmi(exp_pmi, min_exp_pmi)

    return pmi, px, py

def pmi_memory_friendly(X, py=None, min_pmi=0, alpha=0.0, beta=1.0, verbose=False):
    """
    :param X: scipy.sparse.csr_matrix
        (word, contexts) sparse matrix
    :param py: numpy.ndarray
        (1, word) shape, probability of context words.
    :param min_pmi: float
        Minimum value of pmi. all the values that smaller than min_pmi
        are reset to zero.
        Default is zero.
    :param alpha: float
        Smoothing factor. pmi(x,y; alpha) = p_xy /(p_x * (p_y + alpha))
        Default is 0.0
    :param beta: float
        Smoothing factor. pmi(x,y) = log ( Pxy / (Px x Py^beta) )
        Default is 1.0
    :param verbose: Boolean
        If True, verbose mode on

    Returns
    ----------
    pmi : scipy.sparse.dok_matrix
        (word, contexts) pmi value sparse matrix
    px : numpy.ndarray
        Probability of rows (items)
    py : numpy.ndarray
        Probability of columns (features)

    Usage
    -----
        >>> pmi, px, py = pmi_memory_friendly(X, py=None, min_pmi=0, alpha=0, beta=1.0)
    """

    assert 0 < beta <= 1

    # convert x to probability matrix & marginal probability 
    px = (X.sum(axis=1) / X.sum()).reshape(-1)
    if py is None:
        py = (X.sum(axis=0) / X.sum()).reshape(-1)
    pxy = X / X.sum()

    assert py.shape[0] == pxy.shape[1]

    if beta < 1:
        py = py ** beta
        py /= py.sum()

    # transform px and py to diagonal matrix
    # using scipy.sparse.diags
    px_diag = diags(px.tolist()[0])
    py_diag = diags((py).tolist()[0])
    
    # pmi_alpha (x,y) = p(x,y) / ( p(x) x (p(y) + alpha) )
    px_diag.data[0] = np.asarray([0 if v == 0 else 1/v for v in px_diag.data[0]])
    py_diag.data[0] = np.asarray([0 if v == 0 else 1/(v + alpha) for v in py_diag.data[0]])
    
    exp_pmi = px_diag.dot(pxy).dot(py_diag)
    
    # PPMI using threshold
    min_exp_pmi = 1 if min_pmi == 0 else np.exp(min_pmi)

    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    indices = np.where(exp_pmi.data > min_exp_pmi)[0]

    pmi_dok = dok_matrix(exp_pmi.shape)

    # prepare data (rows, cols, data)
    rows, cols = exp_pmi.nonzero()
    data = exp_pmi.data

    # enumerate function for printing status
    for _n_idx, idx in enumerate(indices):
        # print current status        
        if verbose and _n_idx % 10000 == 0:
            print('\rcomputing pmi {:.3} %  mem={} Gb    '.format(
                100 * _n_idx / indices.shape[0], '%.3f' % get_process_memory())
                  , flush=True, end='')
        # apply logarithm
        pmi_dok[rows[idx], cols[idx]] = np.log(data[idx])
    if verbose:
        print('\rcomputing pmi was done{}'.format(' '*30), flush=True)

    return pmi_dok, px, py
