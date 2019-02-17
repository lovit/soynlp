import numpy as np
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
    # because exp_pmi is sparse matrix and type of exp_pmi.data is numpy.ndarray
    indices = np.where(exp_pmi.data < min_exp_pmi)[0]
    exp_pmi.data[indices] = 1

    # apply logarithm
    exp_pmi.data = np.log(exp_pmi.data)
    return exp_pmi

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
    It returns
    ----------
    pmi : scipy.sparse.dok_matrix or scipy.sparse.csr_matrix
        (word, contexts) pmi value sparse matrix
    px : numpy.ndarray
        Probability of rows (items)
    py : numpy.ndarray
        Probability of columns (features)
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

def pmi_memory_friendly(x, min_pmi=0, alpha=0.0001, verbose=False):
    # convert x to probability matrix & marginal probability 
    px = (x.sum(axis=1) / x.sum()).reshape(-1)
    py = (x.sum(axis=0) / x.sum()).reshape(-1)
    pxy = x / x.sum()
    
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

class PMI:
    def __init__(self, windows=3, min_tf=10, verbose=True,
                 tokenizer=lambda x:x.split(), min_pmi=0, alpha=0.0001):

        self.windows = windows
        self.min_tf = min_tf
        self.verbose = verbose
        self.tokenizer = tokenizer
        self.min_pmi = min_pmi
        self.alpha = alpha
    
    def train(self, sents):
        # construct word - context matrix
        self.x, self.idx2vocab = sent_to_word_contexts_matrix(
            sents, self.windows, self.min_tf, self.tokenizer, self.verbose)
        self.vocab2idx = {vocab:idx for idx, vocab in enumerate(self.idx2vocab)}

        # compute pmi
        self.pmi_ = pmi(self.x, min_pmi=self.min_pmi, alpha=self.alpha, verbose=self.verbose)

        return self

    def most_similar_words(self, query, topk=10, filter_func=lambda x:True):
        assert topk > 0

        if not (query in self.vocab2idx):
            return []

        query_idx = self.vocab2idx[query]
        dist = pairwise_distances(self.x[query_idx,:], self.x, metric='cosine')[0]
        similars = []
        for similar_idx in dist.argsort():
            if similar_idx == query_idx:
                continue

            if len(similars) >= topk:
                break

            similar_word = self.idx2vocab[similar_idx]
            if not filter_func(similar_word):
                continue

            similars.append((similar_word, 1-dist[similar_idx]))

        return similars

    def most_related_contexts(self, query, topk=10, filter_func=lambda x:True):
        assert topk > 0

        if not (query in self.vocab2idx):
            return []

        query_idx = self.vocab2idx[query]

        submatrix = self.pmi_[query_idx,:].tocsr()
        contexts = submatrix.nonzero()[1]
        pmi_i = submatrix.data
        most_relateds = [(idx, pmi_ij) for idx, pmi_ij in zip(contexts, pmi_i)]
        most_relateds = sorted(most_relateds, key=lambda x:-x[1])[:topk]
        most_relateds = [(self.idx2vocab[idx], pmi_ij) for idx, pmi_ij in most_relateds]

        return most_relateds