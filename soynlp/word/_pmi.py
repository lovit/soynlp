import numpy as np
from scipy.sparse import diags
from scipy.sparse import dok_matrix
from sklearn.metrics import pairwise_distances
from soynlp.utils import get_process_memory
from soynlp.vectorizer import sent_to_word_context_matrix

def pmi(x, min_pmi=0, alpha=0.0001, verbose=False):
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
            print('\rcomputing pmi {:.3} %  '.format(100 * _n_idx / indices.shape[0]), flush=True, end='')
        # apply logarithm
        pmi_dok[rows[idx], cols[idx]] = np.log(data[idx])
    if verbose:
        print('\rcomputing pmi was done   ', flush=True)

    return pmi_dok

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
        self.x, self.idx2vocab = sent_to_word_context_matrix(
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