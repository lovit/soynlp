import numpy as np
from collections import defaultdict
from scipy.sparse import diags
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from soynlp.utils import get_process_memory

def sent_to_word_context_matrix(sents, windows=3, min_tf=10,
        tokenizer=lambda x:x.split(), verbose=True):

    # counting word frequency, first
    word_counter = defaultdict(int)
    for i_sent, sent in enumerate(sents):
        if verbose and i_sent % 1000 == 0:
            print('\rcounting word frequency from {} sents, mem={} Gb'.format(
                i_sent, '%.3f' % get_process_memory()), flush=True, end='')
        words = tokenizer(sent)
        for word in words:
            word_counter[word] += 1
    if verbose:
        print('\rcounting word frequency from {} sents was done. mem={} Gb'.format(
            i_sent, '%.3f' % get_process_memory()), flush=True, end='')
    
    # filtering with min_tf    
    vocabulary = {word for word, count in word_counter.items() if count >= min_tf}
    vocabulary = {word:idx for idx, word in enumerate(sorted(vocabulary, key=lambda w:-word_counter[w]))}
    idx2vocab = [word for word, _ in sorted(vocabulary.items(), key=lambda w:w[1])]
    del word_counter

    # scanning (word, context) pairs
    base2contexts = defaultdict(lambda: defaultdict(int))

    for i_sent, sent in enumerate(sents):
        if verbose and i_sent % 1000 == 0:
            print('\rscanning (word, context) pairs from {} sents, mem={} Gb'.format(
                i_sent, '%.3f' % get_process_memory()), flush=True, end='')

        words = tokenizer(sent)
        if not words:
            continue

        n = len(words)

        for i, base in enumerate(words):
            if not (base in vocabulary):
                continue

            # left_contexts
            for context in words[max(0, i-windows):i]:
                if not (context in vocabulary):
                    continue
                base2contexts[base][context] += 1

            # right_contexts
            for context in words[min(i+1, n):min(i+windows, n)+1]:
                if not (context in vocabulary):
                    continue
                base2contexts[base][context] += 1

    if verbose:
        print('\rscanning (word, context) pairs from {} sents was done. mem={} Gb'.format(
            i_sent + 1, '%.3f' % get_process_memory()), flush=True, end='')

    # Encoding dict to vectors
    rows = []
    cols = []
    data = []
    for base, contexts in base2contexts.items():
        base_idx = vocabulary[base]
        for context, cooccurrence in contexts.items():
            context_idx = vocabulary[context]
            rows.append(base_idx)
            cols.append(context_idx)
            data.append(cooccurrence)
    x = csr_matrix((data, (rows, cols)))

    if verbose:
        print('\r(word, context) matrix was constructed. shape = {}{}'.format(
            x.shape, ' '*20))

    return x, idx2vocab


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