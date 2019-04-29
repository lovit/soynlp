# -*- encoding:utf8 -*-

import os
import psutil
import sys
from collections import defaultdict
from sklearn.metrics import pairwise_distances


installpath = os.path.sep.join(
    os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])

def get_available_memory():
    """It returns remained memory as percentage"""

    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)

def get_process_memory():
    """It returns the memory usage of current process"""
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def check_dirs(filepath):
    dirname = os.path.dirname(filepath)
    if dirname and dirname == '.' and not os.path.exists(dirname):
        os.makedirs(dirname)
        print('created {}'.format(dirname))

def sort_by_alphabet(filepath):
    if sys.version.split('.')[0] == '2':
        with open(filepath) as f:
            docs = [doc.strip() for doc in f]
            docs = [doc for doc in docs if doc]
    else:
        with open(filepath, encoding= "utf-8") as f:
            docs = [doc.strip() for doc in f]
            docs = [doc for doc in docs if doc]
    if sys.version.split('.')[0] == '2':
        with open(filepath, 'w') as f:
            for doc in sorted(docs):
                f.write('{}\n'.format(doc))
    else:
        with open(filepath, 'w', encoding= "utf-8") as f:
            for doc in sorted(docs):
                f.write('{}\n'.format(doc))

def most_similar(query, vector, item_to_idx, idx_to_item, topk=10):
    """
    :param query: str
        String type query word
    :param vector: numpy.ndarray or scipy.sparse.matrix
        Vector representation of row
    :param item_to_idx: dict
        Mapper from str type item to int type index
    :param idx_to_item: list
        Mapper from int type index to str type item
    :param topk: int
        Maximum number of similar items.
        If set top as negative value, it returns similarity with all words

    Returns
    ----------
    similars : list of tuple
        List contains tuples (item, cosine similarity)
        Its length is topk
    """

    q = item_to_idx.get(query, -1)
    if q == -1:
        return []
    qvec = vector[q].reshape(1,-1)
    dist = pairwise_distances(qvec, vector, metric='cosine')[0]
    sim_idxs = dist.argsort()
    if topk > 0:
        sim_idxs = sim_idxs[:topk+1]
    similars = [(idx_to_item[idx], 1 - dist[idx]) for idx in sim_idxs if idx != q]
    return similars

class DoublespaceLineCorpus:    
    def __init__(self, corpus_fname, num_doc = -1, num_sent = -1, iter_sent = False, skip_header = 0):
        self.corpus_fname = corpus_fname
        self.num_doc = 0
        self.num_sent = 0
        self.iter_sent = iter_sent
        self.skip_header = skip_header
        if (num_doc > 0) or (num_sent > 0):
            self.num_doc, self.num_sent = self._check_length(num_doc, num_sent)

    def _check_length(self, num_doc, num_sent):
        num_sent_ = 0
        try:
            try:
                # python version check
                if sys.version.split('.')[0] == '2':
                    f = open(self.corpus_fname)
                else:
                    f = open(self.corpus_fname, encoding= "utf-8")

                # skip headers
                for _ in range(self.skip_header):
                    next(f)
                
                # check length
                for doc_idx, doc in enumerate(f):
                    if (num_doc > 0) and (doc_idx >= num_doc):
                        return doc_idx, num_sent_
                    sents = doc.split('  ')
                    sents = [sent for sent in sents if sent.strip()]
                    num_sent_ += len(sents)
                    if (num_sent > 0) and (num_sent_ > num_sent):
                        return doc_idx+1, min(num_sent, num_sent_)

            finally:
                f.close()
            return doc_idx+1, num_sent_

        except Exception as e:
            print(e)
            return -1, -1
    
    def __iter__(self):
        try:
            try:
                if sys.version.split('.')[0] == '2':
                    f = open(self.corpus_fname)
                else:
                    f = open(self.corpus_fname, encoding='utf-8')
                
                # skip headers
                for _ in range(self.skip_header):
                    next(f)
                    
                # iteration
                num_sent, stop = 0, False
                for doc_idx, doc in enumerate(f):
                    if stop:
                        break

                    # yield doc
                    if not self.iter_sent:
                        yield doc.strip()
                        if (self.num_doc > 0) and ((doc_idx + 1) >= self.num_doc):
                            stop = True
                        continue

                    # yield sents
                    for sent in doc.split('  '):
                        if (self.num_sent > 0) and (num_sent >= self.num_sent):
                            stop = True
                            break
                        sent = sent.strip()
                        if sent:
                            yield sent
                            num_sent += 1
            finally:
                f.close()

        except Exception as e:
            print(e)

    def __len__(self):
        if self.num_doc == 0:
            self.num_doc, self.num_sent = self._check_length(-1, -1)
        return self.num_sent if self.iter_sent else self.num_doc

class EojeolCounter:
    def __init__(self, sents=None, min_count=1, max_length=15,
        filtering_checkpoint=0, verbose=False, preprocess=None):

        self.min_count = min_count
        self.max_length = max_length
        self.filtering_checkpoint = filtering_checkpoint
        self.verbose = verbose
        self._coverage = 0.0

        if preprocess is None:
            preprocess = lambda x:x
        self.preprocess = preprocess

        if sents is not None:
            self._counter = self._counting_from_sents(sents)
        else:
            self._counter = {}

        self._count_sum = 0
        self._set_count_sum()

    def __getitem__(self, eojeol):
        return self._counter.get(eojeol, 0)

    def __len__(self):
        return len(self._counter)

    def _set_count_sum(self):
        self._count_sum = sum(self._counter.values())

    def _counting_from_sents(self, sents):
        _counter = {}
        for i_sent, sent in enumerate(sents):
            sent = self.preprocess(sent)
            # filtering during eojeol counting
            if (self.min_count > 1 and
                self.filtering_checkpoint > 0 and
                i_sent > 0 and
                i_sent % self.filtering_checkpoint == 0):
                _counter = {k:v for k,v in _counter.items()
                            if v >= self.min_count}
            # add eojeol count
            for eojeol in sent.split():
                if (not eojeol) or (len(eojeol) > self.max_length):
                    continue
                _counter[eojeol] = _counter.get(eojeol, 0) + 1
            # print status
            if self.verbose and i_sent % 100000 == 99999:
                print('\r[EojeolCounter] n eojeol = {} from {} sents. mem={} Gb{}'.format(
                    len(_counter), i_sent + 1, '%.3f'%get_process_memory(), ' '*20), flush=True, end='')
        # final filtering
        _counter = {k:v for k,v in _counter.items() if v >= self.min_count}
        if self.verbose:
            print('\r[EojeolCounter] n eojeol = {} from {} sents. mem={} Gb{}'.format(
                len(_counter), i_sent + 1, '%.3f'%get_process_memory(), ' '*20), flush=True)
        return _counter

    @property
    def coverage(self):
        return self._coverage

    @coverage.setter
    def coverage(self, value):
        if not (0 <= value <= 1):
            raise ValueError('coverage should be in [0, 1]')
        self._coverage = value

    @property
    def num_of_unique_uncovered_eojeols(self):
        return len(self._counter)

    @property
    def num_of_uncovered_eojeols(self):
        return sum(self._counter.values())

    def get_uncovered_eojeols(self, min_count=0):
        return {k:v for k,v in self._counter.items() if v >= min_count}

    def remove_covered_eojeols(self, eojeols):
        self._counter = {k:v for k,v in self._counter.items() if not (k in eojeols)}
        self.coverage = 1 - self.num_of_uncovered_eojeols / self._count_sum

    def get_eojeol_count(self, eojeol):
        return self._counter.get(eojeol, 0)

    def items(self):
        return self._counter.items()

    def to_lrgraph(self, l_max_length=10, r_max_length=9, ignore_one_syllable=False):
        return self._to_lrgraph(self._counter, l_max_length, r_max_length)

    def _to_lrgraph(self, counter, l_max_length=10, r_max_length=9, ignore_one_syllable=False):
        _lrgraph = defaultdict(lambda: defaultdict(int))
        for eojeol, count in counter.items():
            if ignore_one_syllable and len(eojeol) == 1:
                continue
            for e in range(1, min(l_max_length, len(eojeol)) + 1):
                l, r = eojeol[:e], eojeol[e:]
                if len(r) > r_max_length:
                    continue
                _lrgraph[l][r] += count
        _lrgraph = {l:dict(rdict) for l, rdict in _lrgraph.items()}
        lrgraph = LRGraph(lrgraph=_lrgraph,
            l_max_length=l_max_length, r_max_length=r_max_length)
        return lrgraph

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, 'w', encoding='utf-8') as f:
            for eojeol, count in sorted(self._counter.items(), key=lambda x:(-x[1], x[0])):
                f.write('{} {}\n'.format(eojeol, count))

    def load(self, path):
        self._coverage = 0.0
        self._counter = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                word, count = line.split()
                self._counter[word] = int(count)
        self._count_sum = sum(self._counter.values())

class LRGraph:

    def __init__(self, lrgraph=None, sents=None, l_max_length=10, r_max_length=9):

        assert l_max_length > 1 and type(l_max_length) == int
        assert r_max_length > 0 and type(r_max_length) == int

        self.l_max_length = l_max_length
        self.r_max_length = r_max_length

        if sents:
            if lrgraph:
                raise ValueError(
                    'Inserted lrgraph will be ignored. Insert only one (lrgraph, sents)')
            lrgraph = self._construct_graph(sents)
        if lrgraph:
            self._lr, self._rl = self._check_lrgraph(lrgraph)
        else:
            self._lr, self._rl = {}, {}

        self._lr_origin = {l:{r:c for r,c in rdict.items()}
                           for l,rdict in self._lr.items()}

    def _construct_graph(self, sents):
        lrgraph = defaultdict(lambda: defaultdict(int))
        for sent in sents:
            for word in sent.split():
                word = word.strip()
                for e in range(1, min(len(word), self.l_max_length) + 1):
                    l, r = word[:e], word[e:]
                    if len(r) > self.r_max_length:
                        continue
                    lrgraph[l][r] += e
        lrgraph = {l:dict(rdict) for l,rdict in lrgraph.items()}
        return lrgraph

    def _check_lrgraph(self, lrgraph):
        if type(lrgraph) is not dict:
            try:
                lrgraph = dict(lrgraph)
            except:
                raise ValueError('lrgraph type should be dict of dict, not {}'.format(
                    type(lrgraph)))
        nested_dict_type = type(list(lrgraph.values())[0])
        if nested_dict_type == defaultdict:
            lrgraph = {l:dict(rdict) for l,rdict in lrgraph.items()}
        elif not (nested_dict_type == dict):
            raise ValueError('nested value type should be dict, not {}'.format(
                nested_dict_type))
        rlgraph = defaultdict(lambda: defaultdict(int))
        for l, rdict in lrgraph.items():
            for r, c in rdict.items():
                if not r:
                    continue
                rlgraph[r][l] += c
        rlgraph = {r:dict(ldict) for r, ldict in rlgraph.items()}
        return lrgraph, rlgraph

    def reset_lrgraph(self):
        if not self._lr_origin:
            return None

        self._lr, self._rl = self._check_lrgraph(
            {l:{r:c for r,c in rdict.items()}
             for l, rdict in self._lr_origin.items()}
        )

    def add_lr_pair(self, l, r, count=1):
        self._lr[l][r] += count
        if r:
            self._rl[r][l] += count

    def add_eojeol(self, eojeol, count=1):
        for i in range(1, len(eojeol) + 1):
            l, r = eojeol[:i], eojeol[i:]
            self.add_lr_pair(l, r, count)

    def remove_lr_pair(self, l, r, count=1):
        if l in self._lr:
            rdict = self._lr[l]
            if r in rdict:
                rdict[r] -= count
                if rdict[r] <= 0:
                    rdict.pop(r)
                    if len(rdict) <= 0:
                        self._lr.pop(l)
        if r in self._rl:
            ldict = self._rl[r]
            if l in ldict:
                ldict[l] -= count
                if ldict[l] <= 0:
                    ldict.pop(l)
                    if len(ldict) <= 0:
                        self._rl.pop(r)

    def remove_eojeol(self, eojeol, count=1):
        for i in range(1, len(eojeol) + 1):
            l, r = eojeol[:i], eojeol[i:]
            self.remove_lr_pair(l, r, count)

    def get_r(self, l, topk=10):
        rlist = sorted(self._lr.get(l, {}).items(), key=lambda x:-x[1])
        if topk > 0:
            rlist = rlist[:topk]
        return rlist

    def get_l(self, r, topk=10):
        llist = sorted(self._rl.get(r, {}).items(), key=lambda x:-x[1])
        if topk > 0:
            llist = llist[:topk]
        return llist

    def freeze(self):
        """Remove self._lr_origin. Be careful.
        When you excute freeze, you cannot reset_lrgraph anynore."""
        self._lr_origin = None

    def copy_compatified_lrgraph_origin(self):
        """It returns original LRGraph which has no self._lr_origin"""
        lr_graph = LRGraph(
            l_max_length = self.l_max_length,
            r_max_length = self.r_max_length)
        lr_graph._lr, lr_graph._rl = self._check_lrgraph(
            {l:{r:c for r,c in rdict.items()}
             for l, rdict in self._lr_origin.items()}
        )
        return lr_graph

    def to_EojeolCounter(self, reset_lrgraph=False):
        lr = self._lr_origin if reset_lrgraph else self._lr
        counter = {}
        for l, rdict in lr.items():
            for r, count in rdict.items():
                counter[l+r] = count
        eojeol_counter = EojeolCounter(None)
        eojeol_counter._counter = counter
        eojeol_counter._count_sum = sum(counter.values())
        return eojeol_counter

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, 'w', encoding='utf-8') as f:
            for l, rdict in sorted(self._lr_origin.items()):
                for r, c in sorted(rdict.items()):
                    f.write('{} {} {}\n'.format(l, r, c))

    def load(self, path):
        self._lr_origin = {}
        with open(path, encoding='utf-8') as f:
            l = ''
            rdict = {}
            for line in f:
                sep = line.split()
                if not (sep[0] == l):
                    if rdict:
                        self._lr_origin[l] = rdict
                        rdict = {}
                l = sep[0]
                if len(sep) == 2:
                    rdict[''] = int(sep[-1])
                elif len(sep) == 3:
                    rdict[sep[1]] = int(sep[-1])
                else:
                    raise ValueError('Wrong lr-graph format: {}'.format(line))
            if rdict:
                self._lr_origin[l] = rdict
        self._lr, self._rl = self._check_lrgraph(
            {l:{r:c for r,c in rdict.items()}
             for l,rdict in self._lr_origin.items()})
