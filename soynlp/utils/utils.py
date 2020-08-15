import copy
import os
import psutil
import sys
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


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

def check_corpus(corpus):
    """
    Argument
    --------
    corpus : iterable or DoublespaceLineCorpus

    Returns
    -------
    flag : Boolean
        It returns True when __len__ is implemented and the length is larger than 0
    """
    if not hasattr(corpus, '__iter__'):
        raise ValueError('Input corpus must have __iter__ such as list or soynlp.utils.DoublespaceLineCorpus')
    if not hasattr(corpus, '__len__'):
        raise ValueError('Input corpus must have __len__ such as list or soynlp.utils.DoublespaceLineCorpus')
    if len(corpus) <= 0:
        raise ValueError('Input corpus must be longer than 0')
    return True

class DoublespaceLineCorpus:    
    def __init__(self, corpus_fname, num_doc = -1, num_sent = -1, iter_sent = False, skip_header = 0):
        if not os.path.exists(corpus_fname):
            raise ValueError("File {} does not exist".format(corpus_fname))
        self.corpus_fname = corpus_fname
        self.num_doc = 0
        self.num_sent = 0
        self.iter_sent = iter_sent
        self.skip_header = skip_header
        if (num_doc > 0) or (num_sent > 0):
            self.num_doc, self.num_sent = self._check_length(num_doc, num_sent)

    def _check_length(self, num_doc, num_sent):
        num_sent_ = 0

        # python version check
        try:
            if sys.version.split('.')[0] == '2':
                f = open(self.corpus_fname)
            else:
                f = open(self.corpus_fname, encoding= "utf-8")
        except Exception as e:
            print(e)
            return 0, 0

        try:
            # skip headers
            for _ in range(self.skip_header):
                next(f)
        except Exception as e:
            print(e)
            return 0, 0

        # check length
        for doc_idx, doc in enumerate(f):
            if (num_doc > 0) and (doc_idx >= num_doc):
                return doc_idx, num_sent_
            sents = doc.split('  ')
            sents = [sent for sent in sents if sent.strip()]
            num_sent_ += len(sents)
            if (num_sent > 0) and (num_sent_ > num_sent):
                return doc_idx+1, min(num_sent, num_sent_)

        return doc_idx+1, num_sent_

    def __iter__(self):
        try:
            if sys.version.split('.')[0] == '2':
                f = open(self.corpus_fname)
            else:
                f = open(self.corpus_fname, encoding='utf-8')
        except Exception as e:
            print(e)

        try:
            # skip headers
            for _ in range(self.skip_header):
                next(f)
        except Exception as e:
            print(e)

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

    def __len__(self):
        try:
            if self.num_doc == 0:
                self.num_doc, self.num_sent = self._check_length(-1, -1)
            return self.num_sent if self.iter_sent else self.num_doc
        except:
            return -1

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
        check_corpus(sents)

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
                len(_counter), i_sent + 1, '%.3f' % get_process_memory(), ' '*20), flush=True)
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
        return {k: v for k, v in self._counter.items() if v >= min_count}

    def remove_covered_eojeols(self, eojeols):
        self._counter = {k: v for k, v in self._counter.items() if not (k in eojeols)}
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
        _lrgraph = {l: dict(rdict) for l, rdict in _lrgraph.items()}
        lrgraph = LRGraph(lrgraph=_lrgraph, l_max_length=l_max_length, r_max_length=r_max_length)
        return lrgraph

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, 'w', encoding='utf-8') as f:
            for eojeol, count in sorted(self._counter.items(), key=lambda x: (-x[1], x[0])):
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
    """L-R graph

    Args:
        dict_l_to_r (dict or None, optional) : predefined, one-way L -> R graph
        sents (list of str, optional) : sentences
        l_max_length (int, optional) : maximum length of L parts
        r_max_length (int, optional) : maximum length of R parts
        verbose (Boolean, optional) : if True, it shows progress

    Examples::
        # TBD
    """
    def __init__(self, dict_l_to_r=None, sents=None, l_max_length=10, r_max_length=9, verbose=False):
        assert type(l_max_length) == int and l_max_length > 1
        assert type(r_max_length) == int and r_max_length > 0
        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.verbose = verbose

        if (dict_l_to_r is not None) and (sents is not None):
            raise ValueError('LRGraph is already defined')
        if (dict_l_to_r is None) and (sents is not None):
            dict_l_to_r = self._construct_dict_graph(sents)
        if dict_l_to_r is not None:
            self._lr, self._rl = self._to_bidirectional_graph(dict_l_to_r)
        else:
            self._lr, self._rl = {}, {}

        self._lr_origin = {l: {r: c for r, c in rdict.items()}
                           for l, rdict in self._lr.items()}

    def _construct_dict_graph(self, sents):
        if not self.verbose:
            sent_iterator = sents
        else:
            sent_iterator = tqdm(sents, desc='[LRGraph] construct dict graph ... ', total=len(sents))
        lrgraph = defaultdict(lambda: defaultdict(int))
        for sent in sent_iterator:
            for word in sent.split():
                word = word.strip()
                for e in range(1, min(len(word), self.l_max_length) + 1):
                    l, r = word[:e], word[e:]
                    if len(r) > self.r_max_length:
                        continue
                    lrgraph[l][r] += e
        lrgraph = {l: dict(rdict) for l, rdict in lrgraph.items()}
        return lrgraph

    def _to_bidirectional_graph(self, lrgraph):
        rlgraph = defaultdict(lambda: defaultdict(int))
        for l, rdict in lrgraph.items():
            for r, c in rdict.items():
                if not r:
                    continue
                rlgraph[r][l] += c
        rlgraph = {r: dict(ldict) for r, ldict in rlgraph.items()}
        lrgraph = {l: dict(rdict) for l, rdict in lrgraph.items()}
        return lrgraph, rlgraph

    def reset_lrgraph(self):
        """Reset LRgraph with LRGraph._lr_origin"""

        if not hasattr(self, '_lr_origin') or (self._lr_origin is None):
            raise ValueError('This LRGraph instance can not be reset because it has no `_lr_origin`')
        self._lr, self._rl = self._to_bidirectional_graph(
            {l: {r: c for r, c in rdict.items()}
             for l, rdict in self._lr_origin.items()}
        )

    def add_lr_pair(self, l, r, count=1):
        """Add (L, R) pair with count
        if the len(l) <= `l_max_length` and len(r) <= `r_max_length`

        Args:
            l (str) : L part substring
            r (str) : R part substring
            count (int) : (l, r) pair count

        Examples::
            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=3)
            >>> lrgraph.add_lr_pair('abc', 'de')
            >>> print(lrgraph._lr)  # {'abc': {'de': 1}}
            >>> lrgraph.add_lr_pair('abcd', 'de')
            >>> print(lrgraph._lr)  # {'abc': {'de': 1}}
        """
        if (len(l) > self.l_max_length) or (len(r) > self.r_max_length):
            return
        rdict = self._lr.get(l, {})
        rdict[r] = rdict.get(r, 0) + count
        self._lr[l] = rdict
        if r:
            ldict = self._rl.get(r, {})
            ldict[l] = ldict.get(l, 0) + count
            self._rl[r] = ldict

    def add_eojeol(self, eojeol, count=1):
        """Add all possible (L, R) pair from eojeol and count

        Args:
            eojeol (str) : eojeol string
            count (int, optional, defaults to 1) : eojeol count

        Examples::
            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=3)
            >>> lrgraph.add_eojeol('abcde')
            >>> print(lrgraph._lr)  # {'ab': {'cde': 1}, 'abc': {'de': 1}}
            >>> lrgraph.add_eojeol('abcd', count=3)
            >>> print(lrgraph._lr)  # {'ab': {'cde': 4}, 'abc': {'de': 4}}

            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=4)
            >>> lrgraph.add_eojeol('abcde')
            >>> print(lrgraph._lr)  # {'a': {'bcde': 1}, 'ab': {'cde': 1}, 'abc': {'de': 1}}
            >>> lrgraph.add_eojeol('abcde', count=3)
            >>> print(lrgraph._lr)  # {'a': {'bcde': 4}, 'ab': {'cde': 4}, 'abc': {'de': 4}}
        """
        for i in range(1, len(eojeol) + 1):
            l, r = eojeol[:i], eojeol[i:]
            self.add_lr_pair(l, r, count)

    def discount_lr_pair(self, l, r, count=1):
        """Discount (L, R) pair count

        Args:
            l (str) : L part substring
            r (str) : R part substring
            count (int) : (l, r) pair count

        Examples::
            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=4)
            >>> lrgraph.add_eojeol('abcde', count=4)
            >>> print(lrgraph._lr)  # {'a': {'bcde': 4}, 'ab': {'cde': 4}, 'abc': {'de': 4}}
            >>> print(lrgraph._rl)  # {'bcde': {'a': 4}, 'cde': {'ab': 4}, 'de': {'abc': 4}}
            >>> lrgraph.discount_lr_pair('ab', 'cde', 3)
            >>> print(lrgraph._lr)  # {'a': {'bcde': 4}, 'ab': {'cde': 1}, 'abc': {'de': 4}}
            >>> print(lrgraph._rl)  # {'bcde': {'a': 4}, 'cde': {'ab': 1}, 'de': {'abc': 4}}
        """
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

    def discount_eojeol(self, eojeol, count=1):
        """Discount (L, R) pair count

        Args:
            l (str) : L part substring
            r (str) : R part substring
            count (int) : (l, r) pair count

        Examples::
            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=4)
            >>> lrgraph.add_eojeol('abcde', count=4)
            >>> print(lrgraph._lr)  # {'a': {'bcde': 4}, 'ab': {'cde': 4}, 'abc': {'de': 4}}
            >>> print(lrgraph._rl)  # {'bcde': {'a': 4}, 'cde': {'ab': 4}, 'de': {'abc': 4}}
            >>> lrgraph.discount_lr_pair('ab', 'cde', 3)
            >>> print(lrgraph._lr)  # {'a': {'bcde': 4}, 'ab': {'cde': 1}, 'abc': {'de': 4}}
            >>> print(lrgraph._rl)  # {'bcde': {'a': 4}, 'cde': {'ab': 1}, 'de': {'abc': 4}}
        """
        for i in range(1, len(eojeol) + 1):
            l, r = eojeol[:i], eojeol[i:]
            self.discount_lr_pair(l, r, count)

    def get_r(self, l, topk=10):
        """Get R parts conrresponding given `l`

        Args:
            l (str) : L part substring
            topk (int) : the number of most frequent R parts

        Returns:
            rlist (list of tuple (str, int)) : [(R, count), ... ]

        Examples::
            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=4)
            >>> lrgraph.add_eojeol('이것은', 1)
            >>> lrgraph.add_eojeol('이것도', 2)
            >>> lrgraph.add_eojeol('이것이', 3)
            >>> lrgraph.add_eojeol('이것을', 4)
            >>> lrgraph.get_r('이것', topk=3) == [('을', 4), ('이', 3), ('도', 2)]
            >>> lrgraph.get_r('이것', topk=2) == [('을', 4), ('이', 3)]
            >>> lrgraph.get_r('이것', topk=-1) == [('을', 4), ('이', 3), ('도', 2), ('은', 1)]
        """
        rlist = sorted(self._lr.get(l, {}).items(), key=lambda x: -x[1])
        if topk > 0:
            rlist = rlist[: topk]
        return rlist

    def get_l(self, r, topk=10):
        """Get L parts conrresponding given `r`

        Args:
            r (str) : R part substring
            topk (int) : the number of most frequent R parts

        Returns:
            llist (list of tuple (str, int)) : [(l, count), ... ]

        Examples::
            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=4)
            >>> lrgraph.add_eojeol('너의', 1)
            >>> lrgraph.add_eojeol('나의', 2)
            >>> lrgraph.add_eojeol('모두의', 3)
            >>> lrgraph.add_eojeol('시작의', 4)
            >>> assert lrgraph.get_l('의', topk=3) == [('시작', 4), ('모두', 3), ('나', 2)]
            >>> assert lrgraph.get_l('의', topk=2) == [('시작', 4), ('모두', 3)]
            >>> assert lrgraph.get_l('의', topk=-1) == [('시작', 4), ('모두', 3), ('나', 2), ('너', 1)]
        """
        llist = sorted(self._rl.get(r, {}).items(), key=lambda x: -x[1])
        if topk > 0:
            llist = llist[:topk]
        return llist

    def freeze(self):
        """Freeze current L-R graph."""
        self._lr_origin = copy.deepcopy(self._lr)

    def to_EojeolCounter(self, reset_lrgraph=False):
        """Transform LRGraph to soynlp.utils.EojeolCounter

        Args:
            reset_lrgraph (Boolean, optional) :
                If True, EojeolCounter is defined from `self._lr_origin` else `self._lr`

        Returns:
            eojeol_counter (~soynlp.utils.EojeolCounter)

        Examples::
            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=4)
            >>> lrgraph.add_eojeol('너의', 1)
            >>> lrgraph.add_eojeol('나의', 2)
            >>> lrgraph.add_eojeol('모두의', 3)
            >>> lrgraph.add_eojeol('시작의', 4)
            >>> print(sorted(lrgraph.to_EojeolCounter().items(), key=lambda x:x[1]))
            $ [('너의', 1), ('나의', 2), ('모두의', 3), ('시작의', 4)]
        """
        lr = self._lr_origin if reset_lrgraph else self._lr
        counter = {}
        for l, rdict in lr.items():
            for r, count in rdict.items():
                counter[l + r] = count
        eojeol_counter = EojeolCounter(None)
        eojeol_counter._counter = counter
        eojeol_counter._count_sum = sum(counter.values())
        return eojeol_counter

    def save(self, path):
        """Save LRGraph as text file
        The format of line in the text ls (l, r, count) separated by white-space

            이 것은 1
            이것 은 1
            이것은  1

        Args:
            path (str) : file path

        Examples::
            >>> lrgraph = LRGraph(l_max_length=3, r_max_length=4)
            >>> lrgraph.add_eojeol('너의', 1)
            >>> lrgraph.add_eojeol('나의', 2)
            >>> lrgraph.save('./path/to/lrgraph.txt')
        """
        dirname = os.path.dirname(os.path.abspath(path))
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, 'w', encoding='utf-8') as f:
            for l, rdict in sorted(self._lr.items()):
                for r, c in sorted(rdict.items()):
                    f.write('{} {} {}\n'.format(l, r, c))

    def load(self, path):
        """Load LRGraph from text file
        The format of line in the text ls (l, r, count) separated by white-space

            이 것은 1
            이것 은 1
            이것은  1

        Args:
            path (str) : file path

        Examples::
            >>> lrgraph = LRGraph()
            >>> lrgraph.load('./path/to/lrgraph.txt')
        """
        self._lr_origin = {}
        with open(path, encoding='utf-8') as f:
            l = ''
            rdict = {}
            for ln, line in enumerate(f):
                cols = line.split()
                if not (cols[0] == l):
                    if rdict:
                        self._lr_origin[l] = rdict
                        rdict = {}
                l = cols[0]
                if len(cols) == 2:
                    rdict[''] = int(cols[-1])
                elif len(cols) == 3:
                    rdict[cols[1]] = int(cols[-1])
                else:
                    raise ValueError(f'[LRGraph] Failed to parsing line (LN: {ln+1}) {line} in {path}')
            if rdict:
                self._lr_origin[l] = rdict
        self._lr, self._rl = self._to_bidirectional_graph(
            {l: {r: c for r, c in rdict.items()}
             for l, rdict in self._lr_origin.items()})
