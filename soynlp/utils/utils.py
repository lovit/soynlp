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
    dirname = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print(f'created {dirname}')


def most_similar(query, vector, item_to_idx, idx_to_item, topk=10):
    """Find most closest rows
    Args
        query (str) : String type query word
        vector (numpy.ndarray or scipy.sparse.matrix) : Vector representation of row
        item_to_idx (dict) : Mapper from str type item to int type index
        idx_to_item (list) : Mapper from int type index to str type item
        topk (int) : Maximum number of similar items.
            If set top as negative value, it returns similarity with all words

    Returns:
        similars (list of tuple) :
            List contains tuples (item, cosine similarity)
            Its length is topk
    """
    q = item_to_idx.get(query, -1)
    if q == -1:
        return []
    qvec = vector[q].reshape(1, -1)
    dist = pairwise_distances(qvec, vector, metric='cosine')[0]
    sim_idxs = dist.argsort()
    if topk > 0:
        sim_idxs = sim_idxs[:topk + 1]
    similars = [(idx_to_item[idx], 1 - dist[idx]) for idx in sim_idxs if idx != q]
    return similars


def check_corpus(corpus):
    """
    Args:
        corpus (list of str like)

    Returns:
        flag (Boolean)
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
    """Dataset class
    It assumes that a line represents a document.
    And each sentence in a document are separated with double-space.

    Args:
        corpus_path (str) : text file path
        num_doc (int) : number of sample documents, defaults to -1 (all documents)
        num_sent (int) : number of sample sentences, defaults to -1 (all sentences)
        iter_sent (Boolean) : if True, it yields sentence else yields document
        skip_header (int) : number of first lines to be skiped
        verbose (Boolean) : if True, it shows progress
    """
    def __init__(self, corpus_path, num_doc=-1, num_sent=-1, iter_sent=False, skip_header=0, verbose=False):
        self.corpus_path = corpus_path
        self.num_doc = 0
        self.num_sent = 0
        self.iter_sent = iter_sent
        self.skip_header = skip_header
        if (num_doc > 0) or (num_sent > 0):
            self.num_doc, self.num_sent = self._sample_first_lines(num_doc, num_sent)
        self.verbose = verbose

    def _sample_first_lines(self, num_doc, num_sent):
        num_sent_ = 0
        with open(self.corpus_path, encoding="utf-8") as f:
            # skip head
            try:
                for _ in range(self.skip_header):
                    next(f)
            except StopIteration:
                return 0, 0

            # check length
            for doc_idx, doc in enumerate(f):
                if (num_doc > 0) and (doc_idx >= num_doc):
                    return doc_idx, num_sent_
                sents = doc.split('  ')
                sents = [sent for sent in sents if sent.strip()]
                num_sent_ += len(sents)
                if (num_sent > 0) and (num_sent_ > num_sent):
                    return doc_idx + 1, min(num_sent, num_sent_)

        return doc_idx + 1, num_sent_

    def __iter__(self):
        with open(self.corpus_path, encoding="utf-8") as f:
            # skip head
            try:
                for _ in range(self.skip_header):
                    next(f)
            except StopIteration:
                return None

            # set iterator
            if self.verbose:
                if self.iter_sent:
                    line_iterator = tqdm(f, desc='[DoublespaceLineCorpus] iter sent ... ')
                else:
                    line_iterator = tqdm(f, desc='[DoublespaceLineCorpus] iter doc ... ')
            else:
                line_iterator = f

            # iteration
            num_sent, stop_doc_iter = 0, False
            for doc_idx, doc in enumerate(line_iterator):
                if stop_doc_iter:
                    break
                # yield doc
                if not self.iter_sent:
                    yield doc.strip()
                    if (self.num_doc > 0) and ((doc_idx + 1) >= self.num_doc):
                        stop_doc_iter = True
                    continue
                # yield sents
                for sent in doc.split('  '):
                    if (self.num_sent > 0) and (num_sent >= self.num_sent):
                        stop_doc_iter = True
                        break
                    sent = sent.strip()
                    if sent:
                        yield sent
                        num_sent += 1

    def __len__(self):
        if self.num_doc == 0:
            self.num_doc, self.num_sent = self._sample_first_lines(-1, -1)
        return self.num_sent if self.iter_sent else self.num_doc


class EojeolCounter:
    """
    Args:
        sents (list of str like) : sentence list
        min_count (int) : minimum frequency of eojeol
        max_length (int) : maximum length of eojeol
        filtering_checkpoint (int) : it drops eojeols which appear less than `min_count` for every `filtering_checkpoint`
        verbose (Boolean) : if True, it shows progress
        preprocess (callable) : sentence preprocessing function
            Defaults to lambda x: x

    Examples::
        >>> sents = ['이것은 어절 입니다', '이것은 예문 입니다', '이것도 예문 이고요']
        >>> eojeol_counter = EojeolCounter(sents=sents)
        >>> print(eojeol_counter.items())
        $ dict_items([('이것은', 2), ('어절', 1), ('입니다', 2), ('예문', 2), ('이것도', 1), ('이고요', 1)])

        >>> lrgraph = eojeol_counter.to_lrgraph()
        >>> lrgraph.get_r('이것')  # [('은', 2), ('도', 1)]
    """
    def __init__(self, sents=None, min_count=1, max_length=15, filtering_checkpoint=0, verbose=False, preprocess=None):
        self.min_count = min_count
        self.max_length = max_length
        self.filtering_checkpoint = filtering_checkpoint
        self.verbose = verbose

        if preprocess is None:
            def base_preprocessing(x):
                return x
            preprocess = base_preprocessing
        self.preprocess = preprocess

        if sents is not None:
            self._counter = self._counting_from_sents(sents)
        else:
            self._counter = {}

    @property
    def count_sum(self):
        return sum(self._counter.values())

    def __getitem__(self, eojeol):
        return self._counter.get(eojeol, 0)

    def __len__(self):
        return len(self._counter)

    def _counting_from_sents(self, sents):
        check_corpus(sents)
        if self.verbose:
            sent_iterator = tqdm(sents, desc='[EojeolCounter] counting ... ', total=len(sents))
        else:
            sent_iterator = sents
        counter = {}
        for i_sent, sent in enumerate(sent_iterator):
            sent = self.preprocess(sent)
            if (self.filtering_checkpoint > 0) and ((i_sent + 1) % self.filtering_checkpoint == 0):
                counter = {eojeol: count for eojeol, count in counter.items() if count >= self.min_count}
            for eojeol in sent.split():
                if (not eojeol) or (len(eojeol) > self.max_length):
                    continue
                counter[eojeol] = counter.get(eojeol, 0) + 1
        counter = {eojeol: count for eojeol, count in counter.items() if count >= self.min_count}
        return counter

    def remove_eojeols(self, eojeols):
        """Remove eojeols

        Args:
            eojeols (set of str)

        Returns:
            EojeolCounter (self)
        """
        if isinstance(eojeols, str):
            eojeols = {eojeols}
        self._counter = {k: v for k, v in self._counter.items() if not (k in eojeols)}
        return self

    def get_eojeol_count(self, eojeol):
        """Return eojeol count

        Args:
            eojeol (str) : eojeol string

        Returns:
            count (int) : if no exist, it returns 0
        """
        return self._counter.get(eojeol, 0)

    def items(self):
        """Return {key: value} items"""
        return self._counter.items()

    def to_lrgraph(self, max_l_length=10, max_r_length=9, ignore_one_syllable=False):
        """Transform EojeolCounter to LRGraph

        Args:
            max_l_length (int) : maximum length of L parts
            max_r_length (int) : maximum length of R parts
            ignore_one_syllable (Boolean) : If True, it ignores one syllable eojeol.

        Returns:
            lrgraph (~soynlp.utils.LRGraph)

        Examples::
            >>> sents = ['이것은 어절 입니다']
            >>> eojeol_counter = EojeolCounter(sents)
            >>> lrgraph = eojeol_counter.to_lrgraph()
            >>> lrgraph.get_r('이것')  # [('은', 1)]
        """
        return self._to_lrgraph(self._counter, max_l_length, max_r_length)

    def _to_lrgraph(self, counter, max_l_length=10, max_r_length=9, ignore_one_syllable=False):
        l2r = defaultdict(lambda: defaultdict(int))
        for eojeol, count in counter.items():
            if ignore_one_syllable and len(eojeol) == 1:
                continue
            for e in range(1, min(max_l_length, len(eojeol)) + 1):
                l, r = eojeol[:e], eojeol[e:]
                if len(r) > max_r_length:
                    continue
                l2r[l][r] += count
        l2r = {l: dict(rdict) for l, rdict in l2r.items()}
        lrgraph = LRGraph(dict_l_to_r=l2r, max_l_length=max_l_length, max_r_length=max_r_length)
        return lrgraph

    def save(self, path):
        """Save EojeolCounter from text file
        The format of line in the text ls (eojeol, count) separated by white-space

            이것은 1
            어절 1
            입니다 1

        Args:
            path (str) : file path

        Examples::
            >>> sents = ['이것은 어절 입니다']
            >>> eojeol_counter = EojeolCounter(sents)
            >>> eojeol_counter.save('./path/to/eojeol_counter.txt')
        """
        check_dirs(path)
        with open(path, 'w', encoding='utf-8') as f:
            for eojeol, count in sorted(self._counter.items(), key=lambda x: (-x[1], x[0])):
                f.write('{} {}\n'.format(eojeol, count))

    def load(self, path):
        """Load EojeolCounter from text file
        The format of line in the text ls (eojeol, count) separated by white-space

            이것은 1
            어절 1
            입니다 1

        Args:
            path (str) : file path

        Examples::
            >>> eojeol_counter = EojeolCounter()
            >>> eojeol_counter.load('./path/to/eojeol_counter.txt')
        """
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
        max_l_length (int, optional) : maximum length of L parts
        max_r_length (int, optional) : maximum length of R parts
        verbose (Boolean, optional) : if True, it shows progress

    Examples::
        >>> sents = ['이것은 예문 입니다', '이것도 예문 입니다']
        >>> lrgraph = LRGraph(sents=sents)
        >>> lrgraph.get_r('이것')  # [('은, 1'), ('도', 1)]
        >>> lrgraph.get_l('니다')  # [('입', 1)]

        >>> lrgraph.add_lr_pair(l='이것', r='은', count=3)
        >>> print(lrgraph._lr)  # { ...,  '이것': {'도': 2, '은': 5}, ...}

        >>> lrgraph.discount_lr_pair('이것', '은', count=1)
        >>> print(lrgraph._lr)  # { ...,  '이것': {'도': 2, '은': 4}, ...}

        >>> lrgraph.discount_eojeol('이것은', count=1)
        >>> print(lrgraph._lr)  # # { ...,  '이것': {'도': 2, '은': 3}, ...}
    """
    def __init__(self, dict_l_to_r=None, sents=None, max_l_length=10, max_r_length=9, verbose=False):
        assert type(max_l_length) == int and max_l_length > 1
        assert type(max_r_length) == int and max_r_length > 0
        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
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
                for e in range(1, min(len(word), self.max_l_length) + 1):
                    l, r = word[:e], word[e:]
                    if len(r) > self.max_r_length:
                        continue
                    lrgraph[l][r] += 1
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
        if the len(l) <= `max_l_length` and len(r) <= `max_r_length`

        Args:
            l (str) : L part substring
            r (str) : R part substring
            count (int) : (l, r) pair count

        Examples::
            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=3)
            >>> lrgraph.add_lr_pair('abc', 'de')
            >>> print(lrgraph._lr)  # {'abc': {'de': 1}}
            >>> lrgraph.add_lr_pair('abcd', 'de')
            >>> print(lrgraph._lr)  # {'abc': {'de': 1}}
        """
        if (len(l) > self.max_l_length) or (len(r) > self.max_r_length):
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
            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=3)
            >>> lrgraph.add_eojeol('abcde')
            >>> print(lrgraph._lr)  # {'ab': {'cde': 1}, 'abc': {'de': 1}}
            >>> lrgraph.add_eojeol('abcd', count=3)
            >>> print(lrgraph._lr)  # {'ab': {'cde': 4}, 'abc': {'de': 4}}

            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=4)
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
            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=4)
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
            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=4)
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
            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=4)
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
            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=4)
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
            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=4)
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
            >>> lrgraph = LRGraph(max_l_length=3, max_r_length=4)
            >>> lrgraph.add_eojeol('너의', 1)
            >>> lrgraph.add_eojeol('나의', 2)
            >>> lrgraph.save('./path/to/lrgraph.txt')
        """
        check_dirs(path)
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
