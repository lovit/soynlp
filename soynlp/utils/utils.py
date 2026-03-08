import json
import logging
import os
from collections import defaultdict
from collections.abc import Callable, ItemsView, Iterator
from typing import Any

import psutil
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from soynlp.core.lrgraph import LRGraph

logger = logging.getLogger(__name__)

installpath = os.path.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])


def _count_eojeol_chunk(args: tuple[list, int, str]) -> dict[str, int]:
    """Worker function for parallel eojeol counting. Must be module-level for pickling."""
    chunk, max_length, text_key = args
    counter: dict[str, int] = {}
    for item in chunk:
        sent = item[text_key] if isinstance(item, dict) else item
        for eojeol in sent.split():
            if (not eojeol) or (len(eojeol) > max_length):
                continue
            counter[eojeol] = counter.get(eojeol, 0) + 1
    return counter


def get_available_memory() -> float:
    """It returns remained memory as percentage"""
    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)


def get_process_memory() -> float:
    """It returns the memory usage of current process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def check_dirs(filepath: str) -> None:
    dirname = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        logger.info(f"created {dirname}")


def most_similar(
    query: str, vector: Any, item_to_idx: dict[str, int], idx_to_item: list[str], topk: int = 10
) -> list[tuple[str, float]]:
    """Find most closest rows

    Args:
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
    dist = pairwise_distances(qvec, vector, metric="cosine")[0]
    sim_idxs = dist.argsort()
    if topk > 0:
        sim_idxs = sim_idxs[: topk + 1]
    similars = [(idx_to_item[idx], 1 - dist[idx]) for idx in sim_idxs if idx != q]
    return similars


def check_corpus(corpus: Any) -> bool:
    """
    Args:
        corpus (list of str like)

    Returns:
        flag (Boolean)
            It returns True when __len__ is implemented and the length is larger than 0
    """
    if not hasattr(corpus, "__iter__"):
        raise ValueError("Input corpus must have __iter__ such as list or soynlp.utils.CorpusLoader")
    if not hasattr(corpus, "__len__"):
        raise ValueError("Input corpus must have __len__ such as list or soynlp.utils.CorpusLoader")
    if len(corpus) <= 0:
        raise ValueError("Input corpus must be longer than 0")
    return True


class CorpusLoader:
    """JSONL/TXT 코퍼스 로더.

    Args:
        corpus_path (str) : 파일 경로
        format (str) : "jsonl" 또는 "text"
        text_key (str) : JSONL에서 텍스트 필드명 (기본: "text")
        verbose (bool) : 진행률 표시 여부

    iter 시 dict를 반환 (JSONL: 원본 dict, TXT: {text_key: line})
    __len__ 구현으로 EojeolCounter 등 기존 코드 호환

    Examples::
        >>> loader = CorpusLoader("corpus.jsonl", format="jsonl")
        >>> for item in loader:
        ...     print(item["text"])

        >>> loader = CorpusLoader("corpus.txt", format="text")
        >>> for item in loader:
        ...     print(item["text"])
    """

    def __init__(self, corpus_path: str, format: str = "jsonl", text_key: str = "text", verbose: bool = False) -> None:
        if format not in ("jsonl", "text"):
            raise ValueError(f"format must be 'jsonl' or 'text', got '{format}'")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"corpus_path not found: {corpus_path}")
        self.corpus_path = corpus_path
        self.format = format
        self.text_key = text_key
        self.verbose = verbose
        self._num_lines: int | None = None

    def _count_lines(self) -> int:
        count = 0
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def __len__(self) -> int:
        if self._num_lines is None:
            self._num_lines = self._count_lines()
        return self._num_lines

    def __iter__(self) -> Iterator[dict[str, str]]:
        with open(self.corpus_path, encoding="utf-8") as f:
            if self.verbose:
                line_iterator = tqdm(f, desc=f"[CorpusLoader] reading {self.format}", total=len(self))
            else:
                line_iterator = f

            for line in line_iterator:
                line = line.strip()
                if not line:
                    continue
                if self.format == "jsonl":
                    yield json.loads(line)
                else:
                    yield {self.text_key: line}


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

    def __init__(
        self,
        sents: Any | None = None,
        min_count: int = 1,
        max_length: int = 15,
        filtering_checkpoint: int = 0,
        verbose: bool = False,
        preprocess: Callable[[str], str] | None = None,
        text_key: str = "text",
        n_workers: int = 1,
    ) -> None:
        self.min_count = min_count
        self.max_length = max_length
        self.filtering_checkpoint = filtering_checkpoint
        self.verbose = verbose
        self.text_key = text_key

        if preprocess is None:

            def base_preprocessing(x: str) -> str:
                return x

            preprocess = base_preprocessing
        self.preprocess = preprocess

        if sents is not None:
            self._counter = self._counting_from_sents(sents, n_workers=n_workers)
        else:
            self._counter = {}

    @property
    def count_sum(self) -> int:
        return sum(self._counter.values())

    def _set_count_sum(self) -> None:
        self._count_sum = sum(self._counter.values())

    def __getitem__(self, eojeol: str) -> int:
        return self._counter.get(eojeol, 0)

    def __len__(self) -> int:
        return len(self._counter)

    def _counting_from_sents(self, sents: Any, n_workers: int = 1) -> dict[str, int]:
        check_corpus(sents)
        use_custom_preprocess = self.preprocess.__name__ != "base_preprocessing"
        if n_workers != 1 and not use_custom_preprocess:
            return self._counting_from_sents_parallel(sents, n_workers)
        if n_workers != 1 and use_custom_preprocess:
            logger.info("EojeolCounter: custom preprocess는 멀티프로세싱 미지원 — 단일 프로세스로 집계")
        if self.verbose:
            sent_iterator = tqdm(sents, desc="[EojeolCounter] counting eojeols ", total=len(sents))
        else:
            sent_iterator = sents
        counter: dict[str, int] = {}
        for i_sent, item in enumerate(sent_iterator):
            if isinstance(item, dict):
                sent = item[self.text_key]
            else:
                sent = item
            sent = self.preprocess(sent)
            if (self.filtering_checkpoint > 0) and ((i_sent + 1) % self.filtering_checkpoint == 0):
                counter = {eojeol: count for eojeol, count in counter.items() if count >= self.min_count}
            for eojeol in sent.split():
                if (not eojeol) or (len(eojeol) > self.max_length):
                    continue
                counter[eojeol] = counter.get(eojeol, 0) + 1
        counter = {eojeol: count for eojeol, count in counter.items() if count >= self.min_count}
        return counter

    def _counting_from_sents_parallel(self, sents: Any, n_workers: int) -> dict[str, int]:
        from multiprocessing import Pool, cpu_count

        texts = list(sents)
        n = cpu_count() if n_workers == -1 else n_workers
        chunk_size = max(1, len(texts) // n)
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
        worker_args = [(chunk, self.max_length, self.text_key) for chunk in chunks]

        with Pool(processes=n) as pool:
            partial_counters = pool.map(_count_eojeol_chunk, worker_args)

        merged: dict[str, int] = {}
        for partial in partial_counters:
            for eojeol, count in partial.items():
                merged[eojeol] = merged.get(eojeol, 0) + count
        return {eojeol: count for eojeol, count in merged.items() if count >= self.min_count}

    def remove_eojeols(self, eojeols: set[str] | str) -> "EojeolCounter":
        """Remove eojeols

        Args:
            eojeols (set of str)

        Returns:
            EojeolCounter (self)
        """
        if isinstance(eojeols, str):
            eojeols = {eojeols}
        self._counter = {k: v for k, v in self._counter.items() if k not in eojeols}
        return self

    def get_eojeol_count(self, eojeol: str) -> int:
        """Return eojeol count

        Args:
            eojeol (str) : eojeol string

        Returns:
            count (int) : if no exist, it returns 0
        """
        return self._counter.get(eojeol, 0)

    def items(self) -> ItemsView[str, int]:
        """Return {key: value} items"""
        return self._counter.items()

    def to_lrgraph(self, max_l_length: int = 10, max_r_length: int = 9, ignore_one_syllable: bool = False) -> LRGraph:
        """Transform EojeolCounter to LRGraph

        Args:
            max_l_length (int) : maximum length of L parts
            max_r_length (int) : maximum length of R parts
            ignore_one_syllable (Boolean) : If True, it ignores one syllable eojeol.

        Returns:
            lrgraph (~soynlp.core.lrgraph.LRGraph)

        Examples::
            >>> sents = ['이것은 어절 입니다']
            >>> eojeol_counter = EojeolCounter(sents)
            >>> lrgraph = eojeol_counter.to_lrgraph()
            >>> lrgraph.get_r('이것')  # [('은', 1)]
        """
        return self._to_lrgraph(self._counter, max_l_length, max_r_length, ignore_one_syllable)

    def _to_lrgraph(
        self, counter: dict[str, int], max_l_length: int = 10, max_r_length: int = 9, ignore_one_syllable: bool = False
    ) -> LRGraph:
        l2r = defaultdict(lambda: defaultdict(int))
        for eojeol, count in counter.items():
            if ignore_one_syllable and len(eojeol) == 1:
                continue
            for e in range(1, min(max_l_length, len(eojeol)) + 1):
                l, r = eojeol[:e], eojeol[e:]  # noqa: E741
                if len(r) > max_r_length:
                    continue
                l2r[l][r] += count
        l2r = {l: dict(rdict) for l, rdict in l2r.items()}  # noqa: E741
        return LRGraph(l2r, max_l_length=max_l_length, max_r_length=max_r_length)

    def save(self, path: str) -> None:
        """Save EojeolCounter to text file

        Args:
            path (str) : file path
        """
        check_dirs(path)
        with open(path, "w", encoding="utf-8") as f:
            for eojeol, count in sorted(self._counter.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"{eojeol} {count}\n")

    def load(self, path: str) -> None:
        """Load EojeolCounter from text file

        Args:
            path (str) : file path
        """
        self._coverage = 0.0
        self._counter = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, count = line.split()
                self._counter[word] = int(count)
        self._count_sum = sum(self._counter.values())
