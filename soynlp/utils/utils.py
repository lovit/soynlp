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


def _count_eojeol_chunk(args: tuple) -> dict[str, int]:
    """Worker function for parallel eojeol counting. Must be module-level for pickling."""
    chunk, max_length, text_key, preprocess = args
    counter: dict[str, int] = {}
    for item in chunk:
        sent = item[text_key] if isinstance(item, dict) else item
        if preprocess is not None:
            sent = preprocess(sent)
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

    이터레이션 시 각 행을 dict로 반환한다.
    - JSONL 형식: 원본 JSON 객체 전체를 반환
    - TXT 형식: ``{text_key: line}`` 형태의 dict를 반환

    ``__len__`` 을 구현하여 ``EojeolCounter``, ``tqdm`` 등에서 진행률 표시가 가능하다.

    Args:
        corpus_path: 읽을 파일 경로. 존재하지 않으면 ``FileNotFoundError``.
        format: 파일 형식. ``"jsonl"`` 또는 ``"text"`` 중 하나.
        text_key: JSONL 또는 TXT 반환 dict에서 텍스트를 가리키는 키 이름.
            기본값 ``"text"`` — JSONL 행이 ``{"text": "...", "id": 1}`` 형태일 때 사용.
        verbose: True이면 tqdm으로 읽기 진행률을 출력한다.

    Examples::

        JSONL 파일 (각 줄: ``{"text": "안녕하세요", "id": 1}`` 형태)::

            >>> loader = CorpusLoader("corpus.jsonl", format="jsonl")
            >>> for item in loader:
            ...     print(item["text"])  # "안녕하세요"

        TXT 파일 (각 줄이 하나의 문장)::

            >>> loader = CorpusLoader("corpus.txt", format="text")
            >>> for item in loader:
            ...     print(item["text"])  # 각 줄 내용

        사용자 정의 text_key::

            >>> loader = CorpusLoader("corpus.jsonl", format="jsonl", text_key="content")
            >>> for item in loader:
            ...     print(item["content"])
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
    """문장 목록에서 어절(공백 기준 분리 단위) 빈도를 집계한다.

    집계 결과를 ``to_lrgraph()``로 LRGraph로 변환하여 명사 추출에 사용할 수 있다.

    Args:
        sents: 문장 이터러블. ``str`` 리스트, ``CorpusLoader``, 또는 이터러블이면 모두 허용.
            ``None``이면 빈 카운터로 초기화된다.
        min_count: 어절 최소 출현 횟수. 이 값 미만의 어절은 최종 결과에서 제외된다.
        max_length: 어절 최대 길이. 초과하는 어절은 무시된다.
        filtering_checkpoint: ``filtering_checkpoint`` 문장마다 ``min_count`` 미만 어절을 임시 제거한다.
            0이면 중간 정리를 하지 않는다. 메모리 절약에 유용하다.
        verbose: True이면 tqdm으로 카운팅 진행률을 출력한다.
        preprocess: 각 문장에 적용하는 전처리 함수. 기본값은 항등 함수(변환 없음).
            병렬 처리(``n_workers > 1``) 시 pickle 가능한 함수여야 한다.
        text_key: 입력이 JSONL dict 형태일 때 텍스트를 가져올 키 이름. 기본값 ``"text"``.
        n_workers: 병렬 처리 워커 수. 1이면 단일 프로세스.

    Examples::

        문장 리스트에서 생성::

            >>> sents = ['이것은 어절 입니다', '이것은 예문 입니다', '이것도 예문 이고요']
            >>> eojeol_counter = EojeolCounter(sents=sents)
            >>> print(eojeol_counter.items())
            dict_items([('이것은', 2), ('어절', 1), ('입니다', 2), ('예문', 2), ('이것도', 1), ('이고요', 1)])

        LRGraph로 변환::

            >>> lrgraph = eojeol_counter.to_lrgraph()
            >>> lrgraph.get_r('이것')  # [('은', 2), ('도', 1)]

        JSONL 파일에서 생성 (각 줄: ``{"text": "..."}`` 형태)::

            >>> loader = CorpusLoader("corpus.jsonl", format="jsonl")
            >>> eojeol_counter = EojeolCounter(sents=loader, text_key="text")
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

        self._has_custom_preprocess = preprocess is not None
        if preprocess is None:

            def base_preprocessing(x: str) -> str:
                return x

            preprocess = base_preprocessing
            self._parallel_preprocess: Callable[[str], str] | None = None
        else:
            import pickle

            try:
                pickle.dumps(preprocess)
                self._parallel_preprocess = preprocess
            except (pickle.PicklingError, AttributeError):
                self._parallel_preprocess = None
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
        if n_workers != 1 and (not self._has_custom_preprocess or self._parallel_preprocess is not None):
            return self._counting_from_sents_parallel(sents, n_workers, self._parallel_preprocess)
        if n_workers != 1:
            logger.info("EojeolCounter: custom preprocess가 pickle 불가 — 단일 프로세스로 집계")
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

    def _counting_from_sents_parallel(
        self, sents: Any, n_workers: int, preprocess: Callable[[str], str] | None = None
    ) -> dict[str, int]:
        from multiprocessing import Pool, cpu_count

        texts = list(sents)
        n = cpu_count() if n_workers == -1 else n_workers
        chunk_size = max(1, len(texts) // n)
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
        worker_args = [(chunk, self.max_length, self.text_key, preprocess) for chunk in chunks]

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
