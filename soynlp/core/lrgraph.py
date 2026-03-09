import copy
import os
from collections import defaultdict
from collections.abc import Iterable, Sized


class LRGraph:
    """L-R graph for Korean morpheme analysis.

    Args:
        lrgraph: dict of {L: {R: frequency}}
        max_l_length: maximum length of L parts
        max_r_length: maximum length of R parts
    """

    _lr: dict[str, dict[str, int]]
    _rl: dict[str, dict[str, int]]
    _lr_origin: dict[str, dict[str, int]]

    def __init__(self, lrgraph: dict[str, dict[str, int]], max_l_length: int = 10, max_r_length: int = 9) -> None:
        if not (isinstance(max_l_length, int) and max_l_length > 1):
            raise ValueError(f"`max_l_length` must be an integer greater than 1, got {max_l_length}")
        if not (isinstance(max_r_length, int) and max_r_length > 0):
            raise ValueError(f"`max_r_length` must be a positive integer, got {max_r_length}")
        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
        self._lr, self._rl = self._to_bidirectional_graph(lrgraph)
        self._lr_origin = {L: {R: freq for R, freq in R_freq.items()} for L, R_freq in self._lr.items()}

    def _to_bidirectional_graph(
        self, lrgraph: dict[str, dict[str, int]]
    ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
        rlgraph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for L, R_freq in lrgraph.items():
            for R, frequency in R_freq.items():
                if not R:
                    continue
                rlgraph[R][L] += frequency
        rlgraph = {R: dict(L_freq) for R, L_freq in rlgraph.items()}
        lrgraph = {L: dict(R_freq) for L, R_freq in lrgraph.items()}
        return lrgraph, rlgraph

    @classmethod
    def from_sents(
        cls, sents: Iterable[str], max_l_length: int = 10, max_r_length: int = 9, verbose: bool = False
    ) -> "LRGraph":
        if verbose:
            from tqdm import tqdm

            total = len(sents) if isinstance(sents, Sized) else None
            sent_iterator = tqdm(sents, desc="[LRGraph] construct dict graph ... ", total=total)
        else:
            sent_iterator = sents
        lrgraph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for sent in sent_iterator:
            for word in sent.split():
                word = word.strip()
                for e in range(1, min(len(word), max_l_length) + 1):
                    L, R = word[:e], word[e:]
                    if len(R) > max_r_length:
                        continue
                    lrgraph[L][R] += 1
        lrgraph_dict = {L: dict(R_freq) for L, R_freq in lrgraph.items()}
        return cls(lrgraph_dict, max_l_length, max_r_length)

    def reset_lrgraph(self) -> None:
        if not self._lr_origin:
            return
        self._lr, self._rl = self._to_bidirectional_graph(
            {L: {R: freq for R, freq in R_freq.items()} for L, R_freq in self._lr_origin.items()}
        )

    def add_lr_pair(self, L: str, R: str, frequency: int = 1) -> None:
        if (len(L) > self.max_l_length) or (len(R) > self.max_r_length):
            return
        self._lr.setdefault(L, {})[R] = self._lr.get(L, {}).get(R, 0) + frequency
        if R:
            self._rl.setdefault(R, {})[L] = self._rl.get(R, {}).get(L, 0) + frequency

    def add_eojeol(self, eojeol: str, frequency: int = 1) -> None:
        for i in range(1, len(eojeol) + 1):
            L, R = eojeol[:i], eojeol[i:]
            self.add_lr_pair(L, R, frequency)

    def remove_lr_pair(self, L: str, R: str, frequency: int = 1) -> None:
        if L in self._lr:
            R_freq = self._lr[L]
            if R in R_freq:
                R_freq[R] -= frequency
                if R_freq[R] <= 0:
                    R_freq.pop(R)
                    if len(R_freq) <= 0:
                        self._lr.pop(L)
        if R in self._rl:
            L_freq = self._rl[R]
            if L in L_freq:
                L_freq[L] -= frequency
                if L_freq[L] <= 0:
                    L_freq.pop(L)
                    if len(L_freq) <= 0:
                        self._rl.pop(R)

    def remove_eojeol(self, eojeol: str, frequency: int = 1) -> None:
        for i in range(1, len(eojeol) + 1):
            L, R = eojeol[:i], eojeol[i:]
            self.remove_lr_pair(L, R, frequency)

    def get_r(self, L: str, topk: int = 10) -> list[tuple[str, int]]:
        sorted_R_freq = sorted(self._lr.get(L, {}).items(), key=lambda R_freq: -R_freq[1])
        if topk > 0:
            sorted_R_freq = sorted_R_freq[:topk]
        return sorted_R_freq

    def get_l(self, R: str, topk: int = 10) -> list[tuple[str, int]]:
        sorted_L_freq = sorted(self._rl.get(R, {}).items(), key=lambda L_freq: -L_freq[1])
        if topk > 0:
            sorted_L_freq = sorted_L_freq[:topk]
        return sorted_L_freq

    def freeze(self) -> None:
        """Freeze current L-R graph state into _lr_origin via deepcopy."""
        self._lr_origin = copy.deepcopy(self._lr)

    def save(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, "w", encoding="utf-8") as file:
            for L, R_freq in sorted(self._lr.items()):
                for R, freq in sorted(R_freq.items()):
                    file.write(f"{L} {R} {freq}\n")

    @classmethod
    def load(cls, path: str) -> "LRGraph":
        lr_data: dict[str, dict[str, int]] = {}
        with open(path, encoding="utf-8") as file:
            L = ""
            R_freq: dict[str, int] = {}
            for line in file:
                sep = line.split()
                if not sep:
                    continue
                if not (sep[0] == L):
                    if R_freq:
                        lr_data[L] = R_freq
                        R_freq = {}
                L = sep[0]
                if len(sep) == 2:
                    R_freq[""] = int(sep[-1])
                elif len(sep) == 3:
                    R_freq[sep[1]] = int(sep[-1])
                else:
                    raise ValueError(f"Wrong lr-graph format: {line}")
            if R_freq:
                lr_data[L] = R_freq
        return cls(lr_data)


def _build_partial_counter(args: tuple[list[str], int, int]) -> dict[str, dict[str, int]]:
    """Module-level worker function: builds a partial LR counter from a chunk of texts.

    Args:
        args: tuple of (texts_chunk, l_max_length, r_max_length)

    Returns:
        dict of {L: {R: frequency}}
    """
    texts_chunk, l_max_length, r_max_length = args
    counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sent in texts_chunk:
        for word in sent.split():
            word = word.strip()
            for e in range(1, min(len(word), l_max_length) + 1):
                L, R = word[:e], word[e:]
                if len(R) > r_max_length:
                    continue
                counter[L][R] += 1
    return {L: dict(R_freq) for L, R_freq in counter.items()}


def _merge_counters(counters: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    """Merge a list of partial LR counters into one."""
    merged: dict[str, dict[str, int]] = {}
    for counter in counters:
        for L, R_freq in counter.items():
            if L not in merged:
                merged[L] = {}
            for R, freq in R_freq.items():
                merged[L][R] = merged[L].get(R, 0) + freq
    return merged


def corpus_to_lrgraph(texts: list[str], l_max_length: int = 10, r_max_length: int = 9, n_workers: int = 1) -> LRGraph:
    """Build an LRGraph from a list of texts.

    Args:
        texts: list of sentences
        l_max_length: maximum length of L parts
        r_max_length: maximum length of R parts
        n_workers: number of worker processes. Use -1 to use all CPU cores.

    Returns:
        LRGraph built from the input texts
    """
    if n_workers == -1:
        n_workers = os.cpu_count() or 1

    if n_workers <= 1:
        return LRGraph.from_sents(texts, max_l_length=l_max_length, max_r_length=r_max_length)

    # Ensure texts is a list for chunking
    if not isinstance(texts, list):
        texts = list(texts)

    # Split texts into n_workers chunks
    chunk_size = max(1, len(texts) // n_workers)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    # Build partial counters in parallel
    from multiprocessing import Pool

    worker_args = [(chunk, l_max_length, r_max_length) for chunk in chunks]
    with Pool(processes=n_workers) as pool:
        partial_counters = pool.map(_build_partial_counter, worker_args)

    # Merge all partial counters and build LRGraph
    merged = _merge_counters(partial_counters)
    return LRGraph(merged, max_l_length=l_max_length, max_r_length=r_max_length)
