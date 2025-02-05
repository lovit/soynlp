import os
from collections import defaultdict


class LRGraph:
    def __init__(self, lrgraph: dict):
        self._lr, self._rl = self._check_lrgraph(lrgraph)
        self._lr_origin = {L: {R: freq for R, freq in R_freq.items()} for L, R_freq in self._lr.items()}

    def _check_lrgraph(self, lrgraph):
        rlgraph = defaultdict(lambda: defaultdict(int))
        for L, R_freq in lrgraph.items():
            for R, frequency in R_freq.items():
                if not R:
                    continue
                rlgraph[R][L] += frequency
        rlgraph = {R: dict(L_freq) for R, L_freq in rlgraph.items()}
        return lrgraph, rlgraph

    def reset_lrgraph(self):
        if not self._lr_origin:
            return

        self._lr, self._rl = self._check_lrgraph(
            {L: {R: freq for R, freq in R_freq.items()} for L, R_freq in self._lr_origin.items()}
        )

    def add_lr_pair(self, L: str, R: str, frequency: int = 1):
        self._lr[L][R] += frequency
        if R:
            self._rl[R][L] += frequency

    def add_eojeol(self, eojeol: str, frequency: int = 1):
        for i in range(1, len(eojeol) + 1):
            L, R = eojeol[:i], eojeol[i:]
            self.add_lr_pair(L, R, frequency)

    def remove_lr_pair(self, L: str, R: str, frequency: int = 1):
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

    def remove_eojeol(self, eojeol: str, frequency: int = 1):
        for i in range(1, len(eojeol) + 1):
            L, R = eojeol[:i], eojeol[i:]
            self.remove_lr_pair(L, R, frequency)

    def get_r(self, L: str, topk: int = 10):
        sorted_R_freq = sorted(self._lr.get(L, {}).items(), key=lambda R_freq: -R_freq[1])
        if topk > 0:
            sorted_R_freq = sorted_R_freq[:topk]
        return sorted_R_freq

    def get_l(self, R: str, topk: int = 10):
        sorted_L_freq = sorted(self._rl.get(R, {}).items(), key=lambda L_freq: -L_freq[1])
        if topk > 0:
            sorted_L_freq = sorted_L_freq[:topk]
        return sorted_L_freq

    def freeze(self):
        """Remove self._lr_origin. Be careful.
        When you excute freeze, you cannot reset_lrgraph anynore."""
        self._lr_origin = {}

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(path, "w", encoding="utf-8") as file:
            for L, R_freq in sorted(self._lr_origin.items()):
                for R, freq in sorted(R_freq.items()):
                    file.write(f"{L} {R} {freq}\n")

    def load(self, path: str):
        self._lr_origin = {}
        with open(path, encoding="utf-8") as file:
            L = ""
            R_freq = {}
            for line in file:
                sep = line.split()
                if not (sep[0] == L):
                    if R_freq:
                        self._lr_origin[L] = R_freq
                        R_freq = {}
                L = sep[0]
                if len(sep) == 2:
                    R_freq[""] = int(sep[-1])
                elif len(sep) == 3:
                    R_freq[sep[1]] = int(sep[-1])
                else:
                    raise ValueError(f"Wrong lr-graph format: {line}")
            if R_freq:
                self._lr_origin[L] = R_freq
        self._lr, self._rl = self._check_lrgraph(
            {L: {R: freq for R, freq in R_freq.items()} for L, R_freq in self._lr_origin.items()}
        )


def corpus_to_lrgraph(texts: list[str], l_max_length=10, r_max_length=9) -> LRGraph:
    assert l_max_length > 1 and isinstance(l_max_length, int)
    assert r_max_length > 0 and isinstance(r_max_length, int)
    lrgraph = defaultdict(lambda: defaultdict(int))
    for text in texts:
        for eojeol in text.split():
            eojeol = eojeol.strip()
            for e in range(1, min(len(eojeol), l_max_length) + 1):
                L, R = eojeol[:e], eojeol[e:]
                if len(R) > r_max_length:
                    continue
                lrgraph[L][R] += e
    lrgraph = {L: dict(R_frequencys) for L, R_frequencys in lrgraph.items()}
    return LRGraph(lrgraph)
