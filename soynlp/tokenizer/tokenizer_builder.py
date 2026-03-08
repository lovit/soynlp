import logging
from collections import defaultdict

from soynlp.utils import get_process_memory

logger = logging.getLogger(__name__)


class EojeolPatternTrainer:
    """Corpus-based custom tokenizer trainer using LR-graph construction and HITS ranking."""

    def __init__(
        self,
        max_left_length: int = 10,
        max_right_length: int = 6,
        min_frequency: int = 10,
        verbose: bool = True,
    ):
        self.max_left_length = max_left_length
        self.max_right_length = max_right_length
        self.min_frequency = min_frequency
        self.verbose = verbose
        self.lrgraph: dict | None = None
        self.rlgraph: dict | None = None
        self.wordset_l: set[str] | None = None
        self.wordset_r: set[str] | None = None

    def train(self, sents: list[str], wordset_l: set[str] | None = None, wordset_r: set[str] | None = None):
        if (not wordset_l) or (not wordset_r):
            wordset_l, wordset_r = self._scan_vocabulary(sents)
        self.lrgraph, self.rlgraph = self._build_graph(sents, wordset_l, wordset_r)

    def _scan_vocabulary(self, sents: list[str]) -> tuple[set[str], set[str]]:
        """Scan subtoken frequencies and filter by min_frequency."""
        n_sents = len(sents)
        ckpt = max(1, n_sents // 40)

        wordset_l: dict[str, int] = defaultdict(int)
        wordset_r: dict[str, int] = defaultdict(int)

        for i, sent in enumerate(sents):
            for token in sent.split(" "):
                if not token:
                    continue
                token_len = len(token)
                for j in range(1, min(self.max_left_length, token_len) + 1):
                    wordset_l[token[:j]] += 1
                for j in range(1, min(self.max_right_length, token_len)):
                    wordset_r[token[-j:]] += 1
            if i % ckpt == 0:
                pct = 100.0 * i / n_sents
                logger.info("scanning: %.1f%% (%.3f Gb)", pct, get_process_memory())

        result_l = {w for w, f in wordset_l.items() if f >= self.min_frequency}
        result_r = {w for w, f in wordset_r.items() if f >= self.min_frequency}
        logger.info(
            "scanning completed. (L,R) has (%d, %d) tokens. memory = %.3f Gb",
            len(result_l),
            len(result_r),
            get_process_memory(),
        )

        return result_l, result_r

    def _build_graph(self, sents: list[str], wordset_l: set[str], wordset_r: set[str]) -> tuple[dict, dict]:
        self.wordset_l = wordset_l
        self.wordset_r = wordset_r
        self.wordset_r.add("")
        n_sents = len(sents)
        ckpt = max(1, n_sents // 40)

        lrgraph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        rlgraph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for i, sent in enumerate(sents):
            for token in sent.split():
                if not token:
                    continue
                token_len = len(token)
                for j in range(1, min(self.max_left_length, token_len) + 1):
                    l = token[:j]
                    r = token[j:]
                    if (l not in wordset_l) or (r not in wordset_r):
                        continue
                    lrgraph[l][r] += 1
                    rlgraph[r][l] += 1

            if i % ckpt == 0:
                pct = 100.0 * i / n_sents
                logger.info("building lr-graph: %.1f%% (%.3f Gb)", pct, get_process_memory())

        logger.info("building lr-graph completed. memory = %.3f Gb", get_process_memory())

        lrgraph_dict = {l: dict(rdict) for l, rdict in lrgraph.items()}
        rlgraph_dict = {r: dict(ldict) for r, ldict in rlgraph.items()}
        return lrgraph_dict, rlgraph_dict

    def save(self, fname: str):
        with open(fname, "w", encoding="utf-8") as f:
            verbose_flag = 1 if self.verbose else 0
            f.write(f"{self.max_left_length} {self.max_right_length} {self.min_frequency} {verbose_flag}\n")
            f.write("# lrgraph\n")
            for l, rdict in self.lrgraph.items():  # type: ignore[union-attr]
                f.write(f"> {l} ({sum(rdict.values())})\n")
                for r, freq in sorted(rdict.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {r}: {freq}\n")
            f.write("\n# rlgraph\n")
            for r, ldict in self.rlgraph.items():  # type: ignore[union-attr]
                f.write(f"> {r} ({sum(ldict.values())})\n")
                for l, freq in sorted(ldict.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {l}: {freq}\n")

    def load(self, fname: str):
        with open(fname, encoding="utf-8") as f:
            param = next(f).strip()
            args = param.split()
            try:
                int_args = [int(a) for a in args]
                if len(int_args) != 4:
                    raise ValueError(f"Expected 4 parameters, got {len(int_args)}")
                self.max_left_length = int_args[0]
                self.max_right_length = int_args[1]
                self.min_frequency = int_args[2]
                self.verbose = int_args[3] == 1
            except Exception as e:
                raise ValueError(f"First line should be parameter info: {e}") from e

            lrgraph: dict[str, dict[str, int]] = defaultdict(dict)
            rlgraph: dict[str, dict[str, int]] = defaultdict(dict)

            load_type = next(f).strip()
            if load_type != "# lrgraph":
                raise ValueError(f"Cannot find lrgraph data: {load_type}")

            key1 = None
            for row in f:
                row = row.rstrip("\n")
                if not row:
                    continue
                if row == "# rlgraph":
                    break

                if row[:2] == "> ":
                    key1 = row[2 : row.rindex("(")].strip()
                    continue
                if row[:4] == "  - ":
                    key2, freq = row[4:].split(": ")
                    lrgraph[key1][key2] = int(freq)  # type: ignore[index]

            self.wordset_l = set(lrgraph.keys())

            for row in f:
                row = row.rstrip("\n")
                if not row:
                    continue
                if row[:2] == "> ":
                    key1 = row[2 : row.rindex("(")].strip()
                    continue
                if row[:4] == "  - ":
                    key2, freq = row[4:].split(": ")
                    rlgraph[key1][key2] = int(freq)  # type: ignore[index]

            self.wordset_r = set(rlgraph.keys())
            self.lrgraph = dict(lrgraph)
            self.rlgraph = dict(rlgraph)

    def train_hits(
        self,
        lrgraph: dict | None = None,
        rlgraph: dict | None = None,
        sum_of_rank: int = 10000,
        decaying_factor: float = 0.9,
        max_iter: int = 10,
        tolerance: float = 0.0001,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Train HITS algorithm on LR-graph to rank L and R tokens."""

        def normalize(g: dict, sum_of_rank: int, df: float) -> dict:
            factor = df * sum_of_rank / sum(g.values())
            restart = (1 - df) * sum_of_rank / len(g)
            return {word: (factor * rank + restart) for word, rank in g.items() if word != ""}

        if lrgraph is None:
            lrgraph, rlgraph = self.lrgraph, self.rlgraph

        rank = sum_of_rank / len(lrgraph)  # type: ignore[arg-type]
        rank_l = {l: rank for l in lrgraph}  # type: ignore[union-attr]
        rank = sum_of_rank / len(rlgraph)  # type: ignore[arg-type]
        rank_r = {r: rank for r in rlgraph if r != ""}  # type: ignore[union-attr]

        for n_iter in range(max_iter):
            next_rank_l: dict[str, float] = {}
            for l, rdict in lrgraph.items():  # type: ignore[union-attr]
                sum_rrank = sum(freq * rank_r.get(r, 0) for r, freq in rdict.items() if r != "")
                next_rank_l[l] = sum_rrank
            next_rank_l = normalize(next_rank_l, sum_of_rank, decaying_factor)

            next_rank_r: dict[str, float] = {}
            for r, ldict in rlgraph.items():  # type: ignore[union-attr]
                if r == "":
                    continue
                sum_lrank = sum(freq * rank_l.get(l, 0) for l, freq in ldict.items())
                next_rank_r[r] = sum_lrank
            next_rank_r = normalize(next_rank_r, sum_of_rank, decaying_factor)

            logger.info("train hits ... %d in %d", n_iter + 1, max_iter)

            diff = sum(abs(rank - next_rank_l.get(w, 0)) for w, rank in rank_l.items())
            diff += sum(abs(rank - next_rank_r.get(w, 0)) for w, rank in rank_r.items())
            rank_l = next_rank_l
            rank_r = next_rank_r
            if diff < (sum_of_rank * tolerance):
                logger.info("graph was converged at %d iteration", n_iter + 1)
                break

        logger.info("computation was done at %d iteration", n_iter + 1)

        return rank_l, rank_r
