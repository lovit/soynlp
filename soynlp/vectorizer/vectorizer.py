import logging
import os
from collections import Counter
from collections.abc import Callable

from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def _default_split_tokenizer(doc: str) -> list[str]:
    """Default module-level tokenizer for multiprocessing (pickle-able)."""
    return doc.split()


def _fit_chunk(args: tuple) -> tuple[dict[str, int], dict[str, int]]:
    """Worker: compute partial df/tf for a chunk of documents."""
    chunk, tokenizer = args
    df: dict[str, int] = {}
    tf: dict[str, int] = {}
    for doc in chunk:
        counter = Counter(tokenizer(doc))
        for term, freq in counter.items():
            df[term] = df.get(term, 0) + 1
            tf[term] = tf.get(term, 0) + freq
    return df, tf


class BaseVectorizer:
    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] = _default_split_tokenizer,
        min_tf: int = 0,
        max_tf: int = 99999999,
        min_df: float = 0,
        max_df: float = 1.0,
        stopwords: set[str] | None = None,
        lowercase: bool = True,
        verbose: bool = True,
    ) -> None:
        if not (0 <= min_df < 1):
            raise ValueError("min_df must be in [0, 1)")
        if not (0 < max_df <= 1):
            raise ValueError("max_df must be in (0, 1]")

        self.tokenizer = tokenizer
        self.min_tf = min_tf
        self.max_tf = max_tf
        self.min_df = min_df
        self.max_df = max_df
        self.stopwords = stopwords if stopwords else set()
        self.lowercase = lowercase
        self.verbose = verbose

        # 진행 상황 로그를 출력할 문서 처리 간격 (N 문서마다 1회 로그)
        self._check_points: int = 500

        # fit() 후 채워지는 학습 상태 변수
        self.vocabulary_: dict[str, int] = {}  # 어휘 → 인덱스 매핑 (빈도 내림차순 정렬)
        self.idx2vocab: list[str] = []  # 인덱스 → 어휘 역매핑
        self.n_vocabs: int = 0  # 어휘 크기 (= len(idx2vocab))

    def fit_transform(self, docs: list[str], n_workers: int = 1) -> csr_matrix:
        self.fit(docs, n_workers=n_workers)
        return self.transform(docs)

    def fit(self, docs: list[str], n_workers: int = 1) -> "BaseVectorizer":
        if n_workers != 1:
            df, tf = self._fit_parallel(docs, n_workers)
        else:
            df, tf = self._fit_sequential(docs)

        n_docs = len(docs)
        min_df = int(n_docs * self.min_df)
        max_df = int(n_docs * self.max_df)
        df = {term: df_t for term, df_t in df.items() if min_df <= df_t <= max_df}
        tf = {term: tf_t for term, tf_t in tf.items() if self.min_tf <= tf_t <= self.max_tf}

        vocabs = {term: tf_t for term, tf_t in tf.items() if term in df}
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(sorted(vocabs.items(), key=lambda x: -x[1]))}
        self.idx2vocab = [term for term, _ in sorted(self.vocabulary_.items(), key=lambda x: x[1])]
        self.n_vocabs = len(self.idx2vocab)

        logger.info("%d terms are recognized", self.n_vocabs)

        return self

    def _fit_sequential(self, docs: list[str]) -> tuple[dict[str, int], dict[str, int]]:
        df: dict[str, int] = {}
        tf: dict[str, int] = {}
        for i_doc, doc in enumerate(docs):
            if i_doc % self._check_points == 0:
                logger.info("scanned %d docs", i_doc)
            counter = Counter(self.tokenizer(doc))
            for term, freq in counter.items():
                df[term] = df.get(term, 0) + 1
                tf[term] = tf.get(term, 0) + freq
        logger.info("scanning was done")
        return df, tf

    def _fit_parallel(self, docs: list[str], n_workers: int) -> tuple[dict[str, int], dict[str, int]]:
        import pickle
        from multiprocessing import Pool, cpu_count

        try:
            pickle.dumps(self.tokenizer)
            tok = self.tokenizer
        except (pickle.PicklingError, AttributeError):
            logger.info("BaseVectorizer: tokenizer가 pickle 불가 — 단일 프로세스로 fit")
            return self._fit_sequential(docs)

        n = cpu_count() if n_workers == -1 else n_workers
        chunk_size = max(1, len(docs) // n)
        chunks = [docs[i : i + chunk_size] for i in range(0, len(docs), chunk_size)]
        worker_args = [(chunk, tok) for chunk in chunks]

        with Pool(processes=n) as pool:
            results = pool.map(_fit_chunk, worker_args)

        df: dict[str, int] = {}
        tf: dict[str, int] = {}
        for pDF, pTF in results:
            for k, v in pDF.items():
                df[k] = df.get(k, 0) + v
            for k, v in pTF.items():
                tf[k] = tf.get(k, 0) + v
        return df, tf

    def transform(self, docs: list[str]) -> csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        i_doc = 0
        for i_doc, doc in enumerate(docs):
            if i_doc % self._check_points == 0:
                logger.info("transformed %d docs", i_doc)

            bow = self.encode_a_doc_to_bow(doc)
            for term, count in bow.items():
                rows.append(i_doc)
                cols.append(term)
                data.append(count)

        logger.info("transforming docs to term frequency matrix was done")

        return csr_matrix((data, (rows, cols)), shape=(i_doc + 1, self.n_vocabs))

    def fit_to_file(self, docs: list[str], file_path: str, encoding: str = "utf-8") -> None:
        self.fit(docs)
        self.to_file(docs, file_path, encoding)

    def to_file(self, docs: list[str], file_path: str, encoding: str = "utf-8") -> None:
        file_path = os.path.abspath(file_path)
        n_elements = 0
        i = 0
        for i, doc in enumerate(docs):
            if i % self._check_points == 0:
                logger.info("scanning number of elements from %d docs", i)
            words = self.tokenizer(doc)
            n_elements += len({word for word in words if word in self.vocabulary_})
        n_docs = i + 1
        logger.info("scanning number of elements was done. from %d docs", n_docs)

        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "w", encoding=encoding) as f:
            f.write("%%MatrixMarket matrix coordinate integer general\n")
            f.write("%\n")
            f.write(f"{n_docs} {self.n_vocabs} {n_elements}\n")
            for i, doc in enumerate(docs):
                if i % self._check_points == 0:
                    logger.info("writing to file %.1f %%", 100 * i / n_docs)
                words = self.tokenizer(doc)
                words_count = Counter([self.vocabulary_[word] for word in words if word in self.vocabulary_])
                for j, count in words_count.items():
                    f.write(f"{i + 1} {j + 1} {count}\n")
        logger.info("writing to file was done. %d docs", n_docs)

    def __len__(self) -> int:
        return self.n_vocabs

    def encode_a_doc_to_list(self, doc: str) -> list[int]:
        return [self.vocabulary_[term] for term in self.tokenizer(doc) if term in self.vocabulary_]

    def decode_from_list(self, doc: list[int]) -> list[str]:
        return [self.idx2vocab[idx] for idx in doc if 0 <= idx < self.n_vocabs]

    def encode_a_doc_to_bow(self, doc: str) -> dict[int, int]:
        bow = Counter(self.tokenizer(doc))
        return {self.vocabulary_[term]: count for term, count in bow.items() if term in self.vocabulary_}

    def decode_from_bow(self, bow: dict[int, int]) -> dict[str, int]:
        return {self.idx2vocab[idx]: count for idx, count in bow.items() if 0 <= idx < self.n_vocabs}

    def save(self, fname: str) -> None:
        if not fname.endswith(".vocab"):
            fname += ".vocab"
        with open(fname, "w", encoding="utf-8") as f:
            for vocab in self.idx2vocab:
                f.write(f"{vocab}\n")

    def load(self, fname: str) -> None:
        if not fname.endswith(".vocab"):
            fname += ".vocab"
        with open(fname, encoding="utf-8") as f:
            self.idx2vocab = [term.strip() for term in f]
        self.vocabulary_ = {term: idx for idx, term in enumerate(self.idx2vocab)}
        self.n_vocabs = len(self.idx2vocab)

    def vocabs(self) -> list[str]:
        return [term for term in sorted(self.vocabulary_, key=lambda x: self.vocabulary_[x])]

    def _set_vocabulary(self, vocabulary_list: list[str]) -> None:
        self.idx2vocab = vocabulary_list
        self.vocabulary_ = {v: i for i, v in enumerate(self.idx2vocab)}
        self.n_vocabs = len(self.idx2vocab)
