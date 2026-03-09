import logging
import multiprocessing
import pickle
from collections import defaultdict
from collections.abc import Callable

from scipy.sparse import csr_matrix

from soynlp.utils import get_process_memory

logger = logging.getLogger(__name__)


def _default_tokenizer(sent: str) -> list[str]:
    return sent.split()


def _scan_vocab_chunk(args: tuple) -> dict[str, int]:
    chunk, tokenizer = args
    counter: dict[str, int] = {}
    for sent in chunk:
        for word in tokenizer(sent):
            counter[word] = counter.get(word, 0) + 1
    return counter


def _word_context_chunk(args: tuple) -> list[tuple[str, str, float]]:
    chunk, windows, tokenizer, dynamic_weight, vocab2idx = args
    weight = [(windows - i) / windows for i in range(windows)] if dynamic_weight else [1.0] * windows
    pairs: list[tuple[str, str, float]] = []
    for sent in chunk:
        words = tokenizer(sent)
        if not words:
            continue
        n = len(words)
        for i, word in enumerate(words):
            if word not in vocab2idx:
                continue
            for w in range(windows):
                j = i - (w + 1)
                if j >= 0 and words[j] in vocab2idx:
                    pairs.append((word, words[j], weight[w]))
            for w in range(windows):
                j = i + w + 1
                if j < n and words[j] in vocab2idx:
                    pairs.append((word, words[j], weight[w]))
    return pairs


def sent_to_word_contexts_matrix(
    sents: list[str],
    windows: int = 3,
    min_tf: int = 10,
    tokenizer: Callable[[str], list[str]] = _default_tokenizer,
    dynamic_weight: bool = False,
    verbose: bool = True,
    n_workers: int = 1,
) -> tuple[csr_matrix, list[str]]:
    """Create (word, contexts) co-occurrence matrix.

    Args:
        dynamic_weight: Use dynamic weight if True.
            co-occurrence weight = [1, (w-1)/w, (w-2)/w, ... 1/w]
        n_workers: Number of worker processes. Falls back to 1 if tokenizer is not picklable.
    """
    logger.info("Create (word, contexts) matrix")

    if n_workers != 1:
        try:
            pickle.dumps(tokenizer)
        except (pickle.PicklingError, AttributeError):
            logger.info("sent_to_word_contexts_matrix: tokenizer가 pickle 불가 — 단일 프로세스로 실행")
            n_workers = 1

    vocab2idx, idx2vocab = _scanning_vocabulary(sents, min_tf, tokenizer, n_workers)

    word2contexts = _word_context(sents, windows, tokenizer, dynamic_weight, vocab2idx, n_workers)

    x = _encode_as_matrix(word2contexts, vocab2idx)

    logger.info("  - done")
    return x, idx2vocab


def _scanning_vocabulary(
    sents: list[str],
    min_tf: int,
    tokenizer: Callable[[str], list[str]],
    n_workers: int = 1,
) -> tuple[dict[str, int], list[str]]:
    if n_workers != 1:
        return _scanning_vocabulary_parallel(sents, min_tf, tokenizer, n_workers)

    word_counter: dict[str, int] = defaultdict(int)

    i_sent = 0
    for i_sent, sent in enumerate(sents):
        if i_sent % 1000 == 0:
            _log_status("  - counting word frequency", i_sent)

        words = tokenizer(sent)
        for word in words:
            word_counter[word] += 1

    _log_status("  - counting word frequency", i_sent)

    return _build_vocab(word_counter, min_tf)


def _scanning_vocabulary_parallel(
    sents: list[str],
    min_tf: int,
    tokenizer: Callable[[str], list[str]],
    n_workers: int,
) -> tuple[dict[str, int], list[str]]:
    chunk_size = max(1, len(sents) // n_workers)
    chunks = [sents[i : i + chunk_size] for i in range(0, len(sents), chunk_size)]
    args = [(chunk, tokenizer) for chunk in chunks]

    with multiprocessing.Pool(n_workers) as pool:
        partial_counters = pool.map(_scan_vocab_chunk, args)

    word_counter: dict[str, int] = {}
    for counter in partial_counters:
        for word, count in counter.items():
            word_counter[word] = word_counter.get(word, 0) + count

    return _build_vocab(word_counter, min_tf)


def _build_vocab(word_counter: dict[str, int], min_tf: int) -> tuple[dict[str, int], list[str]]:
    vocab2idx = {word for word, count in word_counter.items() if count >= min_tf}
    vocab2idx_map = {word: idx for idx, word in enumerate(sorted(vocab2idx, key=lambda w: -word_counter[w]))}
    idx2vocab = [word for word, _ in sorted(vocab2idx_map.items(), key=lambda w: w[1])]
    return vocab2idx_map, idx2vocab


def _log_status(message: str, i_sent: int) -> None:
    logger.info("%s from %d sents, mem=%.3f Gb", message, i_sent, get_process_memory())


def _word_context(
    sents: list[str],
    windows: int,
    tokenizer: Callable[[str], list[str]],
    dynamic_weight: bool,
    vocab2idx: dict[str, int],
    n_workers: int = 1,
) -> dict[str, dict[str, float]]:
    if n_workers != 1:
        return _word_context_parallel(sents, windows, tokenizer, dynamic_weight, vocab2idx, n_workers)

    word2contexts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    if dynamic_weight:
        weight = [(windows - i) / windows for i in range(windows)]
    else:
        weight = [1.0] * windows

    i_sent = 0
    for i_sent, sent in enumerate(sents):
        if i_sent % 1000 == 0:
            _log_status("  - scanning (word, context) pairs", i_sent)

        words = tokenizer(sent)
        if not words:
            continue

        n = len(words)

        for i, word in enumerate(words):
            if word not in vocab2idx:
                continue

            for w in range(windows):
                j = i - (w + 1)
                if j < 0 or words[j] not in vocab2idx:
                    continue
                word2contexts[word][words[j]] += weight[w]

            for w in range(windows):
                j = i + w + 1
                if j >= n or words[j] not in vocab2idx:
                    continue
                word2contexts[word][words[j]] += weight[w]

    _log_status("  - scanning (word, context) pairs", i_sent)

    return word2contexts


def _word_context_parallel(
    sents: list[str],
    windows: int,
    tokenizer: Callable[[str], list[str]],
    dynamic_weight: bool,
    vocab2idx: dict[str, int],
    n_workers: int,
) -> dict[str, dict[str, float]]:
    chunk_size = max(1, len(sents) // n_workers)
    chunks = [sents[i : i + chunk_size] for i in range(0, len(sents), chunk_size)]
    args = [(chunk, windows, tokenizer, dynamic_weight, vocab2idx) for chunk in chunks]

    with multiprocessing.Pool(n_workers) as pool:
        partial_pairs = pool.map(_word_context_chunk, args)

    word2contexts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for pairs in partial_pairs:
        for word, context, weight in pairs:
            word2contexts[word][context] += weight

    return word2contexts


def _encode_as_matrix(
    word2contexts: dict[str, dict[str, float]],
    vocab2idx: dict[str, int],
) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for word, contexts in word2contexts.items():
        word_idx = vocab2idx[word]
        for context, cooccurrence in contexts.items():
            context_idx = vocab2idx[context]
            rows.append(word_idx)
            cols.append(context_idx)
            data.append(cooccurrence)
    x = csr_matrix((data, (rows, cols)))

    logger.info("  - (word, context) matrix was constructed. shape = %s", x.shape)

    return x
