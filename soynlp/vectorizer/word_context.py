import logging
from collections import defaultdict
from collections.abc import Callable

from scipy.sparse import csr_matrix

from soynlp.utils import get_process_memory

logger = logging.getLogger(__name__)


def sent_to_word_contexts_matrix(
    sents: list[str],
    windows: int = 3,
    min_tf: int = 10,
    tokenizer: Callable[[str], list[str]] = lambda x: x.split(),
    dynamic_weight: bool = False,
    verbose: bool = True,
) -> tuple[csr_matrix, list[str]]:
    """Create (word, contexts) co-occurrence matrix.

    Args:
        dynamic_weight: Use dynamic weight if True.
            co-occurrence weight = [1, (w-1)/w, (w-2)/w, ... 1/w]
    """
    logger.info("Create (word, contexts) matrix")

    vocab2idx, idx2vocab = _scanning_vocabulary(sents, min_tf, tokenizer)

    word2contexts = _word_context(sents, windows, tokenizer, dynamic_weight, vocab2idx)

    x = _encode_as_matrix(word2contexts, vocab2idx)

    logger.info("  - done")
    return x, idx2vocab


def _scanning_vocabulary(
    sents: list[str],
    min_tf: int,
    tokenizer: Callable[[str], list[str]],
) -> tuple[dict[str, int], list[str]]:
    word_counter: dict[str, int] = defaultdict(int)

    i_sent = 0
    for i_sent, sent in enumerate(sents):
        if i_sent % 1000 == 0:
            _log_status("  - counting word frequency", i_sent)

        words = tokenizer(sent)
        for word in words:
            word_counter[word] += 1

    _log_status("  - counting word frequency", i_sent)

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
) -> dict[str, dict[str, float]]:
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
