from collections import defaultdict
from collections.abc import Callable

from scipy.sparse import csr_matrix

from soynlp.utils import get_process_memory


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
    if verbose:
        print("Create (word, contexts) matrix")

    vocab2idx, idx2vocab = _scanning_vocabulary(sents, min_tf, tokenizer, verbose)

    word2contexts = _word_context(sents, windows, tokenizer, dynamic_weight, verbose, vocab2idx)

    x = _encode_as_matrix(word2contexts, vocab2idx, verbose)

    if verbose:
        print("  - done")
    return x, idx2vocab


def _scanning_vocabulary(
    sents: list[str],
    min_tf: int,
    tokenizer: Callable[[str], list[str]],
    verbose: bool,
) -> tuple[dict[str, int], list[str]]:
    word_counter: dict[str, int] = defaultdict(int)

    i_sent = 0
    for i_sent, sent in enumerate(sents):
        if verbose and i_sent % 1000 == 0:
            _print_status("  - counting word frequency", i_sent)

        words = tokenizer(sent)
        for word in words:
            word_counter[word] += 1

    if verbose:
        _print_status("  - counting word frequency", i_sent, new_line=True)

    vocab2idx = {word for word, count in word_counter.items() if count >= min_tf}
    vocab2idx_map = {word: idx for idx, word in enumerate(sorted(vocab2idx, key=lambda w: -word_counter[w]))}
    idx2vocab = [word for word, _ in sorted(vocab2idx_map.items(), key=lambda w: w[1])]

    return vocab2idx_map, idx2vocab


def _print_status(message: str, i_sent: int, new_line: bool = False) -> None:
    print(
        "\r{} from {} sents, mem={} Gb".format(message, i_sent, "%.3f" % get_process_memory()),
        flush=True,
        end="\n" if new_line else "",
    )


def _word_context(
    sents: list[str],
    windows: int,
    tokenizer: Callable[[str], list[str]],
    dynamic_weight: bool,
    verbose: bool,
    vocab2idx: dict[str, int],
) -> dict[str, dict[str, float]]:
    word2contexts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    if dynamic_weight:
        weight = [(windows - i) / windows for i in range(windows)]
    else:
        weight = [1.0] * windows

    i_sent = 0
    for i_sent, sent in enumerate(sents):
        if verbose and i_sent % 1000 == 0:
            _print_status("  - scanning (word, context) pairs", i_sent)

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

    if verbose:
        _print_status("  - scanning (word, context) pairs", i_sent, new_line=True)

    return word2contexts


def _encode_as_matrix(
    word2contexts: dict[str, dict[str, float]],
    vocab2idx: dict[str, int],
    verbose: bool,
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

    if verbose:
        print("  - (word, context) matrix was constructed. shape = {}{}".format(x.shape, " " * 20))

    return x
