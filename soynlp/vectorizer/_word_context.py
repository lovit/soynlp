from soynlp.utils import get_process_memory
from collections import defaultdict
from scipy.sparse import csr_matrix

def sent_to_word_contexts_matrix(sents, windows=3, min_tf=10,
        tokenizer=lambda x:x.split(), verbose=True):

    if verbose:
        print('Create (word, contexts) matrix')

    vocab2idx, idx2vocab = _scanning_vocabulary(
        sents, min_tf, tokenizer, verbose)

    word2contexts = _word_context(
        sents, windows, tokenizer, verbose, vocab2idx)

    x = _encode_as_matrix(word2contexts, vocab2idx, verbose)

    if verbose:
        print('  - done')
    return x, idx2vocab

def _scanning_vocabulary(sents, min_tf, tokenizer, verbose):

    # counting word frequency, first
    word_counter = defaultdict(int)

    for i_sent, sent in enumerate(sents):

        if verbose and i_sent % 1000 == 0:
            _print_status('  - counting word frequency', i_sent)

        words = tokenizer(sent)
        for word in words:
            word_counter[word] += 1

    if verbose:
        _print_status('  - counting word frequency', i_sent, new_line=True)

    # filtering with min_tf    
    vocab2idx = {word for word, count in word_counter.items() if count >= min_tf}
    vocab2idx = {word:idx for idx, word in enumerate(
        sorted(vocab2idx, key=lambda w:-word_counter[w]))}
    idx2vocab = [word for word, _ in sorted(vocab2idx.items(), key=lambda w:w[1])]
    del word_counter

    return vocab2idx, idx2vocab

def _print_status(message, i_sent, new_line=False):
    print('\r{} from {} sents, mem={} Gb'.format(
            message, i_sent, '%.3f' % get_process_memory()),
        flush=True, end='\n' if new_line else ''
    )

def _word_context(sents, windows, tokenizer, verbose, vocab2idx):

    # scanning (word, context) pairs
    word2contexts = defaultdict(lambda: defaultdict(int))

    for i_sent, sent in enumerate(sents):

        if verbose and i_sent % 1000 == 0:
            _print_status('  - scanning (word, context) pairs', i_sent)

        words = tokenizer(sent)
        if not words:
            continue

        n = len(words)

        for i, word in enumerate(words):
            if not (word in vocab2idx):
                continue

            # left_contexts
            for context in words[max(0, i-windows):i]:
                if not (context in vocab2idx):
                    continue
                word2contexts[word][context] += 1

            # right_contexts
            for context in words[min(i+1, n):min(i+windows, n)+1]:
                if not (context in vocab2idx):
                    continue
                word2contexts[word][context] += 1

    if verbose:
        _print_status('  - scanning (word, context) pairs', i_sent, new_line=True)

    return word2contexts

def _encode_as_matrix(word2contexts, vocab2idx, verbose):

    rows = []
    cols = []
    data = []
    for word, contexts in word2contexts.items():
        word_idx = vocab2idx[word]
        for context, cooccurrence in contexts.items():
            context_idx = vocab2idx[context]
            rows.append(word_idx)
            cols.append(context_idx)
            data.append(cooccurrence)
    x = csr_matrix((data, (rows, cols)))

    if verbose:
        print('  - (word, context) matrix was constructed. shape = {}{}'.format(
            x.shape, ' '*20))

    return x