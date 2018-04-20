from soynlp.utils import get_process_memory
from collections import defaultdict
from scipy.sparse import csr_matrix

def sent_to_word_context_matrix(sents, windows=3, min_tf=10,
        tokenizer=lambda x:x.split(), verbose=True):

    # counting word frequency, first
    word_counter = defaultdict(int)
    for i_sent, sent in enumerate(sents):
        if verbose and i_sent % 1000 == 0:
            print('\rcounting word frequency from {} sents, mem={} Gb'.format(
                i_sent, '%.3f' % get_process_memory()), flush=True, end='')
        words = tokenizer(sent)
        for word in words:
            word_counter[word] += 1
    if verbose:
        print('\rcounting word frequency from {} sents was done. mem={} Gb'.format(
            i_sent, '%.3f' % get_process_memory()), flush=True, end='')
    
    # filtering with min_tf    
    vocabulary = {word for word, count in word_counter.items() if count >= min_tf}
    vocabulary = {word:idx for idx, word in enumerate(sorted(vocabulary, key=lambda w:-word_counter[w]))}
    idx2vocab = [word for word, _ in sorted(vocabulary.items(), key=lambda w:w[1])]
    del word_counter

    # scanning (word, context) pairs
    base2contexts = defaultdict(lambda: defaultdict(int))

    for i_sent, sent in enumerate(sents):
        if verbose and i_sent % 1000 == 0:
            print('\rscanning (word, context) pairs from {} sents, mem={} Gb'.format(
                i_sent, '%.3f' % get_process_memory()), flush=True, end='')

        words = tokenizer(sent)
        if not words:
            continue

        n = len(words)

        for i, base in enumerate(words):
            if not (base in vocabulary):
                continue

            # left_contexts
            for context in words[max(0, i-windows):i]:
                if not (context in vocabulary):
                    continue
                base2contexts[base][context] += 1

            # right_contexts
            for context in words[min(i+1, n):min(i+windows, n)+1]:
                if not (context in vocabulary):
                    continue
                base2contexts[base][context] += 1

    if verbose:
        print('\rscanning (word, context) pairs from {} sents was done. mem={} Gb'.format(
            i_sent + 1, '%.3f' % get_process_memory()), flush=True, end='')

    # Encoding dict to vectors
    rows = []
    cols = []
    data = []
    for base, contexts in base2contexts.items():
        base_idx = vocabulary[base]
        for context, cooccurrence in contexts.items():
            context_idx = vocabulary[context]
            rows.append(base_idx)
            cols.append(context_idx)
            data.append(cooccurrence)
    x = csr_matrix((data, (rows, cols)))

    if verbose:
        print('\r(word, context) matrix was constructed. shape = {}{}'.format(
            x.shape, ' '*20))

    return x, idx2vocab