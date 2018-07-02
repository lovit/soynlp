import os
from soynlp.utils import check_dirs

josapath = os.path.dirname(os.path.realpath(__file__))
josapath += '/frequent_enrolled_josa.txt'

with open(josapath, encoding='utf-8') as f:
    josaset = {word.strip() for word in f if word}

min_num_of_josa = 10

def write_log(path, header, words):
    check_dirs(path)
    with open(path, 'a', encoding='utf-8') as f:
        f.write('{}\n'.format(header))
        for word in sorted(words):
            f.write('{}\n'.format(word))

def _select_true_nouns(nouns, removals):
    return {word:score for word, score in nouns.items()
            if (word in removals) == False}

def detaching_features(nouns, features, logpath=None, logheader=None):

    if not logheader:
        logheader = '## Ignored noun candidates from detaching features\n'

    removals = set()

    for word in nouns:

        if len(word) <= 2:
            continue

        for e in range(2, len(word)):

            l, r = word[:e], word[e:]

            # Skip a syllable word such as 고양이, 이력서
            if len(r) <= 1:
                continue

            if (l in nouns) and (r in features):
                removals.add(word)
                break

    if logpath:
        write_log(logpath, logheader, removals)

    nouns_ = _select_true_nouns(nouns, removals)
    return nouns_, removals

def ignore_features(nouns, features, logpath=None, logheader=None):

    if not logheader:
        logheader = '## Ignored noun candidates these are same with features\n'

    removals = set()

    for word in nouns:
        if word in features:
            removals.add(word)

    if logpath:
        write_log(logpath, logheader, removals)

    nouns_ = _select_true_nouns(nouns, removals)
    return nouns_, removals

def check_N_is_NJ(nouns, lrgraph, logpath=None, logheader=None):

    if not logheader:
        logheader = '## Ignored true N+J\n'

    removals = set()
    for word in nouns:

        if not (word[-1] in josaset):
            continue

        features = lrgraph._lr_origin.get(word, {})
        features = [r for r in features if r in josaset]
        n_josa = len(features)

        if n_josa < min_num_of_josa:
            removals.add(word)

    if logpath:
        write_log(logpath, logheader, removals)

    nouns_ = _select_true_nouns(nouns, removals)
    return nouns_, removals