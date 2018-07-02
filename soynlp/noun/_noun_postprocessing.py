from soynlp.utils import check_dirs

def write_log(path, header, words):
    check_dirs(path)
    with open(path, 'a', encoding='utf-8') as f:
        f.write('{}\n'.format(header))
        for word in sorted(words):
            f.write('{}\n'.format(word))

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

    # write log for debug
    if logpath:
        write_log(logpath, logheader, removals)

    nouns_ = {word:score for word, score in nouns.items() if (word in removals) == False}
    return nouns_, removals

def ignore_features(nouns, features, logpath=None, logheader=None):

    if not logheader:
        logheader = '## Ignored noun candidates these are same with features\n'

    removals = set()
    for word in nouns:
        if word in features:
            removals.add(word)

    # write log for debug
    if logpath:
        write_log(logpath, logheader, removals)

    nouns_ = {word:score for word, score in nouns.items() if (word in removals) == False}
    return nouns_, removals

with open('frequent_enrolled_josa.txt', encoding='utf-8') as f:
    enrolled_josa = {word.strip() for word in f if word}