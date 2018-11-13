from soynlp.utils.utils import installpath

def load_default_adverbs(path=None):
    if path is None:
        path = '%s/postagger/dictionary/default/Adverb/adverb.txt'.format(installpath)
    with open(path, encoding='utf-8') as f:
        words = {word.strip().split()[0] for word in f}
    return words

def stem_to_adverb(stems):
    return {stem[:-1]+ending for stem in stems for ending in '이히' if stem[-1] == '하'}