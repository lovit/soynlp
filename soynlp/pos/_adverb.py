from soynlp.utils.utils import installpath

def load_default_adverbs(path=None):
    if path is None:
        path = '%s/postagger/dictionary/default/Adverb/adverb.txt' % installpath
    with open(path, encoding='utf-8') as f:
        words = {word.strip().split()[0] for word in f}
    return words

def stem_to_adverb(stems, suffixs=None):
    if isinstance(stems, str):
        stems = [stems]
    if suffixs is None:
        suffixs = '히'
    return {stem[:-1] + suffix for stem in stems for suffix in suffixs if stem[-1] == '하'}