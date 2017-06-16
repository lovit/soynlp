import re
from soynlp.hangle import decompose, compose

repeatchars_patterns = [
    re.compile('(\w\w\w\w)\\1{3,}'),
    re.compile('(\w\w\w)\\1{3,}'),
    re.compile('(\w\w)\\1{3,}'),
    re.compile('(\w)\\1{3,}')
]

def normalize(sentence, n_repeat=2):
    tokens = sentence.split()
    return ' '.join(_normalize_korean_token(token, n_repeat) for token in tokens)

def _normalize_korean_token(token, n_repeat=2):
    token = _normalize_emoji(token)
    token = _remove_repeat(token, n_repeat)
    return token

def _remove_repeat(token, n_repeat=2):
    if n_repeat > 0:
        for pattern in repeatchars_patterns:
            token = pattern.sub('\\1' * n_repeat, token)
    return token

def _normalize_emoji(token):
    if len(token) <= 1:
        return token
    token_ = []
    decomposeds = [decompose(c) for c in token]
    for char, cd, nd in zip(token, decomposeds, decomposeds[1:]):
        if cd == None or nd == None:
            token_.append(char)
            continue
        # 앜ㅋㅋㅋㅋ -> 아ㅋㅋㅋㅋㅋ
        if (nd[1] == ' ') and (cd[2] == nd[0]):
            token_.append(compose(cd[0], cd[1], ' ') + nd[0])
        # ㅋ쿠ㅜㅜ -> ㅋㅋㅜㅜㅜ
        elif (cd[2] == ' ') and (nd[0] == ' ') and (cd[1] == nd[1]):
            token_.append((cd[0] + cd[1]) if cd [0] != ' ' else cd[1])
        else:
            token_.append(char)
    return ''.join(token_) + token[-1]